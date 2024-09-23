import math
import time

import torch
import torch.utils.data as data_utils

from data import custom_collate
from modules import AverageMeter
from modules import utils
from modules.solver import get_optim
from utils_ssl import compute_req_matrices
from val import validate

logger = utils.get_logger(__name__)


def setup_training(args, net):
    optimizer, scheduler, solver_print_str = get_optim(args, net)
    if args.TENSORBOARD:
        from tensorboardX import SummaryWriter
    source_dir = args.SAVE_ROOT + '/source/'  # where to save the source
    utils.copy_source(source_dir)
    args.START_EPOCH = 1
    if args.RESUME > 0:
        args.START_EPOCH = args.RESUME + 1
        for _ in range(args.RESUME):
            scheduler.step()
        model_file_name = '{:s}/model_{:06d}.pth'.format(args.SAVE_ROOT, args.RESUME)
        optimizer_file_name = '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, args.RESUME)
        net.load_state_dict(torch.load(model_file_name))
        optimizer.load_state_dict(torch.load(optimizer_file_name))
        logger.info('After loading checkpoint from epoch {:}, the learning rate is {:}'.format(args.RESUME, args.LR))
    if args.TENSORBOARD:
        log_dir = '{:s}/log-lo_tboard-{}-{date:%m-%d-%H-%M-%S}_logic-{logic:s}_req-weight-{weight}'.format(
            args.log_dir, args.MODE, date=args.DATETIME_NOW, logic=str(args.LOGIC), weight=args.req_loss_weight)
        args.sw = SummaryWriter(log_dir)
        logger.info('Created tensorboard log dir ' + log_dir)

    if args.pretrained_model_path is not None and args.RESUME == 0:
        net.load_state_dict(torch.load(args.pretrained_model_path))
        logger.info("Load pretrained model {:}".format(args.pretrained_model_path))

    logger.info(str(net))
    logger.info(solver_print_str)

    logger.info('EXPERIMENT NAME:: ' + args.exp_name)
    logger.info('Training FPN with {} + {} as backbone '.format(args.ARCH, args.MODEL_TYPE))
    return args, optimizer, scheduler


def train(args, net, train_dataset, val_dataset):
    epoch_size = len(train_dataset) // args.BATCH_SIZE
    args.MAX_ITERS = epoch_size
    args, optimizer, scheduler = setup_training(args, net)

    train_data_loader = data_utils.DataLoader(train_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                              shuffle=True, pin_memory=True, collate_fn=custom_collate, drop_last=True)

    val_data_loader = data_utils.DataLoader(val_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                            shuffle=False, pin_memory=True, collate_fn=custom_collate)

    iteration = 0
    for epoch in range(args.START_EPOCH, args.MAX_EPOCHS + 1):
        print('LR at epoch {:} is {:}'.format(epoch, scheduler.get_last_lr()[0]))
        net.train()

        if args.FBN:
            if args.MULTI_GPUS:
                net.module.backbone.apply(utils.set_bn_eval)
            else:
                if hasattr(net, 'backbone'):  # Only apply if the model has a backbone
                    net.backbone.apply(utils.set_bn_eval)

        iteration = run_train(args, train_data_loader, net, optimizer, epoch, iteration)

        if epoch % args.VAL_STEP == 0 or epoch == args.MAX_EPOCHS:
            net.eval()
            run_val(args, val_data_loader, val_dataset, net, epoch, iteration)

        scheduler.step()


def run_train(args, train_data_loader, net, optimizer, epoch, iteration):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}
    loc_losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}
    cls_losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}
    req_losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}

    torch.cuda.synchronize()
    start = time.perf_counter()

    if args.LOGIC is not None:
        Cplus, Cminus = compute_req_matrices(args)

    for internel_iter, (mix_images, mix_gt_boxes, mix_gt_labels, mix_counts, mix_img_indexs, mix_wh, _, _, mix_is_ulb) in enumerate(train_data_loader):
        if args.DEBUG_num_iter and internel_iter > 22:
            logger.info('DID 5 ITERATIONS IN TRAIN, break.... for debugging only')
            break

        images = mix_images.cuda(0, non_blocking=True)
        gt_boxes = mix_gt_boxes.cuda(0, non_blocking=True)
        gt_labels = mix_gt_labels.cuda(0, non_blocking=True)

        iteration += 1
        optimizer.zero_grad()

        if args.LOGIC is None:
            # Expect segmentation loss (seg_loss), classification loss (cls_loss), and regression loss (reg_loss)
            seg_loss, cls_loss, reg_loss = net(images, gt_boxes, gt_labels)

            # Compute the total loss
            loss = seg_loss.mean() + cls_loss.mean() + reg_loss.mean()
        else:
            seg_loss, cls_loss, reg_loss, selected_is_ulb = net(images, gt_boxes, gt_labels, logic=args.LOGIC, Cplus=Cplus, Cminus=Cminus)
            loss = seg_loss.mean() + cls_loss.mean() + reg_loss.mean() + args.req_loss_weight * req_loss.mean()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Log every few iterations
        if internel_iter % args.LOG_STEP == 0 and iteration > args.LOG_START:
            logger.info('Iteration [{}], Loss: {:.4f}'.format(internel_iter, loss.item()))

        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - start)
        start = time.perf_counter()

    return iteration


def run_val(args, val_data_loader, val_dataset, net, epoch, iteration):
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, epoch)

    seg_total_loss = 0
    with torch.no_grad():
        for val_iter, (images, gt_masks, gt_boxes, gt_labels) in enumerate(val_data_loader):
            images = images.cuda()
            gt_masks = gt_masks.cuda()
            gt_boxes = gt_boxes.cuda()
            gt_labels = gt_labels.cuda()

            # Forward pass for segmentation and detection
            seg_logits, cls_logits, reg_logits = net(images, gt_boxes=gt_boxes, gt_labels=gt_labels, gt_masks=gt_masks)

            # Compute segmentation loss
            seg_loss = net.segmentation_loss_fn(seg_logits, gt_masks)
            seg_total_loss += seg_loss.item()

    avg_seg_loss = seg_total_loss / len(val_data_loader)
    logger.info(f'Epoch [{epoch}], Validation Segmentation Loss: {avg_seg_loss:.4f}')

    torch.cuda.synchronize()
    logger.info(f'Validation TIME::: {time.perf_counter() - start_time}\n\n')
