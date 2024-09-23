import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.utils as utils
logger = utils.get_logger(__name__)

# FocalLoss for detection loss
class FocalLoss(nn.Module):
    def __init__(self, args, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        '''Compute the focal loss for classification'''
        preds = torch.sigmoid(preds)

        # Add checks for extreme values in preds and targets
        if torch.any(preds < 0) or torch.any(preds > 1):
            logger.warning("Predictions are out of sigmoid range")
        if torch.any(targets < 0) or torch.any(targets > 1):
            logger.warning("Targets have invalid values")

        loss = F.binary_cross_entropy(preds, targets, reduction='none')
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        pt = preds * targets + (1.0 - preds) * (1.0 - targets)
        focal_weight = alpha_factor * ((1 - pt) ** self.gamma)

        loss = (loss * focal_weight).mean()
        return losss

# Smooth L1 Loss for regression
def smooth_l1_loss(input, target, beta=1.0):
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss.mean()

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 using 3D convolutions"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv using 3D pool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv for 3D data"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels // 2 + in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetWithDetection(nn.Module):
    def __init__(self, n_channels, n_classes, num_anchors, args, bilinear=True):
        super(UNetWithDetection, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_anchors = num_anchors
        self.bilinear = bilinear

        # UNet Encoder and Decoder
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.up1 = Up(256, 128, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.up3 = Up(64, 32, bilinear)

        # Segmentation head
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)

        # Detection heads (classification and regression)
        self.classification_head = nn.Conv3d(32, num_anchors * n_classes, kernel_size=3, padding=1)
        self.regression_head = nn.Conv3d(32, num_anchors * 4, kernel_size=3, padding=1)

        if args.MODE == 'train':
            self.criterion_cls = FocalLoss(args)
            self.criterion_reg = smooth_l1_loss

        # Segmentation Loss
        self.segmentation_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, gt_boxes=None, gt_labels=None, gt_masks=None):
        # UNet encoder-decoder for segmentation
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Segmentation output
        segmentation_logits = self.outc(x)

        # Detection outputs (classification and regression)
        classification_logits = self.classification_head(x)
        regression_logits = self.regression_head(x)

        # If ground truth is provided (training mode)
        if gt_boxes is not None and gt_labels is not None and gt_masks is not None:
            # Compute losses
            seg_loss = self.segmentation_loss_fn(segmentation_logits, gt_masks)
            cls_loss = self.criterion_cls(classification_logits, gt_labels)
            reg_loss = self.criterion_reg(regression_logits, gt_boxes)

            # Return segmentation loss, classification loss, and regression loss
            return seg_loss, cls_loss, reg_loss

        # Return the raw outputs if not in training mode
        return segmentation_logits, classification_logits, regression_logits


def build_unet(n_channels, n_classes, num_anchors, args):
    torch.cuda.empty_cache()
    return UNetWithDetection(n_channels=n_channels, n_classes=n_classes, num_anchors=num_anchors, args=args)
