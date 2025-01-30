from .vae import vae_loss, kld_loss
from .flat import *
from .multi import MultiLoss, CombinedLoss
from .segmentation import (
    soft_jaccard_score,
    soft_dice_loss,
    SoftDiceLoss,
    SoftJaccardLoss,
    PixelMSELoss,
    PixelCELoss
)
from .total_variation import total_variation_loss, TotalVariationLoss
from .ncc import ncc_loss, NCCLoss
