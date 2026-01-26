from torch.nn.functional import cross_entropy

from .accuracy import accuracy, f1_score, auroc, auprc, correct, precision, recall
from .entropy import binary_cross_entropy
from .size import model_size

from .segmentation import (
    soft_dice_score,
    soft_jaccard_score,
    pixel_accuracy, 
    pixel_mse,
    dice_score,
    jaccard_score,
)

from .model import module_table, parameter_table
