# Torch imports
import torch
import torchvision.transforms as T
# Misc imports
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Literal, Optional, Any


class ShowPredictions:
    
    def __init__(
        self, 
        exp = None, 
        vis_type: Literal["classification", "segmentation"] = "classification",
        col_wrap: int = 4,
        threshold: float = 0.5,
        size_per_image: int = 5,
        denormalize: Optional[Any] = None
    ):
        self.col_wrap = col_wrap
        self.vis_type = vis_type
        self.threshold = threshold
        self.size_per_image = size_per_image
        # Sometimes we normalize the intensity values so we need to denormalize them for visualization.
        if denormalize is not None:
            # Denormalization transform
            self.denormalize = T.Normalize(
                mean=[-m/s for m, s in zip(denormalize['mean'], denormalize['std'])],
                std=[1/s for s in denormalize['std']]
            )
        else:
            self.denormalize = None


    def __call__(self, batch):

        # Prints some metric stuff
        if "loss" in batch and len(batch["loss"].shape) == 0:
            print("Loss: ", batch["loss"].item())

        if self.vis_type == "classification": 
            ClassificationShowPreds(
                batch, 
                col_wrap=self.col_wrap, 
                threshold=self.threshold, 
                size_per_image=self.size_per_image,
                denormalize=self.denormalize
            )
        elif self.vis_type == "segmentation":
            SegmentationShowPreds(
                batch, 
                threshold=self.threshold, 
                size_per_image=self.size_per_image,
                denormalize=self.denormalize
            )
        elif self.vis_type == "reconstruction":
            ReconstructionShowPreds(
                batch, 
                col_wrap=self.col_wrap, 
                size_per_image=self.size_per_image,
                denormalize=self.denormalize
            )
        else:
            raise ValueError("Invalid vis_type. Must be 'classification' or 'segmentation'.")


def ClassificationShowPreds(
    batch, 
    col_wrap: int,
    threshold: float,
    size_per_image: int,
    denormalize: Any
):
    # Transfer image and label to the cpu.
    x = batch["x"]
    y = batch["y_true"]
    y_hat = batch["y_pred"]

    # Get the predicted label
    if y_hat.shape[1] == 1:
        y_hat = (torch.sigmoid(y_hat) > threshold).astype(int)
    else:
        y_hat = torch.argmax(y_hat, axis=1)
    
    # Print the batch acurracy.
    acc = (y == y_hat).sum().item() / y.shape[0]
    print("Batch Accuracy: ", acc)

    # If x is rgb (has 3 input channels)
    if x.shape[1] == 3:
        img_cmap = None
        if denormalize is not None:
            x = denormalize(x)
            x = x * 255
            x = x.int()
        x = x.permute(0, 2, 3, 1) # Move channel dimension to last.
    else:
        img_cmap = "gray"
    
    # If the image is float, clip between [0, 1]
    if x.dtype == torch.float32:
        x = torch.clamp(x, 0, 1)
    elif x.dtype == torch.int:
        x = torch.clamp(x, 0, 255)

    # Prepare the tensors for visualization as npdarrays.
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    
    # Prepare matplotlib objects.
    bs = y.shape[0]
    ncols = min(bs, col_wrap)
    nrows = int(np.ceil(bs / ncols))
    f, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * size_per_image, nrows * size_per_image))
    # Go through each item in the batch.
    for b_idx in range(bs):
        col_idx = b_idx % ncols
        row_idx = b_idx // ncols
        if bs == 1:
            axarr.set_title(f"Predicted: {y_hat[b_idx]} GT: {y[b_idx]}")
            im1 = axarr.imshow(x[b_idx], cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr, orientation='vertical')
        elif nrows == 1:
            axarr[col_idx].set_title(f"Predicted: {y_hat[b_idx]} GT: {y[b_idx]}")
            im1 = axarr[col_idx].imshow(x[b_idx], cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[col_idx], orientation='vertical')
        else:
            axarr[row_idx, col_idx].set_title(f"Predicted: {y_hat[b_idx]} GT: {y[b_idx]}")
            im1 = axarr[row_idx, col_idx].imshow(x[b_idx], cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[row_idx, col_idx], orientation='vertical')
    # Turn off all of the grids and axes in the subplot array
    if not isinstance(axarr, np.ndarray):
        all_ax = [axarr]
    else:
        all_ax = axarr.flatten()
    for ax in all_ax:
        ax.grid(False)

    plt.show()


def SegmentationShowPreds(
    batch, 
    threshold: float,
    size_per_image: int,
    denormalize: Any
):
    # If our pred has a different batchsize than our inputs, we
    # need to tile the input and label to match the batchsize of
    # the prediction.
    if ("y_probs" in batch) and (batch["y_probs"] is not None):
        pred_cls = "y_probs"
    elif ("y_pred" in batch) and (batch["y_pred"] is not None):
        pred_cls = "y_pred"
    else:
        assert ("y_logits" in batch) and (batch["y_logits"] is not None), "Must provide either probs, preds, or logits."
        pred_cls = "y_logits"

    x = batch["x"]
    y = batch["y_true"]
    # Transfer image and label to the cpu.
    x = x.detach().cpu()
    y = y.detach().cpu() 

    # Get the predicted label
    y_hat = batch[pred_cls].detach().cpu()

    bs = y.shape[0]
    num_pred_classes = y_hat.shape[1]

    if num_pred_classes <= 2:
        label_cm = "gray"
    else:
        colors = [(0, 0, 0)] + [(np.random.random(), np.random.random(), np.random.random()) for _ in range(num_pred_classes - 1)]
        cmap_name = "seg_map"
        label_cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=num_pred_classes)

    # Make a hard prediction.
    if num_pred_classes > 1:
        if pred_cls != "y_probs":
            y_hat = torch.softmax(y_hat, dim=1)
        if num_pred_classes == 2 and threshold != 0.5:
            y_hard = (y_hat[:, 1, :, :] > threshold).int()
        else:
            y_hard = torch.argmax(y_hat, dim=1)
        # Now we want to get the max probs for each pixel.
        y_hat = torch.max(y_hat, dim=1)[0]
    else:
        if pred_cls != "y_probs":
            y_hat = torch.sigmoid(y_hat)
        y_hard = (y_hat > threshold).int()

    # If x is 5 dimensionsal, we need to take the midslice of the last dimension of all 
    # of our tensors.
    if len(x.shape) == 5:
        # We want to look at the slice corresponding to the maximum amount of label.
        # y shape here is (B, C, Spatial Dims)
        y_squeezed = y.squeeze(1) # (B, Spatial Dims)
        # Sum over the spatial dims that aren't the last one.
        lab_per_slice = y_squeezed.sum(dim=tuple(range(1, len(y_squeezed.shape) - 1)))
        # Get the max slices per batch item.
        max_slices = torch.argmax(lab_per_slice, dim=1)
        # Index into our 3D tensors with this.
        x = torch.stack([x[i, ...,  max_slices[i]] for i in range(bs)]) 
        y = torch.stack([y[i, ..., max_slices[i]] for i in range(bs)])
        y_hat = torch.stack([y_hat[i, ..., max_slices[i]] for i in range(bs)])
        y_hard = torch.stack([y_hard[i, ..., max_slices[i]] for i in range(bs)])
    
    # If x is rgb (has 3 input channels)
    if x.shape[1] == 3:
        img_cmap = None
        if denormalize is not None:
            x = denormalize(x)
            x = (x * 255).int()
        # Move the 1st dmension to the end, where we don't know the total
        # number of dims (e.g. 2D, 3D, etc).
        x = x.permute(0, *range(2, len(x.shape)), 1) # Move channel dimension to last.
    else:
        img_cmap = "gray"

    # Squeeze all tensors in prep.
    x = x.numpy().squeeze() # Move channel dimension to last.
    y = y.numpy().squeeze()
    y_hat = y_hat.squeeze()
    y_hard = y_hard.numpy().squeeze()
    # We plot four images per batch item.
    f, axarr = plt.subplots(nrows=bs, ncols=4, figsize=(4 * size_per_image, bs*size_per_image))

    # Go through each item in the batch.
    for b_idx in range(bs):
        if bs == 1:
            axarr[0].set_title("Image")
            im1 = axarr[0].imshow(x, cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[0], orientation='vertical')

            axarr[1].set_title("Label")
            im2 = axarr[1].imshow(y, cmap=label_cm, interpolation='None')
            f.colorbar(im2, ax=axarr[1], orientation='vertical')

            if len(y_hat.shape) == 3:
                max_probs = torch.max(y_hat, dim=0)[0]
            else:
                assert len(y_hat.shape) == 2, "Soft prediction must be 2D if not 3D."
                max_probs = y_hat
            axarr[2].set_title("Max Probs")
            im4 = axarr[2].imshow(max_probs, cmap='gray', vmin=0.0, vmax=1.0, interpolation='None')
            f.colorbar(im4, ax=axarr[2], orientation='vertical')

            axarr[3].set_title("Hard Prediction")
            im3 = axarr[3].imshow(y_hard, cmap=label_cm, interpolation='None')
            f.colorbar(im3, ax=axarr[3], orientation='vertical')
        else:
            axarr[b_idx, 0].set_title("Image")
            im1 = axarr[b_idx, 0].imshow(x[b_idx], cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[b_idx, 0], orientation='vertical')

            axarr[b_idx, 1].set_title("Label")
            im2 = axarr[b_idx, 1].imshow(y[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im2, ax=axarr[b_idx, 1], orientation='vertical')

            axarr[b_idx, 2].set_title("Soft Prediction")
            im3 = axarr[b_idx, 2].imshow(y_hat[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im3, ax=axarr[b_idx, 2], orientation='vertical')

            axarr[b_idx, 3].set_title("Hard Prediction")
            im4 = axarr[b_idx, 3].imshow(y_hard[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im4, ax=axarr[b_idx, 3], orientation='vertical')
    # Turn off all of the grids and axes in the subplot array
    if not isinstance(axarr, np.ndarray):
        all_ax = [axarr]
    else:
        all_ax = axarr.flatten()
    for ax in all_ax:
        ax.grid(False)
    plt.show()


def ReconstructionShowPreds(
    batch, 
    col_wrap: int,
    size_per_image: int,
    denormalize: Any
):
    # Get the predicted label
    if isinstance(batch["y_pred"], dict):
        x = batch["y_true"]["image"]
        y_pred = batch["y_pred"]["image"]
    else:
        x = batch["y_true"]
        y_pred = batch["y_pred"]
    #
    x = x.detach().cpu()
    y_hat = y_pred.detach().cpu()
    bs = x.shape[0]

    # If x is rgb (has 3 input channels)
    if x.shape[1] == 3:
        img_cmap = None
        # First we process the image for visualization.
        x = proc_rgb_image(x, denormalize_fn=denormalize)
        y_hat = proc_rgb_image(y_hat, denormalize_fn=denormalize)
    else:
        img_cmap = "gray"

    # Squeeze all tensors in prep.
    x = x.numpy().squeeze() # Move channel dimension to last.
    y_hat = y_hat.squeeze()
    # We plot four images per batch item.
    if bs == 1:
        f, axarr = plt.subplots(nrows=1, ncols=2, figsize=(2 * size_per_image, size_per_image))
    else:
        num_images = bs * 2
        nrows = max(int(np.ceil(num_images / col_wrap)), 2)
        ncols = min(bs, col_wrap)
        f, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * size_per_image, 4 * size_per_image))

    # Go through each item in the batch.
    for b_idx in range(bs):
        if bs == 1:
            axarr[0].set_title("Image")
            im1 = axarr[0].imshow(x, cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[0], orientation='vertical')

            if 'loss' in batch:
                axarr[1].set_title("Pred Reconstruction\nLoss: {:.3f}".format(batch["loss"].item()))
            else:
                axarr[1].set_title("Pred Reconstruction")
            im2 = axarr[1].imshow(y_hat, cmap=img_cmap, interpolation='None')
            f.colorbar(im2, ax=axarr[1], orientation='vertical')
        else:
            # Get the col and row index based on the batch index and col_wrap.
            col_idx = b_idx % ncols
            rowset = b_idx // ncols

            axarr[2*rowset, col_idx].set_title("Image")
            im1 = axarr[2*rowset, col_idx].imshow(x[b_idx], cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[2*rowset, col_idx], orientation='vertical')
            # Get the loss for this batch item.
            if "loss" in batch:
                b_loss = batch["loss"]
                b_idx_loss = b_loss.item() if len(b_loss.shape) == 0 else b_loss[b_idx].item()
                axarr[2*rowset + 1, col_idx].set_title("Pred Reconstruction\nLoss: {:.3f}".format(b_idx_loss))
            else:
                axarr[2*rowset + 1, col_idx].set_title("Pred Reconstruction")
            im2 = axarr[2*rowset + 1, col_idx].imshow(y_hat[b_idx], cmap=img_cmap, interpolation='None')
            f.colorbar(im2, ax=axarr[2*rowset + 1, col_idx], orientation='vertical')
    # Turn off all of the grids and axes in the subplot array
    if not isinstance(axarr, np.ndarray):
        all_ax = [axarr]
    else:
        all_ax = axarr.flatten()
    for ax in all_ax:
        ax.grid(False)
    plt.show()


def proc_rgb_image(x, denormalize_fn):
    # If using a denorm fn, then we will want to
    # use an integer image.
    if denormalize_fn is not None:
        x = denormalize_fn(x)
        x = x * 255
        x = x.int()
        # Clip y_hat to be between 0 and 255.
        x = torch.clamp(x, 0, 255)
    else:
        x = torch.clamp(x, 0, 1)
    
    x = x.permute(0, 2, 3, 1) # Move channel dimension to last.
    return x 