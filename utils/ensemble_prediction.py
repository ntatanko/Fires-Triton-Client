# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +

import matplotlib.pyplot as plt
import numpy as np

from utils.detect import *

# -

"""
Ensemble prediction for difficult tiles.
Only for models in .pth format.
It is possible to use different models with differenf image sizes (image size must be multiple of 32)
"""

def ensemble_predict(
    path,
    list_models,
    single_threshold=0.99,
    intersections=[0.1, 0.1],
    img_sizes=[384, 512],
    norm=True,
    model_numbers=[1, 2],
    device="cuda",
    return_image=False,
    max_batch_size=8,
    disable_rasterio_warning=True,
):
    assert all([x % 32 == 0 for x in img_sizes]), "image size must be multiple of 32"

    ds = get_dataset(path=path)
    dict_meta = get_meta(path=path)
    img = get_image(
        rasterio_dataset=ds,
        chanels=dict_meta["model_chanels"],
        img_size=np.min(img_sizes),
        disable_rasterio_warning=disable_rasterio_warning,
    )
    if img is not None:
        size_intersection = sorted(
            [[x, y] for x, y in zip(img_sizes, intersections)], reverse=True
        )
        total_mask = np.zeros(img.shape[1:])
        list_single_masks = []
        if any([x for x in img_sizes if x > img.shape[1] or x > img.shape[2]]):
            for i in range(len(size_intersection)):
                if (
                    size_intersection[i][0] > img.shape[1]
                    or size_intersection[i][0] > img.shape[2]
                ):
                    size = (np.min([img.shape[1], img.shape[2]]) // 32 - (i + 1)) * 32
                    size_intersection[i][0] = size
        size_intersection = sorted(size_intersection)
        for img_size, intersection in size_intersection:
            list_coord = get_cut_coord(
                image=img, img_size=img_size, intersection=intersection
            )
            list_images = cut_image(
                image=img,
                sliding_window_coordinates=list_coord,
                norm=norm,
            )
            models = [list_models[x - 1] for x in model_numbers]
            for model in models:
                masks = torch_predict(
                    model=model,
                    device=device,
                    array_images=list_images,
                    max_batch_size=max_batch_size,
                )
                predicted_mask = prediction_to_mask(
                    image=img,
                    list_masks=masks,
                    sliding_window_coordinates=list_coord,
                    threshold=single_threshold,
                )
                list_single_masks.append(predicted_mask)
                total_mask += predicted_mask
    if return_image:
        return total_mask, list_single_masks, img, [x[0] for x in size_intersection]
    else:
        return total_mask, list_single_masks, [x[0] for x in size_intersection]



def plot_ensemble_masks(list_masks, model_numbers, img_sizes):
    """
    Shows predicted mask for every model and image sizes
    """
    n_cols= 4
    n_rows = (len(list_masks)//n_cols)+1
    plt.figure(figsize=(20, 5*n_rows))
    for j in range(len(list_masks)):
        n_mod = len(list_masks)//len(img_sizes)
        plt.subplot(n_rows, n_cols, j+1)
        plt.title(f'model {model_numbers[j%n_mod]} with {img_sizes[j//n_mod]} image size')
        plt.imshow(list_masks[j], alpha=0.3)
    plt.show();


def plot_ensemble_prediction(img, mask, list_masks, thresholds = [4,5,6]):
    """
    Shows predicted mask with different thresholds.
    For example, threshold = 4 means that pixels are true if in 4 or more predictions
    these pixels belong to the burned-out area.
    Usually the optimal threshold is about half of "list_masks" lenght.
    """
    if np.max(thresholds)<=len(list_masks):
        n_cols= 1 if len(thresholds)==1 else 2
        n_rows = (len(thresholds)//n_cols)+1
        plt.figure(figsize=(20, 10*n_rows))
        for j in range(len(thresholds)):
            plt.subplot(n_rows, n_cols, j+1)
            plt.title(f'threshold {thresholds[j]}')
            plt.imshow(img.transpose(1,2,0))
            plt.imshow(mask>=thresholds[j], alpha=0.3)
        plt.show();
    else:
        print(f'Threshold must be less or equal {len(list_masks)}')


