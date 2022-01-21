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

import os

import numpy as np
import torch.onnx
from torch import nn
from torch.autograd import Variable

from utils.detect import *

# -


class Ensemble_for_onnx(nn.Module):
    def __init__(self, models):
        super(Ensemble_for_onnx, self).__init__()
        self.models = models

    def forward(self, x):
        results = []
        with torch.no_grad():
            for model in self.models:
                x1 = model(x)
                results.append(x1)
            output = torch.cat([i for i in results], dim=1)
        return output


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models

    def forward(self, x):
        with torch.no_grad():
            results = torch.stack(
                [torch.squeeze(self.models[i](x), 1) for i in range(len(self.models))],
                1,
            )
        return results


def make_ensemble(models_paths, device, for_onnx=False):
    models = []
    for i in range(len(models_paths)):
        model_0, device = get_torch_model(model_path=models_paths[i], device=device)
        for param in model_0.parameters():
            param.requires_grad = False
        model_0.eval()
        models.append(model_0)
    if for_onnx:
        model = Ensemble_for_onnx(models)
    else:
        model = Ensemble(models)
    return model


def get_ensemble_masks(
    path,
    model,
    single_threshold,
    img_sizes,
    intersections,
    norm=True,
    n_models=7,
    device="cuda",
    max_batch_size=8,
    disable_rasterio_warning=False,
):
    ds = get_dataset(path=path)
    dict_meta = get_meta(path=path)
    total_mask = np.zeros([ds.height, ds.width])
    size_intersection = sorted(
        [[x, y] for x, y in zip(img_sizes, intersections)], reverse=True
    )
    list_single_masks = []
    if any([x for x in img_sizes if x > ds.width or x > ds.height]):
        for i in range(len(size_intersection)):
            if (
                size_intersection[i][0] > ds.width
                or size_intersection[i][0] > ds.height
            ):
                size = (np.min([ds.width, ds.height]) // 32 - (i + 1)) * 32
                size_intersection[i][0] = size
    size_intersection = sorted(size_intersection)
    for img_size, intersection in size_intersection:
        img = get_image(
            rasterio_dataset=ds,
            chanels=dict_meta["model_chanels"],
            img_size=img_size,
            disable_rasterio_warning=disable_rasterio_warning,
        )
        if img is not None:
            list_coord = get_cut_coord(
                image=img, img_size=img_size, intersection=intersection
            )
            list_images = cut_image(
                image=img,
                sliding_window_coordinates=list_coord,
                norm=norm,
            )
            batch_ids = calc_batches(
                array_images=list_images, max_batch_size=max_batch_size
            )
            prediction = np.zeros([list_images.shape[0], n_models, img_size, img_size])
            for b_ix in batch_ids:
                batch_images = list_images[b_ix[0] : b_ix[1]]
                with torch.no_grad():
                    masks = (
                        model.forward(
                            torch.tensor(batch_images, dtype=torch.float).to(device)
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    prediction[b_ix[0] : b_ix[1], :, :, :] = masks

            all_masks = ensemble_prediction_to_masks(
                image=img,
                list_masks=prediction,
                sliding_window_coordinates=list_coord,
                threshold=single_threshold,
            )

            total_mask += np.sum(all_masks, axis=0)
        else:
            ds, img, total_mask, size_intersection = None, None, None, None
    return ds, img, total_mask, size_intersection


def ensemble_prediction_to_masks(
    image, list_masks, sliding_window_coordinates, threshold=0.99
):
    """
    Concatenates model's predictions into one binary mask
    """
    mask = np.zeros([list_masks.shape[1], image.shape[1], image.shape[2]])
    for n in range(len(sliding_window_coordinates)):
        y, x = sliding_window_coordinates[n]
        for j in range(list_masks.shape[1]):
            single_mask = np.where(list_masks[n, j] > threshold, 1, 0)
            mask[j, y[0] : y[1], x[0] : x[1]] = (
                mask[j, y[0] : y[1], x[0] : x[1]] + single_mask
            )
    mask = np.where(mask > 0, 1, 0)
    return mask


def ensemble_to_onnx(torch_model, onnx_path, model_name, device, img_size):
    assert img_size % 32 == 0, "img_size must be a multiple of 32"
    dummy_input = Variable(
        torch.randn(1, 3, img_size, img_size, dtype=torch.float32, requires_grad=False)
    ).to(device)
    torch_model.eval().to(device)

    if not os.path.exists(onnx_path):
        os.makedirs(onnx_path)
    torch.onnx.export(
        torch_model,
        dummy_input,
        os.path.join(onnx_path, f"{model_name}.onnx"),
        opset_version=11,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        training=torch.onnx.TrainingMode.EVAL,
    )


def triton_ensemble_predict(
    path,
    triton_model_names,
    triton_http_service_url,
    single_threshold,
    img_sizes,
    intersections,
    norm=True,
    n_models=7,
    max_batch_size=8,
    disable_rasterio_warning=False,
):
    ds = get_dataset(path=path)
    dict_meta = get_meta(path=path)
    total_mask = np.zeros([ds.height, ds.width])
    for img_size, intersection, triton_model_name in zip(
        img_sizes, intersections, triton_model_names
    ):

        img = get_image(
            rasterio_dataset=ds,
            chanels=dict_meta["model_chanels"],
            img_size=img_size,
            disable_rasterio_warning=disable_rasterio_warning,
        )
        if img is not None:
            list_coord = get_cut_coord(
                image=img, img_size=img_size, intersection=intersection
            )
            list_images = cut_image(
                image=img,
                sliding_window_coordinates=list_coord,
                norm=norm,
            )
            batch_ids = calc_batches(
                array_images=list_images, max_batch_size=max_batch_size
            )
            prediction = np.zeros([list_images.shape[0], n_models, img_size, img_size])
            for b_ix in batch_ids:
                batch_images = list_images[b_ix[0] : b_ix[1]]
                inputs = [
                    httpclient.InferInput(
                        "input",
                        [len(batch_images), 3, img_size, img_size],
                        "FP32",
                    )
                ]
                inputs[0].set_data_from_numpy(batch_images, binary_data=True)

                outputs = [httpclient.InferRequestedOutput("output", binary_data=True)]

                triton_client = httpclient.InferenceServerClient(
                    url=triton_http_service_url,
                    verbose=False,
                    connection_timeout=1000,
                    network_timeout=1000,
                    concurrency=1,
                )

                results = triton_client.infer(
                    triton_model_name,
                    inputs,
                    outputs=outputs,
                    query_params={},
                    request_compression_algorithm="none",
                    response_compression_algorithm="none",
                )
                masks = results.as_numpy("output")
                prediction[b_ix[0] : b_ix[1], :, :, :] = masks
            all_masks = ensemble_prediction_to_masks(
                image=img,
                list_masks=prediction,
                sliding_window_coordinates=list_coord,
                threshold=single_threshold,
            )

            total_mask += np.sum(all_masks, axis=0)
        else:
            ds, img, total_mask = None, None, None, None
    return ds, img, total_mask