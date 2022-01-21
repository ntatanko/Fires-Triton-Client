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

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask
import segmentation_models_pytorch as smp
import shapely.geometry
import torch
import tritonclient.http as httpclient
from IPython.display import clear_output
from skimage import measure

# -


"""
Tile processing logic:
 - tiff is transformed to numpy array,
 - image is cut into separate pictures of 384 * 384 size (for model in .pth format it is possible to use sizes that are multiple of 32),
 - for each of them model predict mask,
 - they are concatenated into a binary mask,
 - the mask is converted to polygons,
 - polygons are filtered by size,
 - internal and external polygons are detected,
 - according to this, the final geo-polygons are calculaited in shapely.geometry.Polygon format
"""


def start_triton(max_batch_size, triton_http_service_url, img_size, model_name):

    """
    For the optimization at the first server start, takes a lot of time.
    Optimization with batch sizes [max_batch_size, 4, 1],
    it is recomended to set max_batch_size to 8, due to time calculation
    there is no point in max_batch_size bigger than that.
    """

    for i in [max_batch_size, 4, 1]:
        batch_images = np.zeros([i, 3, img_size, img_size], dtype="float32")
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
            connection_timeout=1000,  # may be needed to wait for tensorrt to run optimization on first call
            network_timeout=1000,
            concurrency=1,
        )
        results = triton_client.infer(
            model_name,
            inputs,
            outputs=outputs,
            query_params={},
            request_compression_algorithm="none",
            response_compression_algorithm="none",
        )


def get_dataset(path):
    if ".tiff" not in path:
        path = os.path.join(path, "response.tiff")
    rasterio_dataset = rasterio.open(path)
    return rasterio_dataset


def get_meta(path, chanels_for_model=["B11", "B08", "B02"]):
    """
    Use only only if you have "request.json".
    Pull from request.json:
      - tile coordinates,
      - date,
      - all channels in the order in which they are in tiff file (in request ["payload"] ["evalscript"] sample.B01, etc.),
      - channel's indices ["B11", "B08", "B02"] (for rasterio.open().read(), where indexes starts with 1),
    """
    dict_meta = {}
    if ".json" not in path:
        path = os.path.join(path, "request.json")
    if not os.path.exists(path):
        chanels = [0, 1, 2]
        return chanels
    else:
        with open(path, "rb") as f:
            request = json.load(f)
        dict_meta["path"] = path
        dict_meta["date"] = [
            request["payload"]["input"]["data"][0]["dataFilter"]["timeRange"]["from"][
                :10
            ],
            request["payload"]["input"]["data"][0]["dataFilter"]["timeRange"]["to"][
                :10
            ],
        ]
        dict_meta["bbox"] = request["payload"]["input"]["bounds"]["bbox"]
        samples = [
            re.findall(r"sample\..*[^\W]", x)[0]
            for x in re.findall(r"sample\..*", request["payload"]["evalscript"])
        ]
        dict_meta["output_chanels"] = [x.split("sample.")[1] for x in samples]
        dict_meta["model_chanels"] = [
            dict_meta["output_chanels"].index(x) + 1 for x in chanels_for_model
        ]
        return dict_meta


def get_image(
    rasterio_dataset,
    chanels,
    img_size,
    clip_value=10000,
    eps=0.00001,
    disable_rasterio_warning=True,
):
    """
    The output is an image in format np.array(3, width, height) or ***None*** if:
     - np.max() == 0 (error in the date, the tile is black, there is no point in further processing),
     - tile width or height less than ***img_size***

    "chanels" - indices of ["B11", "B08", "B02"] channels in "tiff" file (indexing starts with 1 !!!),
    if there is no ***request.json*** from which the indices can be obtained, then you need to specify
    the channel indices ["B11", "B08", "B02"] in the tiff file in this order

    In rasterio version 1.2.9, when reading channels (rasterio_dataset.read(chanels)), the warning that
    issued  does not belong to Warning class, therefore clear_output() was added, it is not convenient,
    but I did not find any other solution (also it is possible to downgrade rasterio version)

    "img_size" for model must be a multiple of 32
    """

    assert img_size % 32 == 0, "img_size must be a multiple of 32"
    image = rasterio_dataset.read(chanels)
    if disable_rasterio_warning:
        clear_output()
    if image.shape[1] >= img_size and image.shape[2] >= img_size:

        if image.max() != 0:
            image = np.clip(image, 0, clip_value)
            image_norm = np.array(
                [
                    (image[n] - image[n].min()) / (image[n].max() - image[n].min())
                    for n in range(image.shape[0])
                ]
            )
            return image_norm
        else:
            return None
    else:
        return None


def get_cut_coord(image, img_size, intersection=0.1):
    """
    Calculate list of coordinates for cutting image into a set of small images
    with size img_size*img_size with intersection
    """

    y_coord = []
    x_coord = []
    y_coord.append([0, img_size])
    y = int((1 - intersection) * img_size)
    while y < (image.shape[1] - img_size):
        y_coord.append([y, y + img_size])
        y = int((y + img_size) - intersection * img_size)
    y_coord.append([image.shape[1] - img_size, image.shape[1]])

    x_coord.append([0, img_size])
    x = int((1 - intersection) * img_size)
    while x < (image.shape[2] - img_size):
        x_coord.append([x, x + img_size])
        x = int((x + img_size) - intersection * img_size)
    x_coord.append([image.shape[2] - img_size, image.shape[2]])
    sliding_window_coordinates = []
    for y in y_coord:
        for x in x_coord:
            sliding_window_coordinates.append([y, x])
    return sliding_window_coordinates


def cut_image(image, sliding_window_coordinates, norm=True):
    """
    Output is an array of small images from the original image according
    to coordinates from get_cut_coord()
    """

    list_images = []
    for y, x in sliding_window_coordinates:
        small_image = image[:, y[0] : y[1], x[0] : x[1]].copy()
        if norm and small_image.max() != 0:
            for c in range(3):
                small_image[c] = small_image[c] - small_image[c].min()
                small_image[c] = small_image[c] / small_image[c].max()
        list_images.append(small_image.astype("float32"))
    return np.array(list_images)


def calc_batches(array_images, max_batch_size=8):
    """
    Groups images into batches of [max_batch_size, 4, 1] sizes, it
    is nessesary for Triton, because of the batch optimization.

    Return image indexes for batches.
    """
    n_images = array_images.shape[0]
    batch_ids = []
    n = n_images // max_batch_size
    tail = n_images % max_batch_size
    if n != 0:
        for i in range(n):
            batch_ids.append([i * max_batch_size, i * max_batch_size + max_batch_size])
    if tail >= 4:
        batch_ids.append(
            [
                max_batch_size * (n_images // max_batch_size),
                max_batch_size * (n_images // max_batch_size) + 4,
            ]
        )
        tail -= 4
    for i in range(tail):
        batch_ids.append([n_images - tail + i, n_images - tail + i + 1])
    return batch_ids


def triton_predict(
    array_images,
    triton_http_service_url,
    model_name,
    max_batch_size=8,
):
    """
    "array_images" - images in np.array format from "cut_image()"
    "triton_http_service_url" - host of docker container with Fires-API-Server,
    for example TRITON_HTTP_SERVICE_URL = "172.17.0.1:18000"

    Returns array of model's predictions for each image from "array_images"
    """

    batch_ids = calc_batches(array_images=array_images, max_batch_size=max_batch_size)
    img_size = array_images.shape[-1]
    masks_array = np.zeros([array_images.shape[0], img_size, img_size])
    for b_ix in batch_ids:
        batch_images = array_images[b_ix[0] : b_ix[1]]
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
            model_name,
            inputs,
            outputs=outputs,
            query_params={},
            request_compression_algorithm="none",
            response_compression_algorithm="none",
        )
        masks = results.as_numpy("output")
        masks_array[b_ix[0] : b_ix[1], :, :] = np.reshape(
            masks, (-1, img_size, img_size)
        )
    return masks_array


def get_torch_model(model_path, device="cpu"):
    ENCODER = "timm-efficientnet-b4"
    ACTIVATION = "sigmoid"
    model = smp.DeepLabV3Plus(
        in_channels=3,
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=1,
        activation=ACTIVATION,
    )
    try:
        state_dict = torch.load(model_path, map_location=device)["model_state_dict"]
    except:
        state_dict = torch.load(model_path, map_location=device).state_dict()
    model.load_state_dict(
        state_dict,
    )
    model.to(device)
    print("Using device:", device)
    return model, device


def torch_predict(model, device, array_images, max_batch_size=8):
    """
    Output is an array of model predictions for each image from ***array_images***
    """
    batch_ids = calc_batches(array_images=array_images, max_batch_size=max_batch_size)
    img_size = array_images.shape[-1]
    masks_array = np.zeros([array_images.shape[0], img_size, img_size])
    for b_ix in batch_ids:
        batch_images = array_images[b_ix[0] : b_ix[1]]
        with torch.no_grad():
            masks = (
                model.predict(torch.tensor(batch_images, dtype=torch.float).to(device))
                .cpu()
                .detach()
                .numpy()
            )
        masks_array[b_ix[0] : b_ix[1], :, :] = np.reshape(
            masks, (-1, img_size, img_size)
        )
    return masks_array


def prediction_to_mask(image, list_masks, sliding_window_coordinates, threshold=0.9):
    """
    Concatenates model's predictions into one binary mask
    """
    mask = np.zeros([image.shape[1], image.shape[2]])
    for n in range(len(sliding_window_coordinates)):
        y, x = sliding_window_coordinates[n]
        single_mask = np.where(list_masks[n] > threshold, 1, 0)
        mask[y[0] : y[1], x[0] : x[1]] = mask[y[0] : y[1], x[0] : x[1]] + single_mask
    mask = np.where(mask > 0, 1, 0)
    return mask


def mask_to_polygon_coordinates(binary_mask, tolerance=0):
    """
    Converts binary mask to polygon coordinates along the border 1/0
    """
    coordinates = []
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.array(
        [np.subtract(contours[i], 1) for i in range(len(contours))], dtype=object
    )
    for contour in contours:
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        contour = measure.approximate_polygon(contour, tolerance)
        contour = np.floor(contour).astype("int")
        coordinates.append(np.where(contour < 0, 0, contour))
    return coordinates


def get_geo_coordinates(rasterio_dataset, img_coordinates):
    """
    Coordinates of polygons are converted to geo-coordinates
    """
    geo_coordinates = []
    for i in range(len(img_coordinates)):
        xs, ys = rasterio_dataset.xy(img_coordinates[i][:, 0], img_coordinates[i][:, 1])
        geo_coordinates.append([[xs[x], ys[x]] for x in range(len(xs))])
    return geo_coordinates


def coordinates_to_polygons(coordinates):
    """
    Converts coordinates of polygons to polygons of shapely.geometry.Polygon format
    """
    polygons = []
    for i in range(len(coordinates)):
        polygon = shapely.geometry.Polygon(coordinates[i])
        polygons.append(polygon)
    return polygons


def area_filter(all_img_polygons, area_threshold=100):
    """
    Returs polygons with area more than area threshold
    """
    return [
        i
        for i in range(len(all_img_polygons))
        if all_img_polygons[i].area >= area_threshold
    ]


def detect_holes(img_polygons, predicted_mask, buffer_pixels=1):
    """
    Returns lists of indices of empty and burnt polygons
    """
    list_holes = []
    list_true_polygons = []
    for j in range(len(img_polygons)):
        buffer_polygon = img_polygons[j].buffer(buffer_pixels)
        try:
            buffer_coord = [[int(x), int(y)] for x, y in buffer_polygon.exterior.coords]
        except:
            buffer_coord = [
                [int(x), int(y)]
                for z in [p for p in buffer_polygon]
                for x, y in z.exterior.coords
            ]
        zeros = 0
        ones = 0
        for i in range(len(buffer_coord)):
            try:
                if predicted_mask[buffer_coord[i][0], buffer_coord[i][1]] == 0:
                    zeros += 1
                else:
                    ones += 1
            except:
                continue
        if ones <= zeros:
            list_true_polygons.append(j)
        else:
            list_holes.append(j)
    return list_true_polygons, list_holes


def detect_inner_polygons(img_polygons, list_true_polygons, list_holes):
    """
    For all polygons from "list_true_polygons" it is calculated whether polygons from "list_holes" lie inside,
    returns a dictionary of type {outer polygon's index: inner polygon's indices}.
    """
    dict_area = {i: img_polygons[i].area for i in range(len(img_polygons))}
    dict_inner_polygons = {}
    for i in list_true_polygons:
        for j in list_holes:
            if dict_area[i] > dict_area[j]:
                if img_polygons[i].buffer(0.01).contains(img_polygons[j].buffer(0)):
                    try:
                        dict_inner_polygons[i].append(j)
                    except:
                        dict_inner_polygons[i] = [j]
    return dict_inner_polygons


def make_geo_polygons(dict_inner_polygons, geo_coordinates, list_true_polygons):
    """
    Returns a list of geopolygons of shapely.geometry.Polygon type, part of them with "holes" inside
    """
    final_list_polygons = []
    for k, v in dict_inner_polygons.items():
        final_polygon = shapely.geometry.Polygon(
            geo_coordinates[k], [geo_coordinates[x] for x in v]
        )
        final_list_polygons.append(final_polygon)

    list_true_polygons_not_in_dict = list(
        set(list_true_polygons) - set(([k for k, v in dict_inner_polygons.items()]))
    )
    for p in list_true_polygons_not_in_dict:
        final_list_polygons.append(shapely.geometry.Polygon(geo_coordinates[p]))
    return final_list_polygons


def plot(image, binary_mask, geo_polygons, rasterio_dataset, approx_polygons=None):
    """
    Show image and predicted polygons:
     - binary_mask - predicted binary mask,
     - "geo_polygons" - polygons with "holes" from "make_geo_polygons()",
     - "approx_polygons" - polygons without "holes" from "make_geo_polygons()"
     (coordinates_to_polygons(coordinates=[geo_coordinates[x] for x in list_true_polygons]))
    """
    if len(geo_polygons) != 0:
        mask = rasterio.mask.mask(
            rasterio_dataset,
            geo_polygons,
            invert=True,
            nodata=0,
            filled=False,
            indexes=1,
        )
        mask = np.where(mask[0].mask, 1, 0)
    else:
        mask = None
    if approx_polygons is not None and len(approx_polygons) != 0:
        approx_mask = rasterio.mask.mask(
            rasterio_dataset,
            approx_polygons,
            invert=True,
            nodata=0,
            filled=False,
            indexes=1,
        )
        approx_mask = np.where(approx_mask[0].mask, 1, 0)
        y, x = 2, 2
    else:
        y, x = 1, 3
    plt.figure(figsize=(20, 20))
    plt.subplot(y, x, 1)
    plt.title("Image")
    plt.imshow(image.transpose(1, 2, 0))
    plt.subplot(y, x, 2)
    plt.title("Predicted binary mask")
    plt.imshow(image.transpose(1, 2, 0))
    plt.imshow(binary_mask, alpha=0.3)
    plt.subplot(y, x, 3)
    if mask is not None:
        plt.title("Mask from geocoordinates")
        plt.imshow(mask)
    else:
        plt.title("No burnt area")
        plt.imshow(image.transpose(1, 2, 0))
    if approx_polygons is not None and len(approx_polygons) != 0:
        plt.subplot(y, x, 4)
        plt.title("Mask without holes")
        plt.imshow(approx_mask)
    plt.show()


def splot(image, binary_mask):
    """
    Show image and binary_mask).
    """
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image.transpose(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.title("Predicted binary mask")
    plt.imshow(image.transpose(1, 2, 0))
    plt.imshow(binary_mask, alpha=0.3)
    plt.show()