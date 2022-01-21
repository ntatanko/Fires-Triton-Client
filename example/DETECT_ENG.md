# start
- start Fires-Triton-Server, for example docker/docker.sh --gpus='device=0' --triton_service_port=18000 --triton_metrics_port=18002 --triton_grpc_port=18001
- start Fires-Triton-Client, for example docker/docker-forever.sh --image_suffix=0 --gpus='device=0' --jupyter_port=8888 --tensorboard_port=6006
- Fires-Triton-Client/example/detect.ipynb - example of full tile processing (it is possible also to use model.pth without triton, then set TRITON = False)
- Fires-Triton-Client/example/DETECT_RUS.md / Fires-Triton-Client/example/DETECT_ENG.md - description of all functions


# utils/detect.py


## Tile processing logic:
 - tiff is transformed to numpy array,
 - image is cut into separate pictures 384 * 384 (for model in .pth format it is possible to use sizes that are multiple of 32),
 - for each of them model predict mask,
 - they are concatenated into a binary mask,
 - the mask is converted to polygons,
 - polygons are filtered by size,
 - internal and external polygons are detected,
 - according to this, the final geo-polygons are calculaited in shapely.geometry.Polygon format

### start_triton(max_batch_size, triton_http_service_url, img_size)
 - ***max_batch_size*** - maximum batch size, it is recomended to set it to 8, due to time calculation there is no point in max_batch_size bigger than 8,
 - ***triton_http_service_url*** - 172.17.0.1: host of docker container with Fires-Triton-Server, for example TRITON_HTTP_SERVICE_URL = "172.17.0.1:18000",
 - ***img_size*** - size of image for model, for each model in Triton it is fixed,
 - it is needed for the optimization at the first server start, takes a lot of time
 - you can instead define the model_warmup parameter in Fires-Triton-Server/models/fires/config.pbtxt for each batch size.

### get_dataset(path)
  - ***path*** - path to "tiff" file or to the folder with file "response.tiff",
  - returns rasterio dataset
 
 
### get_meta(path, chanels_for_model = ["B11", "B08", "B02"])
  - only if there is "request.json",
  - ***path*** - path to "request.json" or to the folder with "request.json",
  - pulled from request.json:
      - tile coordinates,
      - date,
      - all channels in the order in which they are in tiff (in request ["payload"] ["evalscript"] sample.B01, etc.),
      - channel indices ["B11", "B08", "B02"] (for rasterio.open().read()),
  - the output is a dictionary

### get_image(rasterio_dataset, chanels, img_size, clip_value = 10000, eps = 0.00001, disable_rasterio_warning=True)
 - ***rasterio_dataset*** from get_dataset(),
 - ***chanels*** - indices of ["B11", "B08", "B02"] channels in "tiff" file **(indexing starts with 1 !!!)**,
 - if there is no ***request.json*** from which the indices can be obtained, then you need to specify the channel indices ["B11", "B08", "B02"] in the tiff file in this order **(indexing starts with 1 !!!)**,
 - ***img_size*** image size for model, must be a multiple of 32,
 - rasterio_dataset.read(chanels) - in rasterio version 1.2.9, when reading channels, the warning that issued  does not belong to Warning class, therefore clear_output() was added, it is not convenient, but I did not find any other solution (or downgrade rasterio version),
 - ***disable_rasterio_warning*** - set to False if you want to plot results,
 - tiff is transformed to np.array,
 - values more than ***clip_value***  are clipped, channels are normalized (0-1),
 - the output is an image in format np.array(3, width, height) or ***None*** if:
     - np.max () == 0 (error in the date, the tile is black, there is no point in further processing),
     - tile width or height less than ***img_size***
        
### get_cut_coord(image, img_size, intersection = 0.1)
 - ***image*** image from ***get_image()***,
 - ***img_size*** image size for model,
 - calculate list of coordinates for cutting image into a set of small images with size img_size*img_size with (***intersection***),
 - output is a list of coordinates for cutting image

### cut_image(image, sliding_window_coordinates, norm = True)
 - ***image*** image from ***get_image()***,
 - ***sliding_window_coordinates*** - coordinate list from ***get_cut_coord ()***,
 - output is an array of small images from the original image according to coordinates from ***sliding_window_coordinates*** of type "float32",
 - the values of each channel are normalized to the range 0-1 if ***norm = True***

### calc_batches(array_images, max_batch_size=8)
 - ***array_images*** images in np.array format from ***cut_image()***,
 - ***max_batch_size*** - maximum batch size, due to time calculation there is no point in max_batch_size bigger than 8,
 - groups images into batches of [max_batch_size, 4, 1] sizes, it is nessesary for Triton, because of the batch optimization,
 - return image indexes for batches

### triton_predict(array_images, triton_http_service_url, max_batch_size)
 - ***array_images***   images in np.array format from ***cut_image()***,
 - ***triton_http_service_url*** - 172.17.0.1 host of docker container with Fires-Triton-Server, for example TRITON_HTTP_SERVICE_URL = "172.17.0.1:18000",
 set on Fires-Triton-Server start, ___docker/docker.sh --gpus='device=1' --triton_service_port=18000 --triton_metrics_port=18002 --triton_grpc_port=18001___,
 - ***max_batch_size*** maximum batch size, assigned in Fires-Triton-Server/models/fires/config.pbtxt,
 - output is an array of model predictions for each image from ***array_images***

### get_torch_model(model_path, device)
 - ***model_path*** path to the model in ".pth" format,
 - ***device*** - 'cuda' or 'cpu',
 - model based on DeepLabV3+ (https://arxiv.org/abs/1802.02611) with efficientnet-b4 encoder from https://github.com/qubvel/segmentation_models.pytorch.git
 - returns model and device 

### torch_predict(model, device, array_images,max_batch_size)
 - ***model, device*** model and processor type from *get_torch_model()*,
 - ***array_images*** images from ***cut_image()***,
 - ***max_batch_size*** maximum batch size,
 - output is an array of model predictions for each image from ***array_images***

### prediction_to_mask(img_size, list_masks, sliding_window_coordinates, threshold = 0.9)
 - ***list_masks*** array or list of predictions for all images from ***torch_predict()*** or ***triton_predict()***,
 - ***threshold*** - threshold for assigning a point to True or False, in fact, not a very necessary thing, most of the values are very close to 0 and 1, but sometimes there is a difference between 0.99 and 0.99999,
 - converts each predicted mask into a binary, concatenate them to the mask of original image size

### mask_to_polygon_coordinates(binary_mask, tolerance = 0)
 - ***binary_mask*** mask from ***prediction_to_mask()***,
 - converts the binary mask to polygons coordinates along the border 1/0

### get_geo_coordinates(rasterio_dataset, img_coordinates)
 - coordinates of polygons are converted to geo-coordinates,
 - returns a list of geocoordinates, among them there are both external and internal polygons

### coordinates_to_polygons(coordinates)
 - ***coordinates*** - coordinates from ***get_geo_coordinates()*** or ***mask_to_polygon_coordinates()***,
 - coordinates are converted to polygons of shapely.geometry.Polygon type,
 - returns a list of polygons

### area_filter(all_img_polygons, area_threshold = 100)
 - ***all_img_polygons*** - all polygons of type shapely.geometry.Polygon in image coordinates,
 - ***area_threshold*** - threshold for determining the minimum area, in square pixels,
 - since 1 pixel is 10x10m, it is equivalent to 100 * area_threshold square meters, area of 1 hectare is equivalent to area_threshold=100,
 - makes sense, as there are often a lot of small polygons and their processing takes a lot of time

### detect_holes(img_polygons, predicted_mask, buffer_pixels = 1)
 - ***img_polygons*** - polygons of type shapely.geometry.Polygon in image coordinates,
 - ***predicted_mask*** - predicted binary mask for the whole image,
 - a buffer polygon with a width of ***buffer_pixels*** pixels is created around the polygons,
 - for each point of this polygon, it is calculated whether it is 1 or 0 in the predicted mask,
 - if the majority of points are 1, then there is an empty polygon inside,
 - if the majority of points are 0, then inside is the burned-out area,
 - returns lists of indices of empty and burnt polygons

### detect_inner_polygons(img_polygons, list_true_polygons, list_holes)
 - ***img_polygons*** - polygons of type shapely.geometry.Polygon in image coordinates,
 - ***list_true_polygons, list_holes*** lists of indices of empty and burnt polygons
 - for all polygons from ***list_true_polygons*** it is calculated whether polygons from list_holes lie inside,
 - returns a dictionary of type {outer polygon index: inner polygon indices},
 - the slowest stage of all
 - if skip this stage, it is possible quickly translate geocoordinates from list_true_polygons into geopolygons without "holes" of the shapely.geometry.Polygon type:
```
 geo_polygons = coordinates_to_polygons(coordinates=[geo_coordinates[x] for x in list_true_polygons])
```

### make_geo_polygons(dict_inner_polygons, geo_coordinates, list_true_polygons)
 - ***dict_inner_polygons*** dictionary from * detect_inner_polygons () *,
 - ***geo_coordinates*** list of geo-coordinates of polygons,
 - ***list_true_polygons*** - lists of indices of burnt polygons,
 - returns a list of geopolygons of the shapely.geometry.Polygon type, the part with "holes" inside

### plot(image, binary_mask, geo_polygons, rasterio_dataset, approx_polygons = None)
 - ***image*** image from ***get_image()***,
 - ***binary_mask*** mask from ***prediction_to_mask()***,
 - ***geo_polygons*** with "holes" from ***make_geo_polygons()***,
 - ***rasterio_dataset*** from get_dataset(),
 - ***approx_polygons*** polygons of type shapely.geometry.Polygon, without "holes" inside,
```
 coordinates_to_polygons(coordinates=[geo_coordinates[x] for x in list_true_polygons])
```
 - show images and predicted polygons

### splot(image, binary_mask)
 - ***image*** image from ***get_image()***,
 - ***binary_mask*** mask from ***prediction_to_mask()***,
 - show predicted polygons (binary_mask)