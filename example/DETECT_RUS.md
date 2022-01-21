# start
- start Fires-Triton-Server, for example docker/docker.sh --gpus='device=0' --triton_service_port=18000 --triton_metrics_port=18002 --triton_grpc_port=18001
- start Fires-Triton-Client, for example docker/docker-forever.sh --image_suffix=0 --gpus='device=0' --jupyter_port=8888 --tensorboard_port=6006
- Fires-Triton-Client/example/detect.ipynb - example of full tile processing (it is possible also to use model.pth without triton, then set TRITON = False)
- Fires-Triton-Client/example/DETECT_RUS.md / Fires-Triton-Client/example/DETECT_ENG.md - description of all functions


# utils/detect.py

## Логика обработки тайлов:
 - tiff переводится в numpy,
 - режется на отдельные картинки размером 384*384 (для модели в формате .pth можно любой размер кратный 32),
 - для каждой из них получается предсказание модели,
 - они склеиваются в бинарную маску,
 - маска переводится в полигоны,
 - полигоны фильтруются по размеру,
 - находится список внутренних и внешних полигонов,
 - по этому списку выдаются финальные геополигоны в формате shapely.geometry.Polygon


### start_triton(max_batch_size, triton_http_service_url, img_size)
 - ***max_batch_size*** - максимальный размер батча, рекомендуется 8 (согласно вычислениям времени, нет смысла делать больше 8),
 - ***triton_http_service_url*** - хост докер-контейнера с Fires-Triton-Server, например TRITON_HTTP_SERVICE_URL = "172.17.0.1:18000",
 - ***img_size*** - размер изображения для модели, в Triton он фиксированный,
 - если использовать ансамбли, то для каждой модели и каждого размера нужно делать отдельную модель в формате .onnx, я до конца не разобралась, как это организовать,
 - запускается один раз для оптимизации на старте, занимает много времени,
 - можно вместо этого прописать в Fires-Triton-Server/models/fires/config.pbtxt параметр model_warmup для каждого размера батча.

### get_dataset(path)
 - ***path*** - путь к файлу в формате "tiff" или к папке с файлом "response.tiff",
 - возвращает rasterio dataset

### get_meta(path, chanels_for_model=["B11", "B08", "B02"])
 - если нет request.json, то эта функция не нужна,
 -  ***path*** - путь к файлу "request.json" или к папке с файлом "request.json",
 - из request.json вытаскиваются:
     - координаты тайла,
     - дата,
     - все каналы в том порядке, в котором они в tiff (в request["payload"]["evalscript"] sample.B01 и тд),
     - индексы каналов ["B11", "B08", "B02"] (для rasterio.open().read()),
 - на выходе словарь

### get_image(rasterio_dataset, chanels, img_size, clip_value=10000, eps=0.00001, disable_rasterio_warning=True)
 - ***rasterio_dataset*** из ***get_dataset()***
 - ***chanels*** - индексы каналов ["B11", "B08", "B02"] в tiff-файле **(индексация начинается с 1!!!)**
 - если нет ***request.json***, из которого можно считать индексы, то нужно указать индексы каналов ["B11", "B08", "B02"] в tiff-файле именно в таком порядке **(индексация начинается с 1!!!)**
 - ***img_size*** размер изображения для модели, должен быть кратен 32,
 - rasterio_dataset.read(chanels) - в версии rasterio 1.2.9 при чтении каналов выдается предупреждение, не относящееся к классам Warning, поэтому добавлено clear_output(), не удобно, но других вариантов не нашла (можно понизить версию rasterio),
 - ***disable_rasterio_warning*** - False, если нужно визуализировать результаты в той же ячейке,
 - tiff переводится в np.array,
 - значения больше ***clip_value*** обрезаются, каналы нормализуются (0-1),
 - на выходе изображение в формате np.array(3, width, height) или ***None*** если:
      - np.max()==0 (ошибка в дате, тайл черный, нет смысла делать предсказание),
      - ширина или высота тайла меньше ***img_size***
        
### get_cut_coord(image, img_size, intersection=0.1)
 - ***image*** изображение из get_image,
 - ***img_size*** размер изображения для модели,
 - разбивает изображение на сетку мелких размером ***img_size***х***img_size*** с пересечением (***intersection***),
 - выдает лист координат для разбивки

### cut_image(image, sliding_window_coordinates, norm=True)
 - ***image*** изображение из ***get_image()***,
 - ***sliding_window_coordinates*** - лист координат из ***get_cut_coord()***,
 - выдает np.array из изображений по координатам из ***sliding_window_coordinates*** типа "float32",
 - значения каждого канала приводятся к диапазону 0-1 если ***norm=True***

### calc_batches(array_images, max_batch_size=8)
 - ***array_images*** - изображения из ***cut_image()***,
 - ***max_batch_size*** - максимальный размер батча, согласно тестированию, нет смысла делать больше 8,
 - группирует изображения в батчи размером [max_batch_size, 4, 1] это необходимо для оптимизации Triton,
 - возвращает индексы изображений для батчей

### triton_predict(array_images, triton_http_service_url, max_batch_size)
 - ***array_images*** изображения из ***cut_image()***,
 - ***triton_http_service_url*** - 172.17.0.1 хост докер-контейнера с Fires-Triton-Server, например TRITON_HTTP_SERVICE_URL = "172.17.0.1:18000",  
 прописывантся при запуске Fires-Triton-Server,  ___docker/docker.sh --gpus='device=1' --triton_service_port=18000 --triton_metrics_port=18002 --triton_grpc_port=18001___,
 - ***max_batch_size*** максимальный размер батча, приписывается в Fires-Triton-Server/models/fires/config.pbtxt,
 - возвращает предсказание модели для всех изображений из ***array_images***

### get_torch_model(model_path, device)
 - ***model_path*** путь к модели в формате ".pth",
 - ***device***  - 'cuda' или 'cpu',
 - модель на основе DeepLabV3+ (https://arxiv.org/abs/1802.02611) с энкодером efficientnet-b4 из https://github.com/qubvel/segmentation_models.pytorch.git
 - возвращает model и device

### torch_predict(model, device, array_images, max_batch_size)
 - ***model, device*** модель и тип процессора из *get_torch_model*,
 - ***array_images*** изображения из ***cut_image()***,
  - ***max_batch_size*** максимальный размер батча,
 - возвращает предсказание модели для всех изображений из ***array_images***

### prediction_to_mask(img_size, list_masks, sliding_window_coordinates, threshold=0.9)
 - ***list_masks*** array или list предсказаний для всех изображений из ***torch_predict()*** или ***triton_predict()***,
 - ***threshold*** - порог для отнесения точки к True или False, на самом деле не очень нужная вещь, большинство значений очень близки к 0 и 1, но иногда есть разница между 0,99 и 0,99999,
 - каждую маску переводит в бинарную, складывает их в маску размером с исходное изображение

### mask_to_polygon_coordinates(binary_mask, tolerance=0)
 - ***binary_mask*** маска из ***prediction_to_mask()***,
 - переводит бинарную маску в координаты полигонов относительно изображения по границе 1/0

### get_geo_coordinates(rasterio_dataset, img_coordinates)
 - координаты полигонов относительно изображения переводятся в геокоординаты относительно тайла,
 - возвращает список координат замкнутых полигонов, среди которых есть и внешние и внутренние

### coordinates_to_polygons(coordinates)
 - ***coordinates*** - координаты полигонов из ***get_geo_coordinates()*** или ***mask_to_polygon_coordinates()***,
 - координаты полигонов переводятся в полигоны типа shapely.geometry.Polygon,
 - возвращает список полигонов

### area_filter(all_img_polygons, area_threshold=100)
 - ***all_img_polygons*** - все полигоны типа shapely.geometry.Polygon в координатах изображения, 
 - ***area_threshold*** - порог для определения минимальной площади, в пикселях^2,
 - так как 1 пиксель 10х10м, то равнозначно 100 * area_threshold м^2, площадь 1га - area_threshold=100,
 - очень имеет смысл, так как часто бывает много мелких полигонов и их обработка занимает много времени

### detect_holes(img_polygons, predicted_mask, buffer_pixels=1)
 - ***img_polygons*** - полигоны типа shapely.geometry.Polygon в координатах изображения,
 - ***predicted_mask*** - пердсказанная бинарная маска для целого изображения,
 - вокруг полигонов создается буферный полигон шириной ***buffer_pixels*** пикселей,
 - для каждой точки этого полигона вычисляется, попадает она на 1 или на 0 в предсказанной маске,
 - если большинство точек 1, то внутри пустой полигон,
 - если большинство точек 0, то внутри выгоревшая территория,
 - выдает списки индексов пустых и выгоревших полигонов

### detect_inner_polygons(img_polygons, list_true_polygons, list_holes)
 - ***img_polygons*** - полигоны типа shapely.geometry.Polygon в координатах изображения,
 - ***list_true_polygons, list_holes*** списки индексов пустых и выгоревших полигонов
 - для всех полигонов из list_true_polygons вычисляется, лежат ли полигоны из list_holes внутри,
 - выдает словарь типа {индекс внешнего полигона: индексы внутренных полигонов},
 - самый медленный этап из всех
 - я пробовала искать внутренние полигоны через геополигоны, результат не стабильный, поэтому остановилась на этом варианте,
 - если пропустить этот этап, то можно быстро перевести геокоординаты из list_true_polygons в геополигоны без "дырок" типа shapely.geometry.Polygon:
```
 geo_polygons = coordinates_to_polygons(coordinates=[geo_coordinates[x] for x in list_true_polygons])
```

### make_geo_polygons(dict_inner_polygons, geo_coordinates, list_true_polygons)
 - ***dict_inner_polygons*** словарь из *detect_inner_polygons()*,
 - ***geo_coordinates*** список геокоординат полигонов,
 - ***list_true_polygons*** - списки индексов выгоревших полигонов,
 - возврашает список геополигонов типа shapely.geometry.Polygon, часть с "дырками" внутри

### plot(image, binary_mask, geo_polygons, rasterio_dataset, approx_polygons=None)
 - ***image*** изображение из get_image,
 - ***binary_mask*** маска из ***prediction_to_mask()***,
 - ***geo_polygons*** геополигоны из ***make_geo_polygons()***, 
 - ***rasterio_dataset*** из ***get_dataset()***,
 - ***approx_polygons*** полигоны типа shapely.geometry.Polygon, без "дырок" внутри,
 ```
 coordinates_to_polygons(coordinates=[geo_coordinates[x] for x in list_true_polygons])
```
 - показывает предсказание модели, если approx_polygons=None, то показывает бинарную маску и геополигоны с "дырками"

### splot(image, binary_mask)
 - ***image*** изображение из get_image,
 - ***binary_mask*** маска из ***prediction_to_mask()***,
 - показывает предсказание модели (бинарную маску)