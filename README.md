# Sample Client Code for Fires TRITON Server

## Starting Jupyter Lab

`$ docker/docker-forever.sh [--jupyter_port=####|8888]`

Then navigate to <http://localhost:8888> (replace 8888 if you specified another port).

Default access token is "mytoken". Edit `JUPYTER_TOKEN` argument in `docker/Dockerfile` to change it.

## Links

### TRITON Docs

<https://github.com/triton-inference-server>

### Client Sample Code

- <https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_infer_client.py>
- <https://github.com/triton-inference-server/client/blob/main/src/python/examples/image_client.py>

### Processing

 - example/detect.ipynb - code example
 - DETECT_ENG.md / DETECT_RUS.md - description
 - all models are in "models" folder
 - example/test_region_request.ipynb - example of downloading tiles from <https://www.sentinel-hub.com/> for test region
