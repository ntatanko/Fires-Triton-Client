#!/usr/local/bin/python

import click
import numpy as np
import tritonclient.http as httpclient
from tqdm import tqdm

TRITON_HTTP_SERVICE_URL = "172.17.0.1:18000"
INPUT_NAME = "input"
OUTPUT_NAME = "output"


@click.command()
@click.option("--runs", default=100)
@click.option("--batch", default=8)
@click.option("--concurrency", default=1)
@click.option("--triton_service_url", default=TRITON_HTTP_SERVICE_URL)
def benchmark(runs, batch, concurrency, triton_service_url):

    # prepare sample data
    sample_input = np.random.uniform(0, 1, (batch, 3, 384, 384)).astype(np.float32)
    # sample_input = np.zeros((batch, 3, 384, 384), dtype=np.float32)

    # make requests to triton server

    inputs = [httpclient.InferInput(INPUT_NAME, sample_input.shape, "FP32")]
    outputs = [httpclient.InferRequestedOutput(OUTPUT_NAME, binary_data=True)]

    triton_client = httpclient.InferenceServerClient(
        url=triton_service_url,
        verbose=False,
        connection_timeout=600,
        network_timeout=600,
        concurrency=concurrency,
    )

    print(
        f"Making {runs} requests to TRITON server at {triton_service_url}"
        + f" with batch={sample_input.shape[0]}, "
        + f"concurrency={concurrency} ..."
    )

    with tqdm(
        total=runs * sample_input.shape[0], smoothing=0, unit=" frames"
    ) as progress:

        for _ in range(runs):
            inputs[0].set_data_from_numpy(sample_input, binary_data=True)

            _ = triton_client.infer(
                "fires",
                inputs,
                outputs=outputs,
                query_params={},
                headers=None,
                request_compression_algorithm="none",
                response_compression_algorithm="none",
            )

            progress.update(sample_input.shape[0])


if __name__ == "__main__":
    benchmark()
