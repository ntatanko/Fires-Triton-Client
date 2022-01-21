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

import segmentation_models_pytorch as smp
import torch
import torch.onnx
from torch.autograd import Variable


# -

"""
Convertation of model in .pth format to .onnx format for Fires-Triton-Server.
Models are based on  DeepLabV3+ (https://arxiv.org/abs/1802.02611) with 
efficientnet-b4 encoder from https://github.com/qubvel/segmentation_models.pytorch.git

"""

def pth_to_onnx(torch_model_path, onnx_path, model_name, device, img_size):
    assert img_size % 32 == 0, "img_size must be a multiple of 32"
    model = torch.load(torch_model_path, map_location=device)
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=0.0005),
        ]
    )
    state_dict = model.state_dict()
    ENCODER = "timm-efficientnet-b4"
    ACTIVATION = "sigmoid"
    DEVICE = device
    model = smp.DeepLabV3Plus(
        in_channels=3,
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=1,
        activation=ACTIVATION,
    )
    dummy_input = Variable(torch.randn(1, 3, img_size, img_size, dtype=torch.float32))
    model.load_state_dict(
        state_dict,
    )
    model.eval()

    if not os.path.exists(onnx_path):
        os.makedirs(onnx_path)
    torch.onnx.export(
        model,
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
    )