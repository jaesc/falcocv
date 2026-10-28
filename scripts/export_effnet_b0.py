import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch
import torch.nn as nn
import timm
import onnx
import onnx.checker

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg")
model.eval()
dummy = torch.randn(1, 3, 224, 224)

try:
    torch.onnx.export(
        model,
        dummy,
        "models/EfficientNet-B0.onnx",
        opset_version=17,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
        do_constant_folding=True,
    )
except Exception as e:
    print(f"dynamo=True failed: {e}")
    print("Trying with dynamo=False...")
    torch.onnx.export(
        model,
        dummy,
        "models/EfficientNet-B0.onnx",
        opset_version=17,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )

m = onnx.load("models/EfficientNet-B0.onnx")
onnx.checker.check_model(m)
print("ONNX export OK: models/EfficientNet-B0.onnx")
out_shape = m.graph.output[0].type.tensor_type.shape
print(f"  Output shape: {out_shape}")
