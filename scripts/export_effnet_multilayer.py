"""Export EfficientNet-B0 multi-layer feature extractor to ONNX.

Produces a 256-channel spatial feature map at 28×28 resolution from
a 224×224 input, combining block3 (stride 8, 40ch) and block4 (stride 16,
112ch) via 1×1 adapters, upsample, and concatenation.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = str(_PROJECT_ROOT / "models" / "EfficientNet-B0-MultiLayer.onnx")


class MultiLayerExtractor(nn.Module):
    """EfficientNet-B0 multi-layer feature extractor (spec §3.1).

    Input:  (1, 3, 224, 224)  BGR-normalized (RGB + ImageNet stats)
    Output: (1, 256, 28, 28)  L2-normalized per-patch
    """

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            out_indices=(2, 3),   # block3 (40ch, stride 8) + block4 (112ch, stride 16)
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.adapter3 = nn.Sequential(
            nn.Conv2d(40, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
        )
        self.adapter4 = nn.Sequential(
            nn.Conv2d(112, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
        )

        nn.init.xavier_uniform_(self.adapter3[0].weight)
        nn.init.xavier_uniform_(self.adapter4[0].weight)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        f3, f4 = features[0], features[1]               # (1,40,28,28), (1,112,14,14)

        f3 = self.adapter3(f3)                           # (1,128,28,28)
        f4 = self.adapter4(f4)                           # (1,128,14,14)
        f4_up = F.interpolate(f4, size=f3.shape[-2:],    # (1,128,28,28)
                              mode="bilinear",
                              align_corners=False)

        combined = torch.cat([f3, f4_up], dim=1)         # (1,256,28,28)
        combined = F.normalize(combined, dim=1)          # L2 per-channel
        return combined


def main():
    model = MultiLayerExtractor()
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            _OUTPUT_PATH,
            opset_version=17,
            input_names=["image"],
            output_names=["features"],
            dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
            do_constant_folding=True,
        )

    # Verify
    import onnx
    m = onnx.load(_OUTPUT_PATH)
    onnx.checker.check_model(m)
    out_shape = m.graph.output[0].type.tensor_type.shape
    print(f"Exported {_OUTPUT_PATH}")
    print(f"  Input:  {[d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"  Output: {[d.dim_value for d in out_shape.dim]}")

    # Quick inference test
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(_OUTPUT_PATH, providers=["CPUExecutionProvider"])
        result = sess.run(None, {"image": dummy.numpy()})
        print(f"  ONNX output shape: {result[0].shape}")
        print(f"  Min/max: {result[0].min():.4f} / {result[0].max():.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
