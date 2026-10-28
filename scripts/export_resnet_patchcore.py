"""Export ResNet18 truncated at layer2 to ONNX for PatchCore anomaly detection."""

import torch
import torch.onnx
import torchvision.models as models
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PROJECT_ROOT / "models"
_MODEL_DIR.mkdir(exist_ok=True)
_OUTPUT_PATH = _MODEL_DIR / "ResNet18-Layer2.onnx"


class ResNet18Layer2(torch.nn.Module):
    """ResNet18 truncated after layer2 (stride 8, 128 channels)."""

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Keep up to and including layer2
        self.features = torch.nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
        )
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def main():
    model = ResNet18Layer2()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            _OUTPUT_PATH,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["features"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "features": {0: "batch", 2: "height", 3: "width"},
            },
            dynamo=False,
        )

    print(f"Exported {_OUTPUT_PATH}")
    print(f"  Input:  [batch, 3, H, W] (dynamic H, W)")
    print(f"  Output: [batch, 128, H/8, W/8]")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(str(_OUTPUT_PATH))
    print(f"  ONNX inputs:  {[i.name for i in sess.get_inputs()]}")
    print(f"  ONNX outputs: {[o.name for o in sess.get_outputs()]}")


if __name__ == "__main__":
    main()
