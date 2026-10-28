"""Export ResNet18 truncated at layer3 to ONNX for faster PatchCore."""

import torch
import torch.onnx
import torchvision.models as models
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PROJECT_ROOT / "models"
_MODEL_DIR.mkdir(exist_ok=True)
_OUTPUT_PATH = _MODEL_DIR / "ResNet18-Layer3.onnx"


class ResNet18Layer3(torch.nn.Module):
    """ResNet18 truncated after layer3 (stride 16, 256 channels)."""

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = torch.nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
        )
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def main():
    model = ResNet18Layer3()
    model.eval()

    dummy_input = torch.randn(1, 3, 448, 448)

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
    print(f"  Input:  [batch, 3, 448, 448]")
    print(f"  Output: [batch, 256, 28, 28]  (stride 16)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(_OUTPUT_PATH))
    print(f"  ONNX inputs:  {[i.name for i in sess.get_inputs()]}")
    print(f"  ONNX outputs: {[o.name for o in sess.get_outputs()]}")


if __name__ == "__main__":
    main()
