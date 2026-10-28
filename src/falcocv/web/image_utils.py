import cv2
import numpy as np
import base64


def encode_image(image: np.ndarray, quality: int = 85) -> str:
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def encode_image_png(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def encode_vision_image(vimg, quality: int = 85) -> dict:
    data = vimg.data if hasattr(vimg, 'data') else vimg
    if data is None:
        return {"data_b64": None, "width": 0, "height": 0, "overlays": []}
    h, w = data.shape[:2]
    b64 = encode_image(data, quality)
    overlays = []
    if hasattr(vimg, 'overlay_layers'):
        for name, layer in vimg.overlay_layers.items():
            if not layer.visible:
                continue
            for elem in layer.elements:
                overlays.append(elem.to_dict() if hasattr(elem, 'to_dict') else {})
    return {
        "data_b64": b64,
        "width": w,
        "height": h,
        "overlays": overlays,
    }
