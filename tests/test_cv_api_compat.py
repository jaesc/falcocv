"""OpenCV API compatibility guard.

Every cv2 symbol FalcoCV depends on must exist in the installed opencv-python
build.  If a future upgrade removes or renames one of these, this test fails
loudly up-front instead of crashing inside a tool at runtime.
"""
import cv2

# Top-level cv2 attributes (functions, classes and constants) used across the
# codebase (tools, core, web and tests).
REQUIRED_CV2_ATTRS = [
    "ADAPTIVE_THRESH_GAUSSIAN_C",
    "BORDER_REFLECT_101",
    "COLORMAP_JET",
    "COLOR_BGR2GRAY",
    "COLOR_BGR2HLS",
    "COLOR_BGR2HSV",
    "COLOR_BGR2RGB",
    "COLOR_BGRA2BGR",
    "COLOR_GRAY2BGR",
    "COLOR_RGB2BGR",
    "CV_16S",
    "CV_32F",
    "Canny",
    "FONT_HERSHEY_SIMPLEX",
    "GaussianBlur",
    "IMREAD_COLOR",
    "IMREAD_GRAYSCALE",
    "IMREAD_UNCHANGED",
    "IMWRITE_JPEG_QUALITY",
    "INTER_AREA",
    "INTER_CUBIC",
    "INTER_LANCZOS4",
    "INTER_LINEAR",
    "INTER_NEAREST",
    "MORPH_CLOSE",
    "MORPH_CROSS",
    "MORPH_DILATE",
    "MORPH_ELLIPSE",
    "MORPH_ERODE",
    "MORPH_GRADIENT",
    "MORPH_OPEN",
    "MORPH_RECT",
    "NORM_MINMAX",
    "QRCodeEncoder",
    "ROTATE_180",
    "ROTATE_90_CLOCKWISE",
    "ROTATE_90_COUNTERCLOCKWISE",
    "Sobel",
    "THRESH_BINARY",
    "THRESH_BINARY_INV",
    "THRESH_OTSU",
    "TM_CCOEFF_NORMED",
    "absdiff",
    "adaptiveThreshold",
    "addWeighted",
    "applyColorMap",
    "arcLength",
    "aruco",
    "bilateralFilter",
    "bitwise_not",
    "circle",
    "contourArea",
    "countNonZero",
    "createCLAHE",
    "cvtColor",
    "equalizeHist",
    "fillPoly",
    "filter2D",
    "flip",
    "getPerspectiveTransform",
    "getRotationMatrix2D",
    "getStructuringElement",
    "imdecode",
    "imencode",
    "imread",
    "imwrite",
    "magnitude",
    "matchTemplate",
    "medianBlur",
    "minMaxLoc",
    "moments",
    "morphologyEx",
    "normalize",
    "putText",
    "pyrDown",
    "rectangle",
    "remap",
    "resize",
    "rotate",
    "split",
    "threshold",
    "warpAffine",
    "warpPerspective",
]

# cv2.aruco sub-API used to generate ArUco markers in the barcode tests.
REQUIRED_ARUCO_ATTRS = [
    "DICT_6X6_250",
    "generateImageMarker",
    "getPredefinedDictionary",
]


def test_cv2_major_version():
    major = int(cv2.__version__.split(".")[0])
    assert major >= 5, f"expected OpenCV 5.x, got {cv2.__version__}"


def test_required_cv2_attrs_present():
    missing = sorted(n for n in REQUIRED_CV2_ATTRS if not hasattr(cv2, n))
    assert not missing, f"missing cv2 attributes in {cv2.__version__}: {missing}"


def test_qr_encoder_api():
    assert hasattr(cv2.QRCodeEncoder, "Params")
    assert hasattr(cv2.QRCodeEncoder, "create")


def test_aruco_api():
    missing = [n for n in REQUIRED_ARUCO_ATTRS if not hasattr(cv2.aruco, n)]
    assert not missing, f"missing cv2.aruco attributes: {missing}"
