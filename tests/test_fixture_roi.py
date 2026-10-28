#!/usr/bin/env python3
"""
Test script to verify fixture-aware ROI handling
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import composite_roi_to_full, extract_roi
from falcocv.tools.filters.logic import FiltersTool


def _pattern_image(h=200, w=240):
    """Smooth pattern so warp resampling error stays tiny."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    img[..., 0] = (xx * 255 // max(1, w - 1)).astype(np.uint8)
    img[..., 1] = (yy * 255 // max(1, h - 1)).astype(np.uint8)
    img[..., 2] = 128
    cv2.circle(img, (w // 2, h // 2), min(h, w) // 4, (255, 255, 255), -1)
    cv2.rectangle(img, (10, 10), (40, 30), (0, 0, 255), -1)
    return img

def test_fixture_conversions():
    """Test the fixture_to_image and image_to_fixture conversions"""
    print("Testing fixture-aware ROI conversions...")
    
    # Create a test image
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (10, 10), (90, 90), (255, 255, 255), -1)
    
    # Create a VisionImage with fixture offset and rotation
    vision_img = VisionImage(test_img, x=10.0, y=20.0, angle=30.0)
    
    # Test ROI relative to fixture
    fixture_roi = [5.0, 5.0, 20.0, 20.0, 0.0]  # x, y, w, h, angle relative to fixture
    
    print(f"Original fixture ROI: {fixture_roi}")
    
    # Convert to image coordinates
    image_roi = vision_img.fixture_to_image(fixture_roi)
    print(f"Converted to image ROI: {image_roi}")
    
    # Convert back to fixture coordinates
    back_to_fixture = vision_img.image_to_fixture(image_roi)
    print(f"Converted back to fixture ROI: {back_to_fixture}")
    
    # Check if we got back to the original (within floating point precision)
    tolerance = 0.001
    for i in range(len(fixture_roi)):
        diff = abs(fixture_roi[i] - back_to_fixture[i])
        assert diff <= tolerance, (
            f"Mismatch at index {i}: {fixture_roi[i]} vs {back_to_fixture[i]}"
        )
    
    print("SUCCESS: Round-trip conversion is accurate!")

def test_roi_extraction():
    """Test that ROI extraction works with fixture-relative coordinates"""
    print("\nTesting ROI extraction with fixture-relative coordinates...")
    
    # Create a test image with a distinctive pattern
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (25, 25), (75, 75), (0, 255, 0), -1)  # Green square
    cv2.circle(test_img, (50, 50), 10, (0, 0, 255), -1)  # Red circle in center
    
    # Create VisionImage with fixture at center of image, no rotation
    vision_img = VisionImage(test_img, x=50.0, y=50.0, angle=0.0)
    
    # Define a fixture-relative ROI that should capture the red circle
    # Circle is at (50,50) in image coordinates, which is (0,0) in fixture coordinates
    # Circle radius is 10, so we want a 20x20 ROI centered at fixture origin
    fixture_roi = [-10.0, -10.0, 20.0, 20.0, 0.0]  # x, y, w, h, angle
    
    print(f"Fixture-relative ROI: {fixture_roi}")
    
    # Extract the ROI
    roi_img = vision_img.get_roi_data(fixture_roi)
    assert roi_img is not None, "Failed to extract ROI"
        
    print(f"Extracted ROI shape: {roi_img.shape}")
    
    # Check that the extracted ROI contains the red circle
    # Convert to grayscale and check for red pixels
    gray_roi = cv2.cvtColor(roi_img.data if isinstance(roi_img, VisionImage) else roi_img, 
                           cv2.COLOR_BGR2GRAY)
    red_pixels = cv2.countNonZero(gray_roi)
    
    print(f"Number of non-zero pixels in ROI: {red_pixels}")
    
    assert red_pixels > 50, "ROI extraction did not capture expected region"
    print("SUCCESS: ROI extraction captured the expected region!")


def test_upright_roi_on_rotated_fixture_is_cropped_exactly():
    """Regression: an ROI drawn upright on a rotated fixture.

    The ROI editor stores it fixture-relative, so roi[4] == -fixture.angle and
    total_angle is exactly 0.  get_roi_data() used to ignore the fixture
    rotation in that case and crop ~97 px away from the requested region.
    """
    image = _pattern_image()
    angle = -19.43755792261345
    v_img = VisionImage(image, x=120.0, y=90.0, angle=angle)
    roi = [-60.0, -45.0, 80.0, 60.0, -angle]
    assert v_img.angle + roi[4] == 0.0

    abs_x, abs_y, _, _, _ = v_img.fixture_to_image(roi)
    expected = image[
        int(abs_y):int(abs_y) + int(roi[3]),
        int(abs_x):int(abs_x) + int(roi[2]),
    ].copy()

    crop = v_img.get_roi_data(roi)
    assert np.array_equal(crop.data, expected)

    # extract_roi() must agree with the crop it returns
    extracted = extract_roi(v_img, roi)
    assert (extracted["ax"], extracted["ay"]) == (int(abs_x), int(abs_y))

    # and compositing back must be a lossless round trip
    back = composite_roi_to_full(crop.data, v_img, roi, draw_roi=False)
    assert np.array_equal(back.data, image)


def test_composite_with_rotated_fixture_at_origin():
    """Latent case: total_angle == 0 with the fixture origin at (0, 0)."""
    image = _pattern_image()
    v_img = VisionImage(image, x=0.0, y=0.0, angle=30.0)
    roi = [40.0, 10.0, 30.0, 20.0, -30.0]
    assert v_img.angle + roi[4] == 0.0

    crop = v_img.get_roi_data(roi)
    back = composite_roi_to_full(crop.data, v_img, roi, draw_roi=False)
    assert np.array_equal(back.data, image)


def test_rotated_roi_roundtrip_is_subpixel():
    """Rotated ROIs still go through the warp path and stay pixel-faithful."""
    image = _pattern_image()
    v_img = VisionImage(image, x=120.0, y=90.0, angle=12.0)
    roi = [-50.0, -40.0, 90.0, 70.0, 15.0]

    crop = v_img.get_roi_data(roi)
    crop_data = crop.data if isinstance(crop, VisionImage) else crop
    back = composite_roi_to_full(crop_data, v_img, roi, draw_roi=False)

    diff = np.abs(back.data.astype(np.int16) - image.astype(np.int16))
    assert float(diff.mean()) < 0.5


def test_filters_roi_on_rotated_fixture_processes_the_right_pixels():
    """End-to-end: the filtered pixels must match the requested ROI."""
    image = _pattern_image()
    angle = -19.43755792261345
    v_img = VisionImage(image, x=120.0, y=90.0, angle=angle)
    roi = [-60.0, -45.0, 80.0, 60.0, -angle]

    tool = FiltersTool()
    tool.inputs["input image"] = v_img
    tool.params["roi"] = roi
    tool.params["filters"] = [{"enabled": True, "type": "invert"}]
    tool.run()

    abs_x, abs_y, _, _, _ = v_img.fixture_to_image(roi)
    ix, iy = int(abs_x), int(abs_y)
    rh, rw = int(roi[3]), int(roi[2])

    expected = image.copy()
    expected[iy:iy + rh, ix:ix + rw] = cv2.bitwise_not(expected[iy:iy + rh, ix:ix + rw])

    assert np.array_equal(tool.outputs["output image"].data, expected)

if __name__ == "__main__":
    import sys as _sys
    ok = True
    try:
        test_fixture_conversions()
    except AssertionError as exc:
        print(f"FAIL: fixture conversions: {exc}")
        ok = False
    try:
        test_roi_extraction()
    except AssertionError as exc:
        print(f"FAIL: ROI extraction: {exc}")
        ok = False
    print("All tests PASSED!" if ok else "Some tests FAILED!")
    _sys.exit(0 if ok else 1)
