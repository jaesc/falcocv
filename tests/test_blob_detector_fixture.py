#!/usr/bin/env python3
"""
Test script for Blob Detector with fixture-aware ROI handling
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from falcocv.core.vision_image import VisionImage
from falcocv.tools.blob_detector import BlobDetector

def test_blob_detector_with_fixture():
    """Test Blob Detector with a VisionImage input that has a fixture"""
    print("Testing Blob Detector with fixture-aware ROI...")
    
    # Create a test image with a green circle
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(test_img, (50, 50), 15, (0, 255, 0), -1)  # Green circle
    
    # Create a VisionImage with a fixture (offset and rotation)
    vision_img = VisionImage(test_img, x=50.0, y=50.0, angle=0.0)
    
    # Create a Blob Detector tool
    detector = BlobDetector("Blob Detector")
    detector.params["roi"] = [0.0, 0.0, 100.0, 100.0, 0.0]  # Full image ROI in fixture coordinates
    detector.inputs["input image"] = vision_img
    
    # Run the detector
    detector.run()
    
    # Check results
    results = detector.outputs["results"]
    print(f"Blob Detector found {results.Count} blobs")
    
    if results.Count > 0:
        for i, res in enumerate(results):
            print(f"  Blob {i}: x={res.x:.1f}, y={res.y:.1f}, area={res.area:.1f}")
        print("SUCCESS: Blob Detector works with fixture-aware ROI!")
    else:
        print("ERROR: Blob Detector did not find any blobs")
    assert results.Count > 0, "Blob Detector did not find any blobs"

if __name__ == "__main__":
    import sys as _sys
    try:
        test_blob_detector_with_fixture()
    except AssertionError as exc:
        print(f"Blob Detector test FAILED: {exc}")
        _sys.exit(1)
    print("Blob Detector test PASSED!")
    _sys.exit(0)
