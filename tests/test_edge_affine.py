#!/usr/bin/env python3
"""
Test affine caliper edge detection.

Synthetic image: white 100x100 square on black 400x400 background.
Verifies correct edge positions for all 4 sides and rotation angles.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from falcocv.core.vision_image import VisionImage
from falcocv.tools.edge_detection import EdgeDetection

def make_img():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (150, 150), (250, 250), (255, 255, 255), -1)
    return img

def run_tool(vimg, roi, polarity="Any", mode="Single Edge", e1_pol=None, e2_pol=None,
             min_w=3, max_w=9999, thresh=10, sigma=1.0):
    tool = EdgeDetection()
    tool.inputs["input image"] = vimg
    tool.params["detect_mode"] = mode
    tool.params["single_polarity"] = polarity
    if e1_pol:
        tool.params["edge1_polarity"] = e1_pol
    if e2_pol:
        tool.params["edge2_polarity"] = e2_pol
    tool.params["edge_threshold"] = thresh
    tool.params["min_edge_width"] = min_w
    tool.params["max_edge_width"] = max_w
    tool.params["gaussian_sigma"] = sigma
    tool.params["subpixel"] = True
    tool.params["roi"] = roi
    tool.run()
    count = tool.outputs["count"]
    results = tool.outputs["results"]
    return results[0] if count > 0 and len(results) > 0 else None

def _pol_abbr(pol):
    return pol.replace("\u2192", "->")

def check_edge(label, vimg, roi, expected_x, expected_y, angles, polarities, tol=2.0):
    fail = False
    for a, pol in zip(angles, polarities):
        r = run_tool(vimg, list(roi) + [a], pol)
        pol_str = _pol_abbr(pol)
        if r is None:
            print(f"  FAIL {label} angle={a:3d} pol={pol_str}: NO EDGE (expected near ({expected_x},{expected_y}))")
            fail = True
            continue
        dx = abs(r.x - expected_x)
        dy = abs(r.y - expected_y)
        ok = dx <= tol and dy <= tol
        if not ok:
            print(f"  FAIL {label} angle={a:3d} pol={pol_str}: pos=({r.x:.1f},{r.y:.1f}) "
                  f"expected ~({expected_x},{expected_y})  dx={dx:.1f} dy={dy:.1f}")
            fail = True
        else:
            print(f"  PASS {label} angle={a:3d} pol={pol_str}: pos=({r.x:.1f},{r.y:.1f})")
    return not fail

def test_horizontal_edges():
    """Top edge at y=150, detected with ROI scanning vertically (+Y / -Y)."""
    print("\n=== HORIZONTAL EDGES ===")
    ok = True
    vimg = VisionImage(make_img())

    # Top edge y=150. ROI centered on (200,150), aw=60, ah=40.
    # theta=0: v=(0,1), scan down. Black->White = D->L.
    # theta=180: v=(0,-1), scan up. White->Black = L->D.
    ok &= check_edge("Top", vimg, [170, 130, 60, 40], 200, 150,
                     [0, 180], ["Dark\u2192Light", "Light\u2192Dark"])

    # Bottom edge y=250. ROI centered on (200,250).
    # theta=0: scan down. White->Black = L->D.
    # theta=180: scan up. Black->White = D->L.
    ok &= check_edge("Bottom", vimg, [170, 230, 60, 40], 200, 250,
                     [0, 180], ["Light\u2192Dark", "Dark\u2192Light"])
    assert ok

def test_vertical_edges():
    """Left/right edges at x=150/250, detected with ROI scanning horizontally."""
    print("\n=== VERTICAL EDGES ===")
    ok = True
    vimg = VisionImage(make_img())

    # Left edge x=150. ROI centered on (150,200), aw=40, ah=60.
    # theta=90: v=(-1,0), scan left. White->Black = L->D.
    # theta=270: v=(1,0), scan right. Black->White = D->L.
    ok &= check_edge("Left", vimg, [130, 170, 40, 60], 150, 200,
                     [90, 270], ["Light\u2192Dark", "Dark\u2192Light"])

    # Right edge x=250. ROI centered on (250,200).
    # theta=90: scan left. Black->White = D->L.
    # theta=270: scan right. White->Black = L->D.
    ok &= check_edge("Right", vimg, [230, 170, 40, 60], 250, 200,
                     [90, 270], ["Dark\u2192Light", "Light\u2192Dark"])
    assert ok

def test_edge_pair():
    """Detect left and right edges as a pair with a horizontal-scanning ROI."""
    print("\n=== EDGE PAIR ===")
    ok = True
    vimg = VisionImage(make_img())

    # ROI centered on (200,200), aw=40, ah=200, theta=90.
    # v=(-1,0), scan left. x=300 -> x=100.
    # Right edge at x=250 (D->L, first), left edge at x=150 (L->D, second).
    roi = [180, 100, 40, 200, 90]
    r = run_tool(vimg, roi, mode="Edge Pair",
                 e1_pol="Dark\u2192Light", e2_pol="Light\u2192Dark",
                 min_w=80, max_w=120)
    if r is None:
        print("  FAIL pair: NO EDGE PAIR")
        ok = False
    else:
        dx1 = abs(r.x1 - 250)
        dx2 = abs(r.x2 - 150)
        w_ok = 90 <= r.width <= 110
        if dx1 <= 2 and dx2 <= 2 and w_ok:
            print(f"  PASS pair: ({r.x1:.1f},{r.y1:.1f}) -> ({r.x2:.1f},{r.y2:.1f}) "
                  f"w={r.width:.1f}")
        else:
            print(f"  FAIL pair: ({r.x1:.1f},{r.y1:.1f}) -> ({r.x2:.1f},{r.y2:.1f}) "
                  f"w={r.width:.1f}  expected x1~250 x2~150 w~100")
            ok = False
    assert ok

def test_polarity():
    """Verify polarity filtering works correctly."""
    print("\n=== POLARITY ===")
    ok = True
    vimg = VisionImage(make_img())
    D2L = "Dark\u2192Light"
    L2D = "Light\u2192Dark"

    # Top edge at theta=0, scan down. D->L should find it.
    r = run_tool(vimg, [170, 130, 60, 40, 0], D2L)
    if r is None:
        print("  FAIL top 0deg D->L: NO EDGE")
        ok = False
    else:
        print(f"  PASS top 0deg D->L: ({r.x:.1f},{r.y:.1f})")

    # Same ROI, L->D should NOT find it.
    r = run_tool(vimg, [170, 130, 60, 40, 0], L2D)
    if r is not None:
        print(f"  FAIL top 0deg L->D: unexpected edge at ({r.x:.1f},{r.y:.1f})")
        ok = False
    else:
        print("  PASS top 0deg L->D: NO EDGE (correct)")

    # Top edge at theta=180, scan up. L->D should find it.
    r = run_tool(vimg, [170, 130, 60, 40, 180], L2D)
    if r is None:
        print("  FAIL top 180deg L->D: NO EDGE")
        ok = False
    else:
        print(f"  PASS top 180deg L->D: ({r.x:.1f},{r.y:.1f})")

    assert ok

def test_no_edge():
    """ROI entirely inside or outside should find nothing."""
    print("\n=== NO EDGE ===")
    ok = True
    vimg = VisionImage(make_img())

    for a in [0, 90, 180, 270]:
        r = run_tool(vimg, [180, 180, 40, 40, a])
        if r is not None:
            print(f"  FAIL inside angle={a:3d}: unexpected edge at ({r.x:.1f},{r.y:.1f})")
            ok = False
        else:
            print(f"  PASS inside angle={a:3d}: NO EDGE")

    for a in [0, 90]:
        r = run_tool(vimg, [50, 50, 40, 40, a])
        if r is not None:
            print(f"  FAIL outside angle={a:3d}: unexpected edge at ({r.x:.1f},{r.y:.1f})")
            ok = False
        else:
            print(f"  PASS outside angle={a:3d}: NO EDGE")
    assert ok

def test_all_angles():
    """Same edge detected at all cardinal angles with correct polarity."""
    print("\n=== ALL ANGLES (same physical edge) ===")
    ok = True
    vimg = VisionImage(make_img())

    # Top edge y=150. 0deg scan down (D->L), 180deg scan up (L->D).
    ok &= check_edge("Top", vimg, [170, 130, 60, 40], 200, 150,
                     [0, 180], ["Dark\u2192Light", "Light\u2192Dark"])

    # Left edge x=150. 90deg scan left (L->D), 270deg scan right (D->L).
    ok &= check_edge("Left", vimg, [130, 170, 40, 60], 150, 200,
                     [90, 270], ["Light\u2192Dark", "Dark\u2192Light"])
    assert ok

def main():
    tests = [
        ("Horizontal edges", test_horizontal_edges),
        ("Vertical edges", test_vertical_edges),
        ("Edge pair", test_edge_pair),
        ("Polarity", test_polarity),
        ("No edge", test_no_edge),
        ("All angles", test_all_angles),
    ]

    print("\n" + "=" * 50)
    all_ok = True
    for name, test in tests:
        try:
            test()
            status = "PASS"
        except AssertionError:
            status = "FAIL"
            all_ok = False
        print(f"  {status}: {name}")

    if all_ok:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
