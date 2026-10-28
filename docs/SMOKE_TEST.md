# Manual smoke test (release only)

Most regressions are covered by the automated suite:

```bash
pytest -q --ignore=tests/test_trainer_equivalence.py
falcocv self-test
python scripts/check_wheel.py
python scripts/wheel_smoke.py
```

Only the items below require real hardware or human perception. Run them
before publishing a release.

## 1. GigE / GenTL camera

- [ ] Connect a real camera using the vendor `.cti` producer.
- [ ] Discover and connect.
- [ ] Snapshot and live preview.
- [ ] Software trigger and hardware trigger (if the camera supports it).
- [ ] Disconnect/reconnect without leaving a stale process.

## 2. Depth / 3D

- [ ] Load a real depth image.
- [ ] Generate the point cloud.
- [ ] Open the 3D viewer: rotate, zoom, fit.
- [ ] Draw a 3D ROI and run Plane Fitter.
- [ ] Confirm the plane overlay and Z range look correct.

## 3. HMI

- [ ] Deploy a project with the HMI designer.
- [ ] Open `http://<host>:8000/?id=<project_id>` on the target tablet/PC.
- [ ] Verify live image updates, pass/fail widgets and reconnect after a
      server restart.
- [ ] Check the layout at the target resolution and DPI.

## 4. Visual / desktop

- [ ] Run on the target monitor and scaling (100 %, 125 %, 150 %).
- [ ] Switch dark/light theme and confirm the splash and icons.
- [ ] Open a tool configuration, dock/undock the toolbox, save and reload a
      job.
- [ ] Confirm the 3D viewer uses the expected GPU/OpenGL driver.

## 5. Deployment

- [ ] Install the wheel on a clean machine **without** the source tree.
- [ ] Confirm the model release downloads (or use a mirror via
      `FALCOCV_MODELS_URL`).
- [ ] Start `falcocv`, `falcocv-server` and `falcocv-hmi`.
