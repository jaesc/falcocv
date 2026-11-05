"""End-to-end worker test for the dummy communications runtime."""

from __future__ import annotations


def test_worker_comms_snapshot_returns_definitions_and_values():
    import falcocv.web.hmi_worker as worker
    from falcocv.core.hmi import HMIProject

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "dummy",
                "driver": "dummy",
                "tags": [
                    {
                        "name": "count",
                        "type": "int",
                        "access": "rw",
                        "default": 0,
                    }
                ],
            }
        ],
    }
    project.start_runtime()
    old_proj = worker._proj
    worker._proj = project
    try:
        data = worker._comms_snapshot()
        assert data["tags"][0]["id"] == "dummy.count"
        assert data["values"]["dummy.count"]["value"] == 0
        assert data["statuses"][0]["connected"] is True

        values = worker._comms_values()
        assert values["values"]["dummy.count"]["value"] == 0
        assert values["statuses"][0]["connected"] is True
    finally:
        worker._proj = old_proj
        project.stop_runtime()


def test_hmi_worker_starts_comms_before_script_and_stops_after():
    import falcocv.web.hmi_worker as worker

    worker._proj = None
    worker._finished.clear()
    project = {
        "script": (
            "comms.write('dummy.count', 11)\n"
            "context['captured'] = comms.read('dummy.count')\n"
        ),
        "layout": {"rows": 10, "cols": 10, "widgets": []},
        "slots": [],
        "comms": {
            "version": 1,
            "devices": [
                {
                    "id": "dummy",
                    "driver": "dummy",
                    "tags": [
                        {
                            "name": "count",
                            "type": "int",
                            "access": "rw",
                            "default": 0,
                        }
                    ],
                }
            ],
        },
    }

    try:
        worker.handle_cmd({"cmd": "start", "project": project, "bps": {}})
        assert worker._finished.wait(3.0)
        assert worker._proj is not None
        assert worker._proj.context.get("captured") == 11
        assert worker._proj.comms is None
    finally:
        worker._proj = None
        worker._finished.clear()
