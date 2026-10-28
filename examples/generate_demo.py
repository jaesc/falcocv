"""Generate demo HMI project with two minimal job files and a cycling script."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falcocv.core.workflow import Workflow
from falcocv.tools.block.logic import Block
from falcocv.core.job_manager import JobManager
from falcocv.core.hmi import HMIProject


def create_minimal_job(path: str):
    wf = Workflow()
    block = Block(name="DemoTool")
    wf.add_tool(block)
    wf.stop_on_error = True
    jm = JobManager()
    jm.save_job(wf, path)
    return path


HMI_SCRIPT = """\
# Demo script: alternates between job1 and job2 with logs and states
log("HMI demo started")
while True:
    log("Running job1...")
    run_slot("job1")
    set_slot_status("job1", "pass")
    log("job1 finished: PASS")
    sleep(1)

    log("Running job2...")
    run_slot("job2")
    set_slot_status("job2", "fail")
    log("job2 finished: FAIL")
    sleep(1)
"""


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    j1 = create_minimal_job(os.path.join(base, "demo_job1.cvj"))
    j2 = create_minimal_job(os.path.join(base, "demo_job2.cvj"))

    proj = HMIProject()
    proj.add_slot(name="job1", job_path=j1)
    proj.add_slot(name="job2", job_path=j2)
    w1 = proj.add_widget("StatusIndicator", x=2, y=2)
    w1["slot"] = "job1"
    w1["label"] = "Job 1"
    w1["w"] = 10
    w1["h"] = 8
    w2 = proj.add_widget("StatusIndicator", x=14, y=2)
    w2["slot"] = "job2"
    w2["label"] = "Job 2"
    w2["w"] = 10
    w2["h"] = 8
    w3 = proj.add_widget("LogPanel", x=2, y=12)
    w3["w"] = 22
    w3["h"] = 20
    proj.script = HMI_SCRIPT

    hmi_path = os.path.join(base, "demo.hmi")
    proj.save(hmi_path)

    print(f"Generated: {j1}")
    print(f"Generated: {j2}")
    print(f"Generated: {hmi_path}")
    print()
    print("To run:")
    print(f"  1. Start server:    py -3.13 -m falcocv.web")
    print(f"  2. Deploy project:  py -3.13 -m falcocv.falcocv.web.deploy {hmi_path}")
    print(f"  3. Open browser:    http://localhost:8000")


if __name__ == "__main__":
    main()
