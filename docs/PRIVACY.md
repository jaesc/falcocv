# Privacy policy

FalcoCV does not collect telemetry, analytics or personal data.

It transfers information only to endpoints the user explicitly configures:

- cameras and frame grabbers (GenTL / GigE / USB3 Vision);
- PLCs, fieldbus devices, OPC UA servers and Modbus TCP servers;
- the optional web HMI server, which listens only on the address and port the
  user configures;
- the GitHub Releases URL used to download ONNX models on first use of the
  model-based tools.

Job files, images, logs and configuration stay on the machine where FalcoCV
runs. FalcoCV does not upload them anywhere.

Third-party components bundled with FalcoCV (Qt, OpenCV, ONNX Runtime, NumPy,
FastAPI and the Python standard library) do not send data on their own.

If a job or HMI script written by the user connects to another system, that
connection and its privacy implications are defined by the user's script, not
by FalcoCV.
