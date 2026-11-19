"""Application-level help pages (guides and reference).

Tool documentation lives in :mod:`falcocv.ui.tool_help`; this module adds the
pages a user needs around the tools: install, first workflow, files, CLI, HMI,
web, communications, custom scripts, 3D and troubleshooting.

Each entry uses ``kind="guide"`` and a list of ``sections`` instead of the
input/output/parameter tables.  ``tool_help`` merges this mapping into
``TOOL_HELP`` so the dialog and the tests keep a single lookup.
"""

APP_HELP: dict[str, dict] = {}


def _page(key, category, purpose, sections, see_also=None):
    return {
        "kind": "guide",
        "name": key,
        "category": category,
        "purpose": purpose,
        "sections": sections,
        "see_also": see_also or [],
    }


# ── Getting Started ────────────────────────────────────────────────────────

APP_HELP["Getting Started"] = _page(
    "Getting Started",
    "Getting Started",
    "FalcoCV is an industrial machine-vision environment: build an ordered "
    "vision pipeline, run it on images or live cameras, add a web HMI and "
    "connect PLCs — without writing code for the standard tools.",
    [
        {
            "title": "Ways to run FalcoCV",
            "html": (
                "<ul>"
                "<li><b>Standalone bundle:</b> install <code>FalcoCV-x.y.z-windows-x64-setup.exe</code> "
                "or run the portable folder. Python and every dependency are included.</li>"
                "<li><b>Python package:</b> <code>pip install falcocv</code> in a Python 3.13 "
                "environment, then run <code>falcocv</code>.</li>"
                "</ul>"
            ),
        },
        {
            "title": "First launch",
            "html": (
                "<p>The splash screen loads the tools and checks the ONNX models. In the "
                "standalone bundle the models ship inside the installation; with the pip "
                "package they are downloaded once to the user data folder.</p>"
                "<p>The main window has four areas: the <b>toolbox</b> (left), the "
                "<b>workflow tree</b>, the <b>image viewer</b> with overlays and the "
                "<b>job/config panel</b>. The web server starts automatically on port "
                "<code>8000</code>.</p>"
            ),
        },
        {
            "title": "Where to go next",
            "html": (
                "<p>Follow <b>Your First Workflow</b> to build and run a pipeline in a few "
                "minutes. <b>Files, Folders & Logs</b> explains where the configuration, "
                "models and logs live. <b>Command Line</b> covers headless execution and "
                "the packaging tools.</p>"
            ),
        },
    ],
    ["Your First Workflow", "Vision Editor", "Files, Folders & Logs", "Command Line"],
)

APP_HELP["Your First Workflow"] = _page(
    "Your First Workflow",
    "Getting Started",
    "A five-minute pipeline: load an image, filter it, find blobs and read the "
    "results.",
    [
        {
            "title": "1. Add the tools",
            "html": (
                "<p>Drag tools from the toolbox into the workflow, or double-click them. "
                "The list order is the execution order, so add them from left to right:</p>"
                "<ul><li><b>Image Loader</b></li><li><b>Filters</b></li>"
                "<li><b>Blob Detector</b></li></ul>"
            ),
        },
        {
            "title": "2. Configure them",
            "html": (
                "<p>Double-click a tool to open its configuration. In <b>Image Loader</b> "
                "choose a file or folder. In <b>Filters</b> select a filter such as "
                "<i>Gaussian blur</i>. In <b>Blob Detector</b> set the threshold polarity, "
                "area range and, if needed, draw a search ROI on the image.</p>"
            ),
        },
        {
            "title": "3. Connect the ports",
            "html": (
                "<p>Connect <code>output image</code> of one tool to <code>input image</code> "
                "of the next. Ports are typed: image ports only accept images and plane "
                "ports only accept a reference plane.</p>"
            ),
        },
        {
            "title": "4. Run and inspect",
            "html": (
                "<p>Press <b>Run</b> in the tool configuration or the workflow toolbar. "
                "Results and overlays appear on the viewer; the results table lists one row "
                "per detected feature. <b>Auto</b> re-runs a tool 250 ms after you change a "
                "parameter.</p>"
            ),
        },
        {
            "title": "5. Save the job",
            "html": (
                "<p><b>File → Save Job As</b> writes a <code>.cvj</code> file. It contains "
                "the workflow, the tool parameters and any embedded assets, so it can be "
                "reopened on another machine or executed headlessly with "
                "<code>falcocv run</code>.</p>"
            ),
        },
    ],
    ["Vision Editor", "Running Jobs", "Image Loader", "Blob Detector"],
)

APP_HELP["Files, Folders & Logs"] = _page(
    "Files, Folders & Logs",
    "Getting Started",
    "Where FalcoCV keeps its configuration, models, native libraries, user "
    "packages and logs.",
    [
        {
            "title": "User data folder",
            "html": (
                "<p>Windows: <code>%APPDATA%\\FalcoCV</code><br>"
                "Linux: <code>$XDG_DATA_HOME/FalcoCV</code> (usually "
                "<code>~/.local/share/FalcoCV</code>)</p>"
                "<ul>"
                "<li><code>config.json</code> — UI settings, recent files, web port, theme.</li>"
                "<li><code>falcocv.log</code> — startup log and crash stack traces.</li>"
                "<li><code>models/</code> — ONNX models and the OCR dictionary (pip installs "
                "download them here; the standalone bundle uses its own <code>models/</code>).</li>"
                "<li><code>user-packages/</code> — Python packages installed for user scripts "
                "(standalone bundle).</li>"
                "<li><code>native/&lt;driver&gt;/</code> — vendor DLLs such as the ADLINK DASK "
                "libraries or the Beckhoff <code>TcAdsDll.dll</code>.</li>"
                "</ul>"
            ),
        },
        {
            "title": "Jobs and HMI projects",
            "html": (
                "<p><code>.cvj</code> is a ZIP with <code>workflow.json</code> and optional "
                "assets. <code>.hmi</code> is a JSON project with slots, widgets, layout, "
                "script and communications.</p>"
            ),
        },
        {
            "title": "Overriding the folders",
            "html": (
                "<p><code>FALCOCV_CONFIG_DIR</code> moves <code>config.json</code> (useful for "
                "portable installs and tests). <code>FALCOCV_MODELS_DIR</code> points at an "
                "existing model folder. <code>FALCOCV_USER_LIBS</code> adds user library "
                "directories.</p>"
            ),
        },
        {
            "title": "Logs",
            "html": (
                "<p>Every launch appends to <code>falcocv.log</code>. When the application "
                "closes unexpectedly, the file ends with a <i>Windows fatal exception</i> "
                "block that names the crashing thread and module — include that block in a "
                "bug report.</p>"
            ),
        },
    ],
    ["Troubleshooting", "User Libraries", "Command Line"],
)

APP_HELP["Troubleshooting"] = _page(
    "Troubleshooting",
    "Getting Started",
    "What to check when the application does not start, a tool fails or the web "
    "HMI does not load.",
    [
        {
            "title": "The app closes or does not start",
            "html": (
                "<p>Open <code>%APPDATA%\\FalcoCV\\falcocv.log</code> and read the last "
                "block. A <code>Windows fatal exception</code> with a module name is a native "
                "crash; send that block with the version shown in <b>Help → About</b>.</p>"
            ),
        },
        {
            "title": "Models are missing",
            "html": (
                "<p>With the pip package the models download on first run. If the machine is "
                "offline, copy a <code>models</code> folder and set "
                "<code>FALCOCV_MODELS_DIR</code> or run "
                "<code>falcocv download-models</code>. The standalone bundle already includes "
                "them.</p>"
            ),
        },
        {
            "title": "A PLC or I/O card is not detected",
            "html": (
                "<p>Run <code>falcocv native-deps</code>. It reports the resolved DLL paths. "
                "Vendor libraries are not bundled: place them in "
                "<code>%APPDATA%\\FalcoCV\\native\\&lt;driver&gt;</code>, set "
                "<code>FALCOCV_NATIVE_DIR</code>, or install the vendor software (TwinCAT for "
                "Beckhoff ADS).</p>"
            ),
        },
        {
            "title": "Web dashboard or HMI does not open",
            "html": (
                "<p>Check the port in <b>Tools → Start Web Server</b> (default "
                "<code>8000</code>) and make sure no other process is using it. The dashboard "
                "is <code>http://localhost:8000</code>; an HMI is "
                "<code>http://localhost:8000/?id=&lt;project_id&gt;</code>.</p>"
            ),
        },
        {
            "title": "Windows warns about the installer",
            "html": (
                "<p>The standalone installers are not code-signed yet, so SmartScreen shows "
                "<i>Windows protected your PC</i>. Use <b>More info → Run anyway</b> or run "
                "the portable folder.</p>"
            ),
        },
        {
            "title": "User packages are not imported",
            "html": (
                "<p>Run <code>falcocv packages list</code> and then "
                "<code>falcocv python -c \"import mypackage\"</code> to see the real error. "
                "Only binary wheels for CPython 3.13 are supported, and the packages bundled "
                "with FalcoCV (numpy, OpenCV, Qt) always win.</p>"
            ),
        },
        {
            "title": "Strict signal checks",
            "html": (
                "<p>Set <code>FALCOCV_STRICT_SIGNALS=1</code> before launching to turn "
                "cross-thread widget warnings into errors. Useful when developing custom "
                "tools.</p>"
            ),
        },
    ],
    ["Files, Folders & Logs", "Command Line", "User Libraries"],
)


# ── Guides ─────────────────────────────────────────────────────────────────

APP_HELP["Vision Editor"] = _page(
    "Vision Editor",
    "Guides",
    "How the workflow engine, ports, fixtures, overlays and jobs work.",
    [
        {
            "title": "An ordered pipeline",
            "html": (
                "<p>The tool list <b>is</b> the execution contract: tools run top to bottom "
                "and connections only feed inputs. There is no automatic sorting and no cycle "
                "detection, so place every tool before the ones that consume it.</p>"
            ),
        },
        {
            "title": "Ports and kinds",
            "html": (
                "<p>Ports are typed: <b>Image</b>, <b>Plane</b>, <b>Results</b> or "
                "<b>Any</b>. A typed input only accepts a matching output, and an untyped "
                "output only feeds untyped inputs. The connection menu only offers earlier "
                "tools.</p>"
            ),
        },
        {
            "title": "ROI and fixtures",
            "html": (
                "<p>A ROI is stored relative to the image fixture, so it follows the part "
                "when a matcher or barcode sets a new fixture. Tools that locate a part "
                "(<b>Template Match</b>, <b>Shape Match</b>, <b>Barcode</b>) can act as a "
                "fixture for the downstream tools.</p>"
            ),
        },
        {
            "title": "Overlays",
            "html": (
                "<p>ROIs, contours, lines, circles, points and labels are vector overlays, "
                "never pixels. Each element carries its source tool, so inherited overlays are "
                "cleaned instead of duplicated. <b>Copy overlays to input</b> makes them "
                "visible to downstream tools.</p>"
            ),
        },
        {
            "title": "Run, Auto and background execution",
            "html": (
                "<p><b>Run</b> executes the workflow on a background thread; structural edits "
                "are blocked while it runs and parameter edits stay enabled. <b>Auto</b> "
                "re-runs the same tool 250 ms after the last change and never reruns upstream "
                "or downstream tools.</p>"
            ),
        },
        {
            "title": "Blocks",
            "html": (
                "<p>A <b>Block</b> is a nested workflow. Select several tools and use "
                "<b>Generate Block</b> to wrap them; the block exposes its parent inputs and "
                "outputs. See the <b>Block</b> tool page.</p>"
            ),
        },
    ],
    ["Your First Workflow", "Running Jobs", "Block", "Template Match", "Shape Match"],
)

APP_HELP["Running Jobs"] = _page(
    "Running Jobs",
    "Guides",
    "Run a saved job from the GUI or headlessly and read the JSON report.",
    [
        {
            "title": "Headless execution",
            "html": (
                "<pre>falcocv run job.cvj\n"
                "falcocv run job.cvj -o results.json\n"
                "falcocv run job.cvj --trace</pre>"
                "<p>Exit code <code>0</code> means the job completed; any tool error makes it "
                "non-zero. <code>--trace</code> prints per-tool timing and errors to stderr.</p>"
            ),
        },
        {
            "title": "JSON report",
            "html": (
                "<p>The report has <code>job</code>, <code>ok</code>, "
                "<code>total_time_ms</code>, a <code>tools</code> list with "
                "<code>name</code>/<code>time_ms</code>/<code>error</code> and the "
                "<code>results</code> produced by the tools.</p>"
            ),
        },
        {
            "title": "Error policy",
            "html": (
                "<p><code>stop_on_error</code> stops the workflow at the first failure. Each "
                "tool also has <code>min_results</code>: when it produces fewer results, the "
                "tool fails instead of silently returning an empty list.</p>"
            ),
        },
        {
            "title": "From the GUI",
            "html": (
                "<p>Open the job with <b>File → Open</b>, press <b>Run</b> and inspect the "
                "overlays and the results table. Recent jobs are kept in the File menu.</p>"
            ),
        },
    ],
    ["Command Line", "Vision Editor", "Getting Started"],
)

APP_HELP["HMI Designer"] = _page(
    "HMI Designer",
    "Guides",
    "Build an operator dashboard: slots that run jobs, widgets bound to results "
    "and a script that drives the machine.",
    [
        {
            "title": "Project, slots and widgets",
            "html": (
                "<p>A <code>.hmi</code> project has a grid layout, widgets and one or more "
                "<b>slots</b>. Each slot points at a <code>.cvj</code> job; running a slot "
                "executes that job and updates the widgets.</p>"
            ),
        },
        {
            "title": "The HMI script",
            "html": (
                "<p>The script runs in its own thread and receives <code>run_slot</code>, "
                "<code>log</code>, <code>set_slot_status</code>, <code>sleep</code>, "
                "<code>context</code>, <code>comms</code> and <code>proj</code>. Use "
                "<code>_stop</code> to leave loops cleanly. The full API is in the "
                "<b>HMI Project</b> and <b>HMI Scripting</b> pages.</p>"
            ),
        },
        {
            "title": "Multi-file scripts",
            "html": (
                "<p>Instead of a single script you can point the project at a folder. Its "
                "files are copied to a temporary directory, added to <code>sys.path</code> and "
                "the entry file is executed, so helper modules import normally.</p>"
            ),
        },
        {
            "title": "Preview, deploy and workers",
            "html": (
                "<p><b>Preview</b> renders the layout with sample values. <b>Deploy</b> sends "
                "the project to the web server; the server runs it in an isolated worker "
                "process that can be stopped, reloaded and debugged from the designer.</p>"
            ),
        },
        {
            "title": "Communications",
            "html": (
                "<p>Devices and tags are configured in the Communications tab. Widgets can "
                "read a tag directly, and the script can call "
                "<code>comms.read()</code>/<code>comms.write()</code>. See <b>Communications</b>.</p>"
            ),
        },
    ],
    ["HMI Project", "HMI Scripting", "Communications", "Web Dashboard & HMI"],
)

APP_HELP["Web Dashboard & HMI"] = _page(
    "Web Dashboard & HMI",
    "Guides",
    "The built-in web server: dashboard, operator HMI, workers and API.",
    [
        {
            "title": "Starting the server",
            "html": (
                "<p>The GUI starts the server automatically (the status bar shows "
                "<code>Web: ●</code>). Use <b>Tools → Start/Stop Web Server</b> to control it, "
                "or run <code>falcocv-server --host 0.0.0.0 --port 8000</code>.</p>"
            ),
        },
        {
            "title": "URLs",
            "html": (
                "<ul>"
                "<li>Dashboard: <code>http://localhost:8000</code></li>"
                "<li>Operator HMI: <code>http://localhost:8000/?id=&lt;project_id&gt;</code></li>"
                "<li>Health probe: <code>http://localhost:8000/api/health</code></li>"
                "</ul>"
                "<p>The port comes from <code>web_port</code> in <code>config.json</code>.</p>"
            ),
        },
        {
            "title": "Workers",
            "html": (
                "<p>Each deployed HMI runs in a separate worker process "
                "(<code>FalcoCVServer --worker</code>). That isolates crashes and lets the "
                "designer stop or reload a project without touching the server.</p>"
            ),
        },
        {
            "title": "Images and overlays",
            "html": (
                "<p>Frames are served as JPEG/PNG and overlays as vector JSON, so the HMI "
                "stays sharp at any resolution and the browser never receives rasterized "
                "graphics.</p>"
            ),
        },
    ],
    ["HMI Designer", "Communications", "Troubleshooting"],
)

APP_HELP["Communications"] = _page(
    "Communications",
    "Guides",
    "Connect PLCs, I/O cards and OPC UA/Modbus devices, then share their tags "
    "with the HMI.",
    [
        {
            "title": "Devices, tags and quality",
            "html": (
                "<p>The runtime lives in <code>src/falcocv/comms</code>. A device uses a "
                "driver and exposes tags with a type, address and poll interval. Every value "
                "carries a quality flag (<code>good</code>, <code>stale</code>, "
                "<code>bad</code>, <code>disabled</code>).</p>"
            ),
        },
        {
            "title": "Drivers",
            "html": (
                "<ul>"
                "<li><b>ADLINK DASK</b> — digital/analog I/O on Windows x64.</li>"
                "<li><b>Beckhoff ADS</b> — symbolic PLC variables.</li>"
                "<li><b>Siemens S7</b> — absolute DB/area addresses.</li>"
                "<li><b>Omron</b> — fast struct reads, validated on hardware.</li>"
                "<li><b>Modbus TCP</b> and <b>OPC UA</b> — clients and servers.</li>"
                "</ul>"
                "<p>The <code>plc</code> and <code>protocols</code> extras install the "
                "optional libraries. Each driver page documents its address format.</p>"
            ),
        },
        {
            "title": "Signal Monitor",
            "html": (
                "<p>The designer's <b>Signal Monitor</b> connects to the devices and shows live "
                "values. <b>Live worker</b> mode reads and writes the deployed worker through "
                "the HTTP API, so you see exactly what the operator sees.</p>"
            ),
        },
        {
            "title": "Write verification",
            "html": (
                "<p>PLC writes can verify the read-back with a configurable delay and retry "
                "count. It is opt-in per device because it doubles the traffic and depends on "
                "the relay settling time.</p>"
            ),
        },
        {
            "title": "Servers",
            "html": (
                "<p>FalcoCV can also expose its tags as a Modbus TCP or OPC UA server. Mappings "
                "are explicit and live in the <code>.hmi</code> <code>servers</code> section.</p>"
            ),
        },
    ],
    ["HMI Communications", "Modbus TCP", "OPC UA", "ADLINK driver", "Beckhoff driver"],
)

APP_HELP["Custom Tools & Debugger"] = _page(
    "Custom Tools & Debugger",
    "Guides",
    "Write a Python tool, import your own modules and debug it step by step.",
    [
        {
            "title": "The CustomTool contract",
            "html": (
                "<p>A script defines <code>class CustomTool</code> with "
                "<code>__init__(self, engine_context)</code> and "
                "<code>process(self, inputs)</code>. <code>inputs</code> maps port names to "
                "values (image ports carry a <code>VisionImage</code>); return a dict with the "
                "output port values. Configure the ports in the tool's Ports table.</p>"
            ),
        },
        {
            "title": "Script code or script file",
            "html": (
                "<p>Paste the code in the editor, or point <code>script_path</code> at a "
                "<code>.py</code> file. With a file, its folder is added to "
                "<code>sys.path</code>, so <code>helpers.py</code> next to it imports "
                "normally.</p>"
            ),
        },
        {
            "title": "Engine context",
            "html": (
                "<p><code>engine_context</code> is a dict shared with the HMI runtime. The HMI "
                "can expose <code>context['log']</code>, a connected PLC or any other live "
                "object; the tool receives it on construction and it is never serialized.</p>"
            ),
        },
        {
            "title": "Debugging",
            "html": (
                "<p>Click the gutter to add a breakpoint, then press <b>Debug</b>. The toolbar "
                "offers <b>Continue</b> (F5), <b>Step over</b> (F10), <b>Step into</b> (F11) and "
                "<b>Step out</b> (Shift+F11), with locals and a watch panel.</p>"
                "<p>The debugger runs on the workflow thread and the pause is delivered to the "
                "GUI thread, so the editor keeps responding while the script is stopped.</p>"
            ),
        },
        {
            "title": "Using your own libraries",
            "html": (
                "<p>See <b>User Libraries</b>. The standalone bundle appends the user folders "
                "to <code>sys.path</code>; the packages bundled with FalcoCV keep priority.</p>"
            ),
        },
    ],
    ["Python Script", "User Libraries", "HMI Scripting"],
)

APP_HELP["3D & Depth"] = _page(
    "3D & Depth",
    "Guides",
    "Depth images, point clouds, planes, height measurements and the 3D viewer.",
    [
        {
            "title": "Depth sources",
            "html": (
                "<p><b>Depth/Range Loader</b> reads TIFF, PNG or the FalcoCV <code>.fcd</code> "
                "container. FCD v3 stores the calibrated raw uint16 counts plus the header "
                "calibration and decodes much faster than LZW TIFF. Integer inputs are scaled "
                "with <code>scale_z</code>/<code>offset_z</code>.</p>"
            ),
        },
        {
            "title": "Coordinates",
            "html": (
                "<p><code>z = depth * z_res + z_offset</code>; X is mirrored so image-left maps "
                "to world +X. Planes point to +Z/up, so <code>plane.height(point) &gt; 0</code> "
                "means above the plane. ROIs are stored in fixture coordinates.</p>"
            ),
        },
        {
            "title": "Point clouds",
            "html": (
                "<p>The native kernel builds the cloud with a stride derived from the target "
                "point count. Colors are generated lazily, only when a 3D viewer uploads the "
                "cloud, which keeps headless jobs fast.</p>"
            ),
        },
        {
            "title": "Planes and heights",
            "html": (
                "<p><b>Plane Fitter</b> publishes a <code>PlaneModel</code> that other tools "
                "consume through the <code>reference plane</code> port. <b>Height Blobs 3D</b> "
                "measures the regions above or below that plane in world millimetres, with "
                "grid ordering for matrices of parts.</p>"
            ),
        },
        {
            "title": "3D viewer",
            "html": (
                "<p>Rotate with the left button, pan with the middle button and zoom with the "
                "wheel. 3D ROIs are drawn with large handles and no depth test so they remain "
                "usable inside dense clouds; <b>Shift + centre handle</b> moves the Z band.</p>"
            ),
        },
    ],
    ["Depth/Range Loader", "Plane Fitter", "Height Blobs 3D"],
)

APP_HELP["User Libraries"] = _page(
    "User Libraries",
    "Guides",
    "Install third-party packages for pipeline and HMI scripts in the "
    "standalone bundle.",
    [
        {
            "title": "Where packages are looked up",
            "html": (
                "<p>FalcoCV appends these folders to <code>sys.path</code>, in order:</p>"
                "<ol>"
                "<li><code>FALCOCV_USER_LIBS</code> (a <code>;</code>-separated list)</li>"
                "<li><code>&lt;install&gt;\\site-packages</code> — libraries shipped by the "
                "integrator</li>"
                "<li><code>%APPDATA%\\FalcoCV\\user-packages</code> — managed by "
                "<code>falcocv packages</code></li>"
                "</ol>"
                "<p>They are <b>appended</b>, so the numpy, OpenCV, Qt and ONNX Runtime that "
                "ship with FalcoCV always keep priority.</p>"
            ),
        },
        {
            "title": "Installing packages",
            "html": (
                "<pre>falcocv packages install pandas requests\n"
                "falcocv packages list\n"
                "falcocv packages remove pandas\n"
                "falcocv packages dir</pre>"
                "<p>Installation uses the bundled <code>uv</code> and only accepts binary "
                "wheels for CPython 3.13.</p>"
            ),
        },
        {
            "title": "Offline machines",
            "html": (
                "<p>Point at a local wheelhouse with "
                "<code>falcocv packages install --find-links D:\\wheels openpyxl</code> or set "
                "<code>FALCOCV_WHEELHOUSE</code>.</p>"
            ),
        },
        {
            "title": "Verifying an import",
            "html": (
                "<p><code>falcocv python -c \"import pandas; print(pandas.__version__)\"</code> "
                "runs the code inside the frozen interpreter, exactly as a pipeline or HMI "
                "script would.</p>"
            ),
        },
        {
            "title": "Limitations",
            "html": (
                "<p>A package that requires a different numpy/OpenCV version will import the "
                "bundled one; packages that need a C compiler cannot be installed. If a "
                "dependency is critical, build a custom bundle with it included.</p>"
            ),
        },
    ],
    ["Troubleshooting", "Custom Tools & Debugger", "Command Line"],
)


# ── Reference ──────────────────────────────────────────────────────────────

APP_HELP["Command Line"] = _page(
    "Command Line",
    "Reference",
    "Every FalcoCV executable and its options.",
    [
        {
            "title": "Entry points",
            "html": (
                "<ul>"
                "<li><code>falcocv</code> — GUI when called without arguments; also the "
                "command dispatcher below.</li>"
                "<li><code>falcocv-server</code> — web/HMI server "
                "(<code>--host</code>, <code>--port</code>, <code>--reload</code>).</li>"
                "<li><code>falcocv-hmi [project.hmi]</code> — HMI Designer.</li>"
                "<li><code>falcocv-download-models</code> — fetch the ONNX models.</li>"
                "</ul>"
                "<p>The standalone bundle ships the same commands as "
                "<code>FalcoCV.exe</code>, <code>FalcoCVServer.exe</code>, "
                "<code>FalcoCVCLI.exe</code> and <code>FalcoCVHMI.exe</code>.</p>"
            ),
        },
        {
            "title": "falcocv run",
            "html": (
                "<pre>falcocv run job.cvj\n"
                "falcocv run job.cvj -o results.json\n"
                "falcocv run job.cvj --trace</pre>"
                "<p>Runs the job in the stored order and prints a JSON report. "
                "<code>--trace</code> writes per-tool timing and errors to stderr.</p>"
            ),
        },
        {
            "title": "falcocv self-test",
            "html": (
                "<p>Checks the installation without opening a window: imports, every tool "
                "config, help data, a synthetic pipeline, a job round-trip, the native kernel "
                "and the models. It is the first thing to run on a new machine.</p>"
            ),
        },
        {
            "title": "falcocv native-deps",
            "html": (
                "<p>Reports where the ADLINK DASK and Beckhoff ADS DLLs were found and whether "
                "the optional PLC/Modbus/OPC UA libraries are installed.</p>"
            ),
        },
        {
            "title": "falcocv packages",
            "html": (
                "<pre>falcocv packages install NAME [NAME ...] [--find-links DIR]\n"
                "falcocv packages list\n"
                "falcocv packages remove NAME [NAME ...]\n"
                "falcocv packages dir</pre>"
                "<p>Manages the user library folder. See <b>User Libraries</b>.</p>"
            ),
        },
        {
            "title": "falcocv python",
            "html": (
                "<pre>falcocv python -c \"import sys; print(sys.version)\"\n"
                "falcocv python script.py</pre>"
                "<p>Runs code with the application interpreter and user libraries loaded. "
                "Used to diagnose imports; a script file's folder is added to "
                "<code>sys.path</code>.</p>"
            ),
        },
        {
            "title": "Exit codes",
            "html": (
                "<p><code>0</code> success, non-zero on failure. <code>falcocv run</code> "
                "returns non-zero when a tool fails or the workflow stops on an error.</p>"
            ),
        },
    ],
    ["Running Jobs", "User Libraries", "Troubleshooting"],
)
