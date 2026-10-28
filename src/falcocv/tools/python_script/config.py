import os
import sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QLineEdit, QFileDialog,
                             QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
                             QSplitter)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.Qsci import QsciScintilla
from falcocv.ui.style import INFO, SUCCESS, WARNING, ALT_BG, BORDER, TEXT, TEXT_MUTED, SCRIPT_BG
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.widgets.script_editor import ScriptEditor, FindReplaceBar
from falcocv.ui.widgets.debug_panels import DebugToolbar, VariableInspector, WatchPanel
from falcocv.core.tool_base import BaseTool

_PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

DEFAULT_TEMPLATE = '''# FalcoCV Custom Tool Template
#
# Input & Output ports are defined in the "Terminals" tab.
# All image ports receive / return VisionImage objects (never raw numpy arrays).
#
# Python {} embedded runtime -- modules must match this version.
#
# __init__() runs ONCE when the script is loaded/reloaded.
# process()  runs every time the workflow executes this tool.

import cv2
import numpy as np
from falcocv.core.vision_image import VisionImage


class CustomTool:
    # Called once when the tool is loaded or the script is recompiled.
    def __init__(self, engine_context):
        self.engine = engine_context

    # Called on every workflow execution.
    # Args:
    #     inputs: dict {port_name: value} - image ports always carry VisionImage.
    # Returns:
    #     dict {port_name: value} mapping to the tool output ports.
    def process(self, inputs):
        # Get input image (always VisionImage)
        v_img: VisionImage = inputs.get("image_in")
        if v_img is None:
            return {"image_out": None}

        # Fixture offset of the image in job coordinates
        fx, fy, fangle = v_img.x, v_img.y, v_img.angle

        # Raw numpy pixel buffer (image data only, no fixture transform)
        raw = v_img.data
        h, w = raw.shape[:2]

        # Fixture coordinate conversion
        # ROI defined in fixture-relative coordinates (e.g. from a RoiEditor)
        roi_fixture = [10, 10, 100, 80, 0]   # [x, y, w, h, angle] rel. to fixture

        # Convert to absolute image pixel coordinates
        roi_abs = v_img.fixture_to_image(roi_fixture)
        # roi_abs = [abs_x, abs_y, w, h, abs_angle]  (pixel coords in the image)

        # Convert back: absolute -> fixture-relative
        roi_back = v_img.image_to_fixture(roi_abs)

        # Processing example: convert to grayscale
        if raw.ndim == 3 and raw.shape[2] in (3, 4):
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        elif raw.ndim == 3 and raw.shape[2] == 1:
            gray = raw[:, :, 0]
        else:
            gray = raw.copy()

        # Build output VisionImage (shares pixel data, no copy)
        out = VisionImage(gray, x=fx, y=fy, angle=fangle)
        # Propagate upstream overlays from input (ROIs, contours, etc.)
        out.inherit_overlays_from(v_img)

        # Drawing overlays (absolute image pixel coordinates)
        layer = out.get_overlay_layer("default")

        # ROI from fixture coordinates (converted to absolute)
        layer.add_roi(roi_abs, color=(255, 128, 0), thickness=2)

        # Cross-hair at center (absolute coords)
        layer.add_line((w // 2, 0), (w // 2, h), color=(0, 255, 0), thickness=1)
        layer.add_line((0, h // 2), (w, h // 2), color=(0, 255, 0), thickness=1)

        # Center dot
        layer.add_circle((w // 2, h // 2), radius=5, color=(0, 0, 255), fill=True)

        # Text label (absolute coords)
        layer.add_text(f"({w}x{h})  fixt:{fangle:.1f}deg",
                       position=(10, 20),
                       color=(255, 255, 0), font_scale=0.7, thickness=2)

        # Return results mapped to output ports
        return {
            "image_out": out,
        }
'''.replace("{}", _PY_VER)


# ==============================================================================
# Tool (engine)
# ==============================================================================

from .logic import PythonScriptTool


# ==============================================================================
# UI Configuration
# ==============================================================================

class PythonScriptConfig(QWidget):
    def __init__(self, tool: PythonScriptTool):
        super().__init__()
        self.tool = tool

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        # Tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Tab 1: Script
        self.tabs.addTab(self._build_script_tab(), "Script")

        # Tab 2: Terminals
        self.tabs.addTab(self._build_terminals_tab(), "Terminals")

        # Tab 3: Help
        self.tabs.addTab(self._build_help_tab(), "Help")

        # Footer
        self.footer = QWidget()
        self.footer.setFixedHeight(20)
        self.footer.setStyleSheet(f"background-color: {ALT_BG}; border-top: 1px solid {BORDER};")
        f_lyt = QHBoxLayout(self.footer)
        f_lyt.setContentsMargins(5, 0, 5, 0)
        self.label_time = QLabel("0.0 ms")
        self.label_time.setStyleSheet(f"color: {SUCCESS}; font-weight: bold; font-size: 11px;")
        f_lyt.addWidget(self.label_time)
        f_lyt.addStretch()
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)

        # Debugger connections
        self.tool.debugger_paused.connect(self._on_debugger_paused)
        self.tool.debugger_resumed.connect(self._on_debugger_resumed)

    def on_config_closed(self):
        """Called by ToolConfigDialog.closeEvent when the config window closes.
        Clears breakpoints so debugger doesn't block silently on next run."""
        self.editor.clear_breakpoints()
        self.tool._debug_breakpoints.clear()
        self.tool._debug_editor = None
        sizes = list(self.debug_splitter.sizes()) + list(self.script_splitter.sizes())
        from falcocv.core.config import ConfigManager
        ConfigManager().set_tool_splitter("PythonScriptTool", sizes)

    def _build_script_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # File path bar
        path_bar = QWidget()
        path_lyt = QHBoxLayout(path_bar)
        path_lyt.setContentsMargins(5, 5, 5, 5)
        path_lyt.addWidget(QLabel("File:"))
        self.edit_path = QLineEdit(self.tool.params.get("script_path", ""))
        self.edit_path.textChanged.connect(self._on_path_changed)
        path_lyt.addWidget(self.edit_path, 1)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse)
        btn_browse.setAutoDefault(False)
        path_lyt.addWidget(btn_browse)
        btn_new = QPushButton("New Script")
        btn_new.clicked.connect(self._on_new_script)
        btn_new.setAutoDefault(False)
        path_lyt.addWidget(btn_new)
        layout.addWidget(path_bar)

        # Splitter: editor (top) vs debug panel (bottom)
        self.script_splitter = QSplitter(Qt.Orientation.Vertical)

        # Editor container
        editor_container = QWidget()
        editor_lyt = QVBoxLayout(editor_container)
        editor_lyt.setContentsMargins(0, 0, 0, 0)

        self.editor = ScriptEditor()
        script_code = self.tool.params.get("script_code", "")
        if not script_code:
            script_code = DEFAULT_TEMPLATE
            self.tool.params["script_code"] = script_code
        self.editor.setText(script_code)
        self.editor.textChanged.connect(self._on_code_changed)
        editor_lyt.addWidget(self.editor, 1)

        # Find / Replace bar (anchored to bottom of editor)
        self.find_bar = FindReplaceBar(self.editor)
        self.editor.set_find_bar(self.find_bar)
        editor_lyt.addWidget(self.find_bar)

        self.script_splitter.addWidget(editor_container)

        # Debug panel
        debug_container = QWidget()
        debug_lyt = QVBoxLayout(debug_container)
        debug_lyt.setContentsMargins(0, 0, 0, 0)

        self.debug_toolbar = DebugToolbar()
        self.debug_toolbar.continue_requested.connect(self.tool._debug_continue)
        self.debug_toolbar.step_over_requested.connect(self.tool._debug_step_over)
        self.debug_toolbar.step_into_requested.connect(self.tool._debug_step_into)
        self.debug_toolbar.step_out_requested.connect(self.tool._debug_step_out)
        self.debug_toolbar.stop_requested.connect(self._on_debug_stop)
        debug_lyt.addWidget(self.debug_toolbar)

        # Inject Qt wait strategy so debugger pause doesn't freeze the UI
        from PyQt6.QtWidgets import QApplication

        def _qt_wait(event):
            while not event.wait(0.05):
                QApplication.processEvents()

        self.tool._debugger.set_wait_strategy(_qt_wait)

        # Horizontal splitter: locals (left) | watch (right)
        self.debug_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.var_inspector = VariableInspector()
        self.debug_splitter.addWidget(self.var_inspector)

        self.watch_panel = WatchPanel()
        self.debug_splitter.addWidget(self.watch_panel)

        self.debug_splitter.setSizes([400, 300])
        debug_lyt.addWidget(self.debug_splitter, 1)

        self.script_splitter.addWidget(debug_container)
        self.debug_container = debug_container

        from falcocv.core.config import ConfigManager
        saved = ConfigManager().get_tool_splitter("PythonScriptTool")
        if saved and len(saved) == 4:
            self.debug_splitter.setSizes(saved[:2])
            self.script_splitter.setSizes(saved[2:])
        else:
            self.debug_splitter.setSizes([400, 300])
            self.script_splitter.setSizes([700, 300])
        layout.addWidget(self.script_splitter, 1)

        # Show debug panel when breakpoints exist; hide when none
        self.editor.breakpoints_changed.connect(self._update_debug_visibility)
        QTimer.singleShot(0, self._update_debug_visibility)

        # Actions
        action_bar = QWidget()
        action_lyt = QHBoxLayout(action_bar)
        action_lyt.setContentsMargins(5, 0, 5, 5)

        btn_reload = QPushButton("⟳ Reload / Compile")
        btn_reload.setAutoDefault(False)
        btn_reload.clicked.connect(self._on_reload)
        action_lyt.addWidget(btn_reload)

        btn_save = QPushButton("💾 Save")
        btn_save.setAutoDefault(False)
        btn_save.clicked.connect(self._on_save)
        action_lyt.addWidget(btn_save)

        btn_clear_bp = QPushButton("⊙ Clear BPs")
        btn_clear_bp.setAutoDefault(False)
        btn_clear_bp.clicked.connect(self.editor.clear_all_breakpoints)
        btn_clear_bp.setToolTip("Clear all breakpoints")
        action_lyt.addWidget(btn_clear_bp)

        action_lyt.addStretch()

        self.label_status = QLabel("Status: Ready")
        action_lyt.addWidget(self.label_status)

        layout.addWidget(action_bar)

        # Keyboard shortcuts
        sc_find = QShortcut(QKeySequence("Ctrl+F"), self.editor)
        sc_find.activated.connect(self._on_find)
        sc_find_replace = QShortcut(QKeySequence("Ctrl+H"), self.editor)
        sc_find_replace.activated.connect(self._on_find_replace)

        # Debug shortcuts (only active when debug toolbar is shown)
        self._sc_continue = QShortcut(QKeySequence("F5"), self)
        self._sc_continue.activated.connect(self.tool._debug_continue)
        self._sc_step_over = QShortcut(QKeySequence("F10"), self)
        self._sc_step_over.activated.connect(self.tool._debug_step_over)
        self._sc_step_into = QShortcut(QKeySequence("F11"), self)
        self._sc_step_into.activated.connect(self.tool._debug_step_into)
        sc_step_out = QShortcut(QKeySequence("Shift+F11"), self)
        sc_step_out.activated.connect(self.tool._debug_step_out)

        # Share editor reference with the tool so run() can query current breakpoints
        self.tool._debug_editor = self.editor

        return panel

    def _on_find(self):
        sel_text = ""
        if self.editor.hasSelectedText():
            sel_text = self.editor.selectedText()
        self.find_bar.show(text=sel_text)

    def _on_find_replace(self):
        self._on_find()
        self.find_bar.input_replace.setFocus()

    def _on_debugger_paused(self, line, locals_dict):
        self.debug_toolbar.show()
        self.var_inspector.show()
        self.watch_panel.show()
        self.debug_container.show()
        self.editor.set_debug_line(line)
        self.var_inspector.update_variables(locals_dict)
        self.watch_panel.update_values(locals_dict)
        self.debug_toolbar.set_state(f"Paused at line {line}", WARNING)

    def _on_debugger_resumed(self):
        self.editor.clear_debug_line()
        if not self.tool._debug_active:
            self.var_inspector.clear_variables()
            self.watch_panel.clear_values()
            self.label_status.setText("Status: Debug finished")
            self.label_status.setStyleSheet(f"color: {INFO};")
        self._update_debug_visibility()

    def _on_debug_stop(self):
        self.tool._stop_debug()
        self.editor.clear_debug_line()
        self.var_inspector.clear_variables()
        self.watch_panel.clear_values()
        self.label_status.setText("Status: Debug finished")
        self.label_status.setStyleSheet(f"color: {INFO};")
        self._update_debug_visibility()

    def _update_debug_visibility(self):
        """Show debug panel when breakpoints exist or debugging is active."""
        has_bps = len(self.tool._get_breakpoints()) > 0
        is_active = self.tool._debug_active
        if has_bps or is_active:
            self.debug_container.show()
            self.debug_toolbar.show()
            self.var_inspector.show()
            self.watch_panel.show()
            # Restore splitter ratio when showing
            total = sum(self.script_splitter.sizes())
            if total > 0 and self.script_splitter.sizes()[1] < 50:
                self.script_splitter.setSizes([700, 300])
        else:
            self.debug_container.hide()
            self.var_inspector.clear_variables()
            self.watch_panel.clear_values()

    # --- Terminals tab ---

    def _build_terminals_tab(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)

        from falcocv.core.config import ConfigManager
        config = ConfigManager()
        c_in = config.get_color("tool_input")
        c_out = config.get_color("tool_output")

        # Inputs
        input_container = QWidget()
        input_vbox = QVBoxLayout(input_container)
        input_vbox.addWidget(QLabel(f"<b style='color: {c_in};'>INPUTS</b>"))
        self.tbl_inputs = QTableWidget(0, 1)
        self.tbl_inputs.horizontalHeader().setStretchLastSection(True)
        self.tbl_inputs.horizontalHeader().hide()
        input_vbox.addWidget(self.tbl_inputs)

        btn_in_row = QHBoxLayout()
        btn_add_in = QPushButton("+ Add Input")
        btn_add_in.setAutoDefault(False)
        btn_add_in.clicked.connect(self._on_add_input)
        btn_del_in = QPushButton("- Delete Input")
        btn_del_in.setAutoDefault(False)
        btn_del_in.clicked.connect(self._on_del_input)
        btn_in_row.addWidget(btn_add_in)
        btn_in_row.addWidget(btn_del_in)
        input_vbox.addLayout(btn_in_row)
        layout.addWidget(input_container)

        # Outputs
        output_container = QWidget()
        output_vbox = QVBoxLayout(output_container)
        output_vbox.addWidget(QLabel(f"<b style='color: {c_out};'>OUTPUTS</b>"))
        self.tbl_outputs = QTableWidget(0, 1)
        self.tbl_outputs.horizontalHeader().setStretchLastSection(True)
        self.tbl_outputs.horizontalHeader().hide()
        output_vbox.addWidget(self.tbl_outputs)

        btn_out_row = QHBoxLayout()
        btn_add_out = QPushButton("+ Add Output")
        btn_add_out.setAutoDefault(False)
        btn_add_out.clicked.connect(self._on_add_output)
        btn_del_out = QPushButton("- Delete Output")
        btn_del_out.setAutoDefault(False)
        btn_del_out.clicked.connect(self._on_del_output)
        btn_out_row.addWidget(btn_add_out)
        btn_out_row.addWidget(btn_del_out)
        output_vbox.addLayout(btn_out_row)
        layout.addWidget(output_container)

        self._refresh_tables()
        self.tbl_inputs.cellChanged.connect(self._on_table_changed)
        self.tbl_outputs.cellChanged.connect(self._on_table_changed)

        return panel

    # --- Help tab ---

    def _build_help_tab(self):
        from PyQt6.QtWidgets import QTextEdit

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet(f"""
            QTextEdit {{
                background: {SCRIPT_BG}; color: {TEXT}; border: none;
                font: 12px Consolas;
            }}
        """)
        text.setHtml(r"""
<style>
  body { color: #D4D4D4; font-family: Consolas, monospace; font-size: 11pt; }
  h2 { color: #569CD6; margin-top: 18px; }
  h3 { color: #D7BA7D; margin-top: 14px; }
  kbd { background: #2D2D2D; border: 1px solid #555; border-radius: 3px;
        padding: 1px 5px; font-family: Consolas; font-size: 10pt; color: #E5C07B; }
  table { border-collapse: collapse; margin: 8px 0; width: 100%; }
  th { background: #252526; color: #858585; text-align: left; padding: 4px 8px;
       border: 1px solid #3C3C3C; font-weight: bold; }
  td { padding: 4px 8px; border: 1px solid #3C3C3C; }
  .key { color: #E5C07B; font-weight: bold; }
  .desc { color: #CE9178; }
  .note { color: #6A9955; }
</style>

<h2>Script Editor — Debugging Guide</h2>

<h3>Keyboard Shortcuts</h3>
<table>
<tr><th>Key</th><th>Action</th></tr>
<tr><td><kbd>F9</kbd></td><td>Toggle breakpoint on current line</td></tr>
<tr><td><kbd>F5</kbd></td><td>Continue execution (until next breakpoint)</td></tr>
<tr><td><kbd>F10</kbd></td><td>Step Over — execute current line, pause on next</td></tr>
<tr><td><kbd>F11</kbd></td><td>Step Into — enter function call and pause</td></tr>
<tr><td><kbd>Shift+F11</kbd></td><td>Step Out — run until current function returns</td></tr>
<tr><td><kbd>Ctrl+F</kbd></td><td>Find</td></tr>
<tr><td><kbd>Ctrl+H</kbd></td><td>Find &amp; Replace</td></tr>
</table>

<h3>How to Debug</h3>

<ol>
  <li><b>Set breakpoints</b> — click the left margin or press <kbd>F9</kbd> on any line.
      A red dot appears on lines with active breakpoints.</li>
  <li><b>Run the workflow</b> — when the script executes, the debugger activates automatically
      and pauses at the first breakpoint it encounters.</li>
  <li><b>Use the debug toolbar</b> that appears at the bottom of the editor:
    <ul>
      <li><span class="key">▶ Continue</span> — resume execution until the next breakpoint</li>
      <li><span class="key">⬇ Step Over</span> — execute one line (don't enter functions)</li>
      <li><span class="key">↘ Step Into</span> — enter a function call to debug inside it</li>
      <li><span class="key">⬆ Step Out</span> — finish the current function and return</li>
      <li><span class="key">■ Stop</span> — terminate the debugging session immediately</li>
    </ul>
  </li>
  <li><b>Inspect variables</b> — when paused, all local variables appear in the
      <b>Variable Inspector</b> panel below the toolbar. Values update each time
      execution pauses.</li>
</ol>

<h3>Breakpoint Tips</h3>
<ul>
  <li>Breakpoints persist while the config window is open. When you close the
      window, all breakpoints are automatically cleared so they don't block
      the next workflow run.</li>
  <li>You can set breakpoints on any executable line. Setting one on a
      comment, blank line, or <span class="desc">def</span> line will pause
      on the next valid line instead.</li>
  <li>Use breakpoints together with <kbd>F10</kbd> (Step Over) to inspect code
      line-by-line without entering helper functions.</li>
</ul>

<h3>Editor Features</h3>
<ul>
  <li><b>Syntax highlighting</b> — Python keywords, strings, numbers, comments,
      and decorators are color-coded.</li>
  <li><b>Auto-completion</b> — after typing 2+ characters, matching words from
      the document are suggested.</li>
  <li><b>Code folding</b> — collapse/expand functions and class blocks using the
      <span class="note">−</span>/<span class="note">+</span> icons in the fold margin.</li>
  <li><b>Brace matching</b> — matching parentheses, brackets, and braces are highlighted.</li>
  <li><b>Indentation guides</b> — vertical lines show indentation depth.</li>
  <li><b>Guideline at column 88</b> — a vertical line helps keep lines within the
      recommended length.</li>
</ul>

<h3>Troubleshooting</h3>
<ul>
  <li><b>Script error on run:</b> check the status bar at the bottom of the
      Script tab for the error message.</li>
  <li><b>Debugger doesn't pause:</b> make sure at least one breakpoint is set
      (red dot in the margin). The debugger only activates when breakpoints exist.</li>
  <li><b>Breakpoints not working:</b> verify the script uses <span class="desc">process(self, inputs)</span>
      — the entry point the workflow engine calls.</li>
  <li><b>Changes not applied:</b> click <span class="desc">⟳ Reload / Compile</span>
      to reload the module before running the workflow.</li>
</ul>
""")
        layout.addWidget(text)

        return panel

    @safe_refresh
    def refresh_ui(self):
        self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")

    # --- Script tab handlers ---

    def _on_path_changed(self, text):
        self.tool.params["script_path"] = text

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Python Script", "", "Python Files (*.py);;All Files (*)"
        )
        if path:
            self.edit_path.setText(path)

    def _on_new_script(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Create New Python Script", "logic.py", "Python Files (*.py)"
        )
        if path:
            if not path.endswith('.py'):
                path += '.py'
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(DEFAULT_TEMPLATE)
                self.tool.params["script_path"] = path
                self.tool.params["script_code"] = DEFAULT_TEMPLATE
                self.edit_path.setText(path)
                self.editor.setText(DEFAULT_TEMPLATE)
                self.tool.data_changed.emit()
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to create script: {e}")

    def _on_reload(self):
        # Replace tabs with 4 spaces before compiling
        text = self.editor.text()
        if '\t' in text:
            text = text.replace('\t', ' ' * 4)
            self.editor.setText(text)

        path = self.tool.params.get("script_path")
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
                self.editor.setText(code)
                self.tool.params["script_code"] = code
                self.label_status.setText(f"Status: Loaded from {os.path.basename(path)}")
                self.label_status.setStyleSheet(f"color: {INFO};")
            except Exception as e:
                self.label_status.setText(f"Status: ❌ Failed to read file: {e}")
                self.label_status.setStyleSheet(f"color: {WARNING};")
                return

        try:
            self.tool._instance = None
            self.tool._load_module()
            self.label_status.setText("Status: ✅ Script loaded successfully")
            self.label_status.setStyleSheet(f"color: {SUCCESS};")
            self.tool.data_changed.emit()
        except Exception as e:
            self.label_status.setText(f"Status: ❌ {e}")
            self.label_status.setStyleSheet(f"color: {WARNING};")

    def _on_save(self):
        path = self.tool.params.get("script_path")
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Script", "custom_logic.py", "Python Files (*.py)"
            )
            if not path:
                return
            if not path.endswith('.py'):
                path += '.py'
            self.edit_path.setText(path)
            self.tool.params["script_path"] = path

        try:
            code = self.editor.text()
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            self.tool.params["script_code"] = code
            self.label_status.setText(f"Status: 💾 Saved to {os.path.basename(path)}")
            self.label_status.setStyleSheet(f"color: {SUCCESS};")
        except Exception as e:
            self.label_status.setText(f"Status: ❌ Save failed: {e}")
            self.label_status.setStyleSheet(f"color: {WARNING};")

    def _on_code_changed(self):
        self.tool.params["script_code"] = self.editor.text()

    # --- Terminals tab handlers ---

    def _refresh_tables(self):
        self.tbl_inputs.blockSignals(True)
        self.tbl_inputs.setRowCount(0)
        for p in self.tool.params.get("input_ports", []):
            self._add_row(self.tbl_inputs, p)
        self.tbl_inputs.blockSignals(False)

        self.tbl_outputs.blockSignals(True)
        self.tbl_outputs.setRowCount(0)
        for p in self.tool.params.get("output_ports", []):
            self._add_row(self.tbl_outputs, p)
        self.tbl_outputs.blockSignals(False)

    def _add_row(self, table, text):
        r = table.rowCount()
        table.insertRow(r)
        table.setItem(r, 0, QTableWidgetItem(text))

    def _on_add_input(self):
        self.tool.add_input_port()
        self._refresh_tables()
        self.tool.structure_changed.emit()
        self.tool.data_changed.emit()

    def _on_add_output(self):
        self.tool.add_output_port()
        self._refresh_tables()
        self.tool.structure_changed.emit()
        self.tool.data_changed.emit()

    def _on_del_input(self):
        row = self.tbl_inputs.currentRow()
        if row >= 0:
            self.tool.remove_input_port(row)
            self._refresh_tables()
            self.tool.structure_changed.emit()
            self.tool.data_changed.emit()

    def _on_del_output(self):
        row = self.tbl_outputs.currentRow()
        if row >= 0:
            self.tool.remove_output_port(row)
            self._refresh_tables()
            self.tool.structure_changed.emit()
            self.tool.data_changed.emit()

    def _on_table_changed(self):
        in_ports = []
        for r in range(self.tbl_inputs.rowCount()):
            item = self.tbl_inputs.item(r, 0)
            if item and item.text().strip():
                in_ports.append(item.text().strip())

        out_ports = []
        for r in range(self.tbl_outputs.rowCount()):
            item = self.tbl_outputs.item(r, 0)
            if item and item.text().strip():
                out_ports.append(item.text().strip())

        self.tool.set_input_ports(in_ports)
        self.tool.set_output_ports(out_ports)
        self._refresh_tables()
        self.tool.structure_changed.emit()
        self.tool.data_changed.emit()
