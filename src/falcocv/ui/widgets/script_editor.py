from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PyQt6.Qsci import QsciScintilla, QsciLexerPython

from falcocv.ui.style import (
    SCRIPT_BG, SCRIPT_TEXT, SCRIPT_KEYWORD, SCRIPT_NUMBER, SCRIPT_COMMENT,
    SCRIPT_DECORATOR, SCRIPT_BP_BG, SCRIPT_BP_FG, SCRIPT_ARROW_BG, SCRIPT_ARROW_FG,
    SCRIPT_HIGHLIGHT_BG, SCRIPT_MARGIN_BG, SCRIPT_FOLD_EDGE, SCRIPT_MARGIN_FG,
    SCRIPT_LINE_HIGHLIGHT, SCRIPT_CARET, SCRIPT_SELECTION_BG, SCRIPT_EDGE,
    SCRIPT_DEBUG_LINE_BG, SCRIPT_INPUT_BG, SCRIPT_BAR_BG,
    TEXT, TEXT_MUTED, BORDER, SUCCESS, DANGER,
)


class FindReplaceBar(QWidget):
    def __init__(self, editor, parent=None):
        super().__init__(parent)
        self._editor = editor
        self._last_search = ""
        self._case_sensitive = False
        self._regex_mode = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # Find row
        find_row = QHBoxLayout()
        find_row.setSpacing(4)
        lbl_find = QLabel("Find:")
        lbl_find.setStyleSheet(f"color: {TEXT};")
        self.input_find = QLineEdit()
        self.input_find.setPlaceholderText("Search...")
        self.input_find.setStyleSheet(f"background: {SCRIPT_INPUT_BG}; color: {TEXT}; border: 1px solid {BORDER}; padding: 2px 4px;")
        self.input_find.textChanged.connect(lambda: self._do_find(from_start=True))
        self.input_find.returnPressed.connect(self._find_next)
        find_row.addWidget(lbl_find)
        find_row.addWidget(self.input_find, 1)

        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        find_row.addWidget(self.lbl_count)

        self.btn_prev = QPushButton("\u25b2")
        self.btn_prev.setFixedWidth(28)
        self.btn_prev.clicked.connect(self._find_prev)
        self.btn_prev.setToolTip("Find Previous (Shift+Enter)")
        find_row.addWidget(self.btn_prev)

        self.btn_next = QPushButton("\u25bc")
        self.btn_next.setFixedWidth(28)
        self.btn_next.clicked.connect(self._find_next)
        self.btn_next.setToolTip("Find Next (Enter)")
        find_row.addWidget(self.btn_next)

        self.chk_case = QCheckBox("Aa")
        self.chk_case.setToolTip("Case Sensitive")
        self.chk_case.setStyleSheet(f"color: {TEXT};")
        self.chk_case.toggled.connect(lambda v: setattr(self, '_case_sensitive', v))
        find_row.addWidget(self.chk_case)

        self.chk_regex = QCheckBox(".*")
        self.chk_regex.setToolTip("Regex")
        self.chk_regex.setStyleSheet(f"color: {TEXT};")
        self.chk_regex.toggled.connect(lambda v: setattr(self, '_regex_mode', v))
        find_row.addWidget(self.chk_regex)

        self.btn_close = QPushButton("\u2715")
        self.btn_close.setFixedWidth(22)
        self.btn_close.clicked.connect(self.hide)
        self.btn_close.setFlat(True)
        self.btn_close.setStyleSheet(f"color: {TEXT_MUTED};")
        find_row.addWidget(self.btn_close)

        layout.addLayout(find_row)

        # Replace row
        replace_row = QHBoxLayout()
        replace_row.setSpacing(4)
        lbl_rep = QLabel("Rep:")
        lbl_rep.setStyleSheet(f"color: {TEXT};")
        self.input_replace = QLineEdit()
        self.input_replace.setPlaceholderText("Replace...")
        self.input_replace.setStyleSheet(f"background: {SCRIPT_INPUT_BG}; color: {TEXT}; border: 1px solid {BORDER}; padding: 2px 4px;")
        replace_row.addWidget(lbl_rep)
        replace_row.addWidget(self.input_replace, 1)

        self.btn_replace = QPushButton("Replace")
        self.btn_replace.clicked.connect(self._replace)
        replace_row.addWidget(self.btn_replace)

        self.btn_replace_all = QPushButton("All")
        self.btn_replace_all.clicked.connect(self._replace_all)
        replace_row.addWidget(self.btn_replace_all)

        replace_row.addStretch()
        layout.addLayout(replace_row)

        self.setStyleSheet(f"background-color: {SCRIPT_BAR_BG}; border-top: 1px solid {BORDER};")
        self.hide()

    def show(self, text=None):
        super().show()
        if text:
            self.input_find.setText(text)
        self.input_find.setFocus()
        self.input_find.selectAll()
        self._find_next()

    def _do_find(self, forward=True, from_start=False):
        text = self.input_find.text()
        if not text:
            return
        self._last_search = text
        line, index = (-1, -1) if from_start else (self._editor.getCursorPosition()[0], -1)
        found = self._editor.findFirst(
            text, self._regex_mode, self._case_sensitive,
            False, True,
            forward,
            line, index,
            False
        )
        if found:
            self.input_find.setStyleSheet(f"background: #3C3C3C; color: {TEXT}; border: 1px solid {SUCCESS}; padding: 2px 4px;")
            self.lbl_count.setText("")
        else:
            self.input_find.setStyleSheet(f"background: #3C3C3C; color: {DANGER}; border: 1px solid {DANGER}; padding: 2px 4px;")
            self.lbl_count.setText("0 matches")

    def _find_next(self):
        self._do_find(forward=True)

    def _find_prev(self):
        self._do_find(forward=False)

    def _replace(self):
        text = self.input_find.text()
        repl = self.input_replace.text()
        if not text:
            return
        if self._editor.hasSelectedText():
            self._editor.replaceSelectedText(repl)
        self._find_next()

    def _replace_all(self):
        text = self.input_find.text()
        repl = self.input_replace.text()
        if not text:
            return
        self._editor.replaceAll(text, repl, self._regex_mode, self._case_sensitive, False, False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self._editor.setFocus()
        elif event.key() == Qt.Key.Key_Return and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._find_prev()
        elif event.key() == Qt.Key.Key_Return:
            self._find_next()
        else:
            super().keyPressEvent(event)


class ScriptEditor(QsciScintilla):
    breakpoints_changed = pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._breakpoints = set()
        self._debug_line = -1
        self._find_bar = None

        # --- Lexer ---
        lexer = QsciLexerPython(self)
        lexer.setPaper(QColor(SCRIPT_BG))
        lexer.setDefaultPaper(QColor(SCRIPT_BG))
        lexer.setDefaultColor(QColor(SCRIPT_TEXT))
        lexer.setColor(QColor(SCRIPT_KEYWORD), QsciLexerPython.Keyword)
        lexer.setColor(QColor(SCRIPT_NUMBER), QsciLexerPython.Number)
        lexer.setColor(QColor(SCRIPT_COMMENT), QsciLexerPython.Comment)
        lexer.setColor(QColor(SCRIPT_COMMENT), QsciLexerPython.CommentBlock)
        lexer.setColor(QColor(SCRIPT_NUMBER), QsciLexerPython.DoubleQuotedString)
        lexer.setColor(QColor(SCRIPT_NUMBER), QsciLexerPython.SingleQuotedString)
        lexer.setColor(QColor(SCRIPT_NUMBER), QsciLexerPython.TripleDoubleQuotedString)
        lexer.setColor(QColor(SCRIPT_NUMBER), QsciLexerPython.TripleSingleQuotedString)
        lexer.setColor(QColor(SCRIPT_NUMBER), QsciLexerPython.DoubleQuotedFString)
        lexer.setColor(QColor(SCRIPT_NUMBER), QsciLexerPython.SingleQuotedFString)
        lexer.setColor(QColor(SCRIPT_NUMBER), QsciLexerPython.TripleDoubleQuotedFString)
        lexer.setColor(QColor(SCRIPT_NUMBER), QsciLexerPython.TripleSingleQuotedFString)
        lexer.setColor(QColor(SCRIPT_DECORATOR), QsciLexerPython.Decorator)
        lexer.setFont(QFont("Consolas", 10))
        self.setLexer(lexer)

        # HMI API autocompletion
        import os
        from PyQt6.Qsci import QsciAPIs
        api_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "hmi_api.api")
        if os.path.isfile(api_path):
            api = QsciAPIs(lexer)
            api.load(api_path)
            api.prepare()
            lexer.setAPIs(api)

        # Line numbers
        self.setMarginType(0, QsciScintilla.MarginType.NumberMargin)
        self.setMarginWidth(0, "000")
        self.setMarginLineNumbers(0, True)

        # Breakpoint margin
        self.setMarginType(1, QsciScintilla.MarginType.SymbolMargin)
        self.setMarginWidth(1, 18)
        self.setMarginSensitivity(1, True)
        self.marginClicked.connect(self._on_margin_clicked)
        self.markerDefine(QsciScintilla.MarkerSymbol.Circle, 0)
        self.markerDefine(QsciScintilla.MarkerSymbol.RightArrow, 1)
        self.markerDefine(QsciScintilla.MarkerSymbol.Background, 2)
        self.setMarkerBackgroundColor(QColor(SCRIPT_BP_BG), 0)
        self.setMarkerForegroundColor(QColor(SCRIPT_BP_FG), 0)
        self.setMarkerBackgroundColor(QColor(SCRIPT_ARROW_BG), 1)
        self.setMarkerForegroundColor(QColor(SCRIPT_ARROW_FG), 1)
        self.setMarkerBackgroundColor(QColor(SCRIPT_HIGHLIGHT_BG), 2)

        # Folding
        self.setMarginType(2, QsciScintilla.MarginType.SymbolMargin)
        self.setMarginWidth(2, 14)
        self.setFolding(QsciScintilla.FoldStyle.PlainFoldStyle, 2)
        self.setFoldMarginColors(QColor(SCRIPT_MARGIN_BG), QColor(SCRIPT_FOLD_EDGE))

        self.setMarginsBackgroundColor(QColor(SCRIPT_MARGIN_BG))
        self.setMarginsForegroundColor(QColor(SCRIPT_MARGIN_FG))

        # Caret
        self.setCaretLineVisible(True)
        self.setCaretLineBackgroundColor(QColor(SCRIPT_LINE_HIGHLIGHT))
        self.setCaretWidth(2)
        self.setCaretForegroundColor(QColor(SCRIPT_CARET))

        # Indentation
        self.setIndentationGuides(True)
        self.setIndentationWidth(4)
        self.setTabWidth(4)
        self.setAutoIndent(True)

        # Brace matching
        self.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self.setMatchedBraceBackgroundColor(QColor(SCRIPT_LINE_HIGHLIGHT))
        self.setMatchedBraceForegroundColor(QColor(SCRIPT_TEXT))

        # Selection
        self.setSelectionBackgroundColor(QColor(SCRIPT_SELECTION_BG))
        self.setSelectionForegroundColor(QColor(SCRIPT_TEXT))

        # Edge line
        self.setEdgeMode(QsciScintilla.EdgeMode.EdgeLine)
        self.setEdgeColumn(88)
        self.setEdgeColor(QColor(SCRIPT_EDGE))

        self.setEolMode(QsciScintilla.EolMode.EolUnix)
        self.setPaper(QColor(SCRIPT_BG))
        self.setColor(QColor(SCRIPT_TEXT))

        # Auto-completion
        self.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAll)
        self.setAutoCompletionThreshold(2)
        self.setAutoCompletionCaseSensitivity(False)
        self.setAutoCompletionReplaceWord(True)

        # F9 to toggle breakpoint
        self._sc_breakpoint = QShortcut(QKeySequence("F9"), self)
        self._sc_breakpoint.activated.connect(self._toggle_breakpoint_at_cursor)

    # --- Breakpoints ---

    def _sync_breakpoints_from_markers(self):
        new_bps = set()
        mask = 1 << 0
        line = -1
        while True:
            line = self.markerFindNext(line + 1, mask)
            if line < 0:
                break
            new_bps.add(line)
        self._breakpoints = new_bps

    def _toggle_breakpoint_at_cursor(self):
        line, _ = self.getCursorPosition()
        self._toggle_breakpoint(line)

    def _toggle_breakpoint(self, line):
        self._sync_breakpoints_from_markers()
        if line in self._breakpoints:
            self.markerDelete(line, 0)
            self._breakpoints.discard(line)
        else:
            self.markerAdd(line, 0)
            self._breakpoints.add(line)
        self.breakpoints_changed.emit(self.get_breakpoints())

    def clear_all_breakpoints(self):
        self.markerDeleteAll(0)
        self._breakpoints.clear()
        self.breakpoints_changed.emit(set())

    def clear_breakpoints(self):
        self.clear_all_breakpoints()

    def set_debug_line(self, line):
        if self._debug_line >= 0:
            self.markerDelete(self._debug_line, 1)
            self.markerDelete(self._debug_line, 2)
        qline = line - 1
        self._debug_line = qline
        self.markerAdd(qline, 1)
        self.markerAdd(qline, 2)
        self.setCaretLineBackgroundColor(QColor(SCRIPT_DEBUG_LINE_BG))
        self.setCursorPosition(qline, 0)
        self.repaint()

    def clear_debug_line(self):
        if self._debug_line >= 0:
            self.markerDelete(self._debug_line, 1)
            self.markerDelete(self._debug_line, 2)
        self._debug_line = -1
        self.setCaretLineBackgroundColor(QColor(SCRIPT_LINE_HIGHLIGHT))

    def set_find_bar(self, bar):
        self._find_bar = bar

    def _on_margin_clicked(self, margin, line, modifiers):
        if margin == 1:
            self._toggle_breakpoint(line)

    def get_breakpoints(self):
        self._sync_breakpoints_from_markers()
        return {b + 1 for b in self._breakpoints}
