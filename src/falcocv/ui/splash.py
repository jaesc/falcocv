"""FalcoCV splash screen.

Generates a branded splash programmatically (no external image needed).
Shows version, build info, and loading progress during startup.
"""

import os

from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtGui import QPainter, QColor, QFont, QPixmap
from PyQt6.QtCore import Qt, QRect
from falcocv.ui.style import DARK_BG, INFO, TEXT, TEXT_DIM, TEXT_MUTED, WHITE
from falcocv.version import __version__, __build__

SPLASH_W = 520
SPLASH_H = 320


def _render_splash() -> QPixmap:
    """Paint the splash screen pixmap."""
    bg = QColor(DARK_BG)
    accent = QColor(INFO)
    text_color = QColor(TEXT)
    muted = QColor(TEXT_MUTED)
    white = QColor(WHITE)

    pix = QPixmap(SPLASH_W, SPLASH_H)
    pix.fill(bg)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Accent bar at the top
    p.fillRect(0, 0, SPLASH_W, 4, accent)

    # Brand lockup (mark + wordmark)
    logo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "logo_lockup_dark.png"
    )
    logo = QPixmap(logo_path) if os.path.exists(logo_path) else QPixmap()
    if not logo.isNull():
        logo = logo.scaledToHeight(72, Qt.TransformationMode.SmoothTransformation)
        p.drawPixmap(30, 44, logo)
    else:
        p.setPen(white)
        title_font = QFont("Segoe UI", 28, QFont.Weight.Bold)
        p.setFont(title_font)
        p.drawText(QRect(30, 60, SPLASH_W - 60, 50), Qt.AlignmentFlag.AlignLeft, "FalcoCV")

    # Subtitle / tagline
    p.setPen(text_color)
    sub_font = QFont("Segoe UI", 12)
    p.setFont(sub_font)
    p.drawText(QRect(30, 132, SPLASH_W - 60, 30), Qt.AlignmentFlag.AlignLeft,
               "Modular Machine Vision Environment")

    # Version
    p.setPen(muted)
    info_font = QFont("Segoe UI", 10)
    p.setFont(info_font)
    p.drawText(QRect(30, 168, SPLASH_W - 60, 20), Qt.AlignmentFlag.AlignLeft,
               f"v{__version__}  build {__build__}")

    # Bottom accent bar (thinner)
    p.fillRect(0, SPLASH_H - 3, SPLASH_W, 3, accent)

    # Footer message placeholder (drawn by showMessage later)
    p.setPen(QColor(TEXT_DIM))
    foot_font = QFont("Segoe UI", 9)
    p.setFont(foot_font)
    p.drawText(QRect(30, SPLASH_H - 50, SPLASH_W - 60, 20),
               Qt.AlignmentFlag.AlignLeft,
               "Loading…")

    p.end()
    return pix


def create_splash() -> QSplashScreen:
    """Build and return a QSplashScreen ready to show()."""
    pix = _render_splash()
    splash = QSplashScreen(pix)
    splash.setWindowFlags(
        Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
    )
    return splash
