# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building social-graph as a standalone binary."""

import sys

a = Analysis(
    ["src/social_graph/cli.py"],
    pathex=["src"],
    binaries=[],
    datas=[],
    hiddenimports=["anthropic"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="social-graph",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    codesign_identity=None,
    entitlements_file=None,
)
