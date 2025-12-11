# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置文件
使用方法: pyinstaller iris_service.spec
"""

import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# 收集隐藏导入
hiddenimports = [
    'flask',
    'flask_cors',
    'cv2',
    'numpy',
    'pywt',
    'PIL',
    'PIL.Image',
]

# 添加 cv2 和 numpy 的子模块
hiddenimports += collect_submodules('cv2')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('pywt')

a = Analysis(
    ['iris_service.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('util', 'util'),  # 包含 util 目录
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',      # 不需要 GUI
        'matplotlib',   # 不需要绑定
        'PyQt5',
        'PyQt6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='iris_service',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

