# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['face_album_qt.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
              'PySide6.Qt3DCore', 'PySide6.QtMultimedia', 'PySide6.QtQml',
              'PySide6.QtQuick', 'PySide6.QtQuickWidgets', 'PySide6.QtCharts',
              'PySide6.QtDataVisualization', 'PySide6.QtBluetooth',
              'PySide6.QtNfc', 'PySide6.QtPositioning', 'PySide6.QtSensors',
              'PySide6.QtSerialPort', 'PySide6.QtSql', 'PySide6.QtTest',
              'PySide6.QtWebChannel', 'PySide6.QtWebSockets', 'PySide6.QtPdf',
              'PySide6.QtHelp', 'PySide6.QtDesigner', 'PySide6.QtUiTools',
              'PySide6.QtRemoteObjects', 'tkinter'],
    noarchive=False,
    optimize=0,
)

# ★ 剔除 onnxruntime 自带的 CUDA/cuDNN 大体积 DLL（合计约 2.5GB）。
#   程序主要跑 CPU（GPU 勾选也只是优先尝试，失败会自动退回 CPU），
#   不带这些库能保持 exe 只有一百多 MB，与之前的发布版一致。
_CUDA_KW = ('cublas', 'cudnn', 'cuda', 'cudart', 'nvrtc', 'nvjitlink',
            'cufft', 'curand', 'cusolver', 'cusparse')
a.binaries = [b for b in a.binaries
              if not any(k in b[0].lower() for k in _CUDA_KW)]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='人脸聚类相册',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
