# -*- mode: python ; coding: utf-8 -*-
import os

# ★ 打包成单个 exe，尽量压缩体积：
#   1. 人脸模型(buffalo_l) / GPU 组件都不内置 —— 程序会在首次使用时联网下载；
#   2. 剔除 onnxruntime 自带的 CUDA/cuDNN 大体积 DLL（合计约 2.5GB）；
#   3. 排除用不到的 Qt 模块；
#   4. optimize=1（去掉 assert；不能开 2，会剥掉 docstring 导致 numpy 启动报错）
#      配合 UPX 压缩（构建机器上需有 upx.exe 在 PATH）。

# insightface 的 data/objects/meanshape_68.pkl（姿态估计必需）默认不会被打包。
# insightface 在打包(frozen)状态下会去 sys._MEIPASS/objects/ 找，所以整目录放到 exe 根。
import insightface as _insightface
_IF_DATA = os.path.join(os.path.dirname(_insightface.__file__), 'data')

a = Analysis(
    ['face_album_qt.py'],
    pathex=[],
    binaries=[],
    datas=[('star.png', '.'),
           (_IF_DATA, '.')],
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
    optimize=1,
)

# ★ 剔除 onnxruntime 自带的 CUDA/cuDNN/TensorRT 大体积 DLL（合计约 2.6GB）。
#   程序主要跑 CPU（GPU 勾选也只是优先尝试，失败会自动退回 CPU），
#   不带这些库能保持 exe 体积小。
_CUDA_KW = ('cublas', 'cudnn', 'cuda', 'cudart', 'nvrtc', 'nvjitlink',
            'cufft', 'curand', 'cusolver', 'cusparse',
            'nvinfer', 'nvonnxparser', 'onnxruntime_providers_tensorrt')
a.binaries = [b for b in a.binaries
              if not any(k in b[0].lower() for k in _CUDA_KW)]

# ★ 进一步剔除根本用不到的 Qt 模块 DLL（本程序只用 QtWidgets）。
#   Qt6Qml/Quick/Pdf/VirtualKeyboard/Network/OpenGL 模块都不会被加载，
#   OpenGL 软件渲染器 opengl32sw.dll 仍保留（兼容无显卡的机器）。
_QSKIP = ('qt6qml', 'qt6quick', 'qt6pdf', 'qt6virtualkeyboard',
          'qt6network', 'qt6opengl.dll')

# ★ 本程序只用 cv2 读图（imread），不读视频 → 去掉 OpenCV 的 ffmpeg 插件(10MB)。
_EXTRA_SKIP = ('opencv_videoio_ffmpeg',)

a.binaries = [b for b in a.binaries
              if not any(k in b[0].lower() for k in (_QSKIP + _EXTRA_SKIP))]
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
    upx_exclude=['onnxruntime.dll'],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
