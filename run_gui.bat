@echo off
chcp 65001 >nul
REM 隐藏控制台窗口并用 pythonw 启动 GUI（相对脚本所在目录，方便拷贝到别处）
set "ROOT=%~dp0"

REM 尝试用 conda 环境 face-album 激活；若没有 conda 则直接用系统 python
where conda >nul 2>nul
if %errorlevel%==0 (
    call conda activate face-album 2>nul
)

start "" pythonw "%ROOT%face_album_qt.py"
