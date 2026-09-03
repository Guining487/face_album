@echo off
chcp 65001 >nul
REM 人脸聚类相册 —— 图形界面一键启动（无控制台黑窗口）
call E:\MiniConda\Scripts\activate.bat face-album
start "" pythonw E:\face_album\face_album_gui.py
