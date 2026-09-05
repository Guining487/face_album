# -*- coding: utf-8 -*-
"""人脸聚类相册 —— Qt (PySide6) 毛玻璃版。

把原来的 tkinter 界面整体迁移到 Qt，并升级成「毛玻璃 + 圆角 + 丝滑动画」风格：
  · 全局背景：复用当前 VSCode 的背景图（E:\\vscode_background\\star.png），高斯模糊 + 压暗后铺满，
    窗口四周做成大圆角，玻璃质感。
  · 所有面板 / 卡片：半透明磨砂材质（低透明度 + 细边框），自带悬停 / 按压 / 上浮动画。
  · 聚类等待时的名言卡片：从进度条下方「模块下放」式滑出，轮播切换，结束收起。
  · 聚类 / 导出 / 命名 / 快捷方式导出等业务逻辑与旧版完全一致。

运行环境：Python 3.9+，PySide6。
"""

# ============================================================================
# ★ 给 Python / Qt 初学者的阅读导读（先读这里，后面会顺很多）
# ============================================================================
# 本程序 = “人脸聚类相册”。它扫描你选的照片文件夹，识别每张照片里的人脸，
# 把“同一个人”的照片归到一组，像手机相册那样按人物分组、可命名、可导出。
#
# 理解这份代码，先抓住下面几条主线：
#
#  1) 界面(UI) + 逻辑分离：
#       上部“画 UI”的代码(类名大写开头，如 MainWindow / GlassButton / FaceCard)
#       只管界面长什么样、怎么响应鼠标；
#       下部“算数据”的代码(函数名带下划线，如 collect_images / smart_cluster)
#       只管读图、做人脸特征、聚类，不碰界面。
#
#  2) 什么是“类”和“对象”：
#       class 头像名: ...        —— 先画一张“图纸”
#       头像名()                  —— 再按图纸造出一个“实物”
#       类里的 def __init__(self) —— 造实物时自动执行，self 代表“这个实物自己”。
#       调用成员函数要带对象：window.resize(100, 50)
#
#  3) Qt 的几个“新名词”（GUI 必用）：
#       QWidget    —— 一个“能放东西的矩形”，几乎所有界面元素都是它或它的子孙。
#       QLabel     —— 显示文字/图片的标签。
#       QPushButton—— 按钮。
#       QVBoxLayout—— 竖着排队的“布局”，把上下控件自动排列好。
#       setStyleSheet —— 用 CSS 语法给控件“化妆”（颜色/圆角/边框）。
#       Signal(信号) —— 控件对外广播“我怎么了”（点了/改了）；用 .connect(函数) 订阅。
#       self.update() —— 通知 Qt“我该重画了”，重画时会自动调用 paintEvent。
#       paintEvent —— 自己画自己的函数（QPainter 像一支画笔）。
#
#  4) 为什么界面不卡：多线程 + 消息队列。
#       耗时的“人脸识别/聚类”放到一个后台线程跑；线程算完把结果塞进一个 queue
#       (消息队列)；界面用 QTimer 每 120ms 来取一次消息，取到就更新界面。
#       界面绝不直接等后台，所以永远流畅。
#
#  5) 动画是怎么“丝滑”的：
#       用一个 QVariantAnimation/QPropertyAnimation 把某个值从 A 平滑变到 B，
#       每变一帧就调用一个回调去重画，Qt 按 60fps 驱动，看起来就是动画。
#
#  读的顺序建议：main → MainWindow(_build_ui) → GlassButton/FaceCard 的 paintEvent
#  → _start_cluster → _cluster_worker → _poll_queue。算法部分(smart_cluster 等)
#  是进阶内容，可先跳过。
# ============================================================================

# ---------------------------------------------------------------------------
# 导入（import：把别人写好的功能“借”进来用）
# ---------------------------------------------------------------------------
# --- Python 自带库 ---
import os            # 路径拼接、遍历文件夹、打开文件等操作系统功能
import sys           # 解释器信息（版本、是否打包成 exe）
import re            # 正则表达式：导出时清洗掉文件夹名里的非法字符
import shutil        # 文件操作：复制原图、移动 DLL
import subprocess    # 调用外部程序：用 PowerShell 批量建 .lnk 快捷方式
import tempfile      # 生成临时文件（快捷方式的 .ps1 脚本）
import threading     # 多线程：把耗时的聚类放到后台跑
import queue         # 线程安全的队列：后台线程 -> 界面 传消息
import urllib.request  # 联网下载人脸模型 / GPU 组件
import json          # 解析 PyPI 返回的数据
import zipfile       # 解压下载回来的模型压缩包
import traceback     # 出错时打印完整“调用栈”，方便排查

# --- 第三方科学计算 / 机器学习库 ---
import numpy as np              # 高性能数组与数学运算（人脸特征就是 np 数组）
import cv2                      # OpenCV：读图、裁剪、缩放、模糊（图像是 np 数组）
from PIL import Image           # Pillow：图像缩放（np 数组 <-> 图片）
from sklearn.cluster import DBSCAN   # 聚类算法：把相似的人脸特征自动归堆
import onnxruntime              # 推理引擎（人脸模型跑在上面）
from insightface.app import FaceAnalysis   # 现成的人脸检测 + 512 维特征提取

# --- Qt (PySide6) 界面库 ---
# QtCore：核心（坐标、定时器、动画、信号）
from PySide6.QtCore import (Qt, QRectF, QTimer, QPropertyAnimation,
                            QVariantAnimation, QEasingCurve, QAbstractAnimation,
                            QRect, Signal)
# QtGui：画笔/颜色/图片/字体
from PySide6.QtGui import (QPainter, QPainterPath, QColor, QPixmap, QImage, QFont,
                           QPen, QCursor, QLinearGradient, QBrush, QFontMetrics)
# QtWidgets：现成的界面控件（窗口、按钮、输入框……）
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QLineEdit,
                               QCheckBox, QRadioButton, QFrame, QScrollArea,
                               QGridLayout, QHBoxLayout, QVBoxLayout, QFileDialog,
                               QProgressBar, QDialog)

# ============================================================================
# 常量区：把“会变的配置值”集中写在这里，方便统一修改
# ============================================================================

# ---------------------------------------------------------------------------
# ① 人脸模型放哪里
# ---------------------------------------------------------------------------
# 人脸识别需要“模型文件”（约几百 MB），第一次用时联网下载。下面列出它可能
# 存放的位置：优先程序同目录，找不到就退回用户主目录（insightface 默认位置）。
_CANDIDATE_ROOTS = [
    # sys.frozen == True 表示“我是被打包成 exe 在跑”，此时程序目录是 exe 所在处
    os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
                 else os.path.dirname(os.path.abspath(__file__)), '.insightface'),
    os.path.join(os.path.expanduser('~'), '.insightface'),   # C:\Users\你的名字\.insightface
]
# 从候选里挑“第一个真的存在 models/buffalo_l 目录”的路径当模型根目录
MODEL_ROOT = next((r for r in _CANDIDATE_ROOTS
                   if os.path.isdir(os.path.join(r, 'models', 'buffalo_l'))),
                  _CANDIDATE_ROOTS[-1])

# ---------------------------------------------------------------------------
# ② 支持的图片格式 / 下载地址
# ---------------------------------------------------------------------------
SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}   # 只认这几种扩展名
MODEL_DL_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
GPU_RUNTIME_URL = ''            # 留空 = 从 PyPI 自动下载 GPU 组件
GPU_RUNTIME_DLLS = ('onnxruntime_providers_cuda.dll', 'cublas64_12.dll',
                    'cublasLt64_12.dll', 'cudart64_12.dll', 'cudnn64_9.dll')

# ---------------------------------------------------------------------------
# ③ 界面配色（想换主题只改这里几行）
#   QColor(r, g, b, a)：第 4 个数 a 是“透明度”，0=全透明，255=不透明
# ---------------------------------------------------------------------------
BG_IMG_PATH = r'E:\vscode_background\star.png'     # VSCode 用的星空背景图
PANEL_TINT = QColor(28, 34, 44, 95)                # 面板底色：深蓝灰、95 半透明
PANEL_BORDER = QColor(186, 200, 216, 70)           # 面板的细边框（浅蓝灰）
CARD_TINT = QColor(255, 255, 255, 26)              # 人物卡片底色：更透的“白玻璃”
CARD_HOVER = QColor(255, 255, 255, 46)             # 鼠标悬停时卡片亮一档
CARD_RADIUS = 14                                   # 卡片圆角半径
ACCENT = '#2f7bc4'                                 # 主色（Aero 蓝），进度条等用它
ACCENT_HOVER = '#5496d8'                           # 主色悬停变浅
ACCENT_PRESS = '#1d5fa0'                           # 主色按下变深
CLOSE_HOVER = '#c42b1c'                            # 关闭按钮悬停时的红

# 文字颜色（都带点透明度，柔和不刺眼）
TEXT = QColor(245, 247, 250)        # 主文字：近白
TEXT_SUB = QColor(245, 247, 250, 160)   # 次要文字
TEXT_DIM = QColor(245, 247, 250, 95)    # 更弱/占位文字

FONT_FAMILY = 'Microsoft YaHei'    # 默认中文字体（微软雅黑）

# ---------------------------------------------------------------------------
# ④ 聚类等待时轮播的名言（每次开始聚类挑一句，4 秒一换）
# ---------------------------------------------------------------------------
QUOTES = [
    "真正的朋友，是即使多年不见，一开口仍是当年的人。",
    "最好的友情，不是形影不离，而是各自忙碌，又互相牵挂。",
    "照片会泛黄，但那些一起大笑的日子，永远闪闪发光。",
    "所谓回忆，就是把每一个平凡的当下，活成以后想回去的时光。",
    "友谊不是一段长久的相识，而是一段永久的铭记。",
    "岁月可以偷走容颜，却偷不走我们共度的青春。",
    "有些人注定是生命里的光，哪怕只照亮一瞬，也值得一生珍藏。",
    "真正的朋友会在整个世界都将你遗忘时，仍然记得你。",
    "回忆是时间的礼物，友情是回忆里最温暖的篇章。",
    "不要忘了那些陪你走过风雨的人，他们是岁月里最珍贵的收藏。",
    "友谊像一坛老酒，越久越醇，而那些回忆就是最好的酿造。",
    "最珍贵的照片，不是拍得最好的那张，而是藏着最多回忆的那张。",
    "时光不老，我们不散——送给每一份真挚友谊的约定。",
    "多年以后我们或许走散在人海，但回忆会让彼此一次次重逢。",
]

# ---------------------------------------------------------------------------
# ⑤ 内存保护阈值 & 网格尺寸
# ---------------------------------------------------------------------------
FACE_TIER_MID = 30000      # 人脸总数超过它 → 特征改用 float16 存（省一半内存）
FACE_TIER_HUGE = 150000    # 超过它 → 先弹窗让用户确认，防止内存爆掉
SIM_CHUNK = 4096           # 相似度矩阵分块计算的行数（省内存，结果不变）
GRID_CARD_W = 236          # 人物卡片固定宽度
GRID_CARD_H = 300          # 人物卡片固定高度


# ============================================================================
# 三个“界面小工具函数”
# ============================================================================
def qss_color(c):
    """把 QColor 转成样式表可写的 'rgba(r,g,b,a)' 字符串（保留透明度）。"""
    if isinstance(c, str):      # 如果传进来的本来就是字符串，直接用
        return c
    return f'rgba({c.red()},{c.green()},{c.blue()},{c.alpha()})'


def anim_value(fn, c1, c2, ms=180, widget=None, easing=QEasingCurve.OutCubic,
               on_done=None):
    """★ 动画核心小工具：把某个值从 c1 平滑变到 c2，每变一帧回调一次 fn。

    整个界面“丝滑”都靠它。QVariationAnimation 就像一台机器，自动按 60fps
    从起点一路推到终点；你在 valueChanged 里接住每一个中间值去重画即可。
    参数：
      fn       接收当前值的函数（动画每帧都会调用）
      c1 / c2  起止值（数字、QColor 都行）
      ms       动画时长（毫秒），180 约等于一次很顺的过渡
      easing   加减速曲线：OutCubic=开头快结尾慢，最自然
      on_done  动画播完后再调用的函数（可省略）
    """
    a = QVariantAnimation(widget)     # 动画对象（指定父窗口，防止被误回收）
    a.setStartValue(c1)               # 设定起点
    a.setEndValue(c2)                 # 设定终点
    a.setDuration(ms)                 # 设定时长
    a.setEasingCurve(easing)          # 设定加减速曲线
    a.valueChanged.connect(fn)        # “值变了”→ 通知 fn
    if on_done:
        a.finished.connect(on_done)   # “播完了”→ 通知 on_done
    a.start(QAbstractAnimation.DeleteWhenStopped)  # 开始播；播完自动自我销毁，不留垃圾
    return a


# ============================================================================
# 基础业务函数（读图 / GPU 检测 / 淡入动画）
# ============================================================================
def _fade_in_window(widget, ms=200):
    """子窗口打开时做 Windows 风格淡入：透明度 0 → 1。

    为什么要自己加？我们的子窗口是无边框自绘的（不透明度的由 Qt 合成），
    默认会“唰”地一下直接出现。加上渐显后更接近 Windows 原生的弹出观感。
    windowOpacity 对顶层窗口是原生支持的，可靠且不闪烁。
    """
    try:
        widget.setWindowOpacity(0.0)   # 先把整个窗口设为全透明
        a = QPropertyAnimation(widget, b'windowOpacity', widget)  # 用属性动画改透明度
        a.setStartValue(0.0)
        a.setEndValue(1.0)
        a.setDuration(ms)
        a.setEasingCurve(QEasingCurve.OutCubic)
        a.start(QAbstractAnimation.DeleteWhenStopped)
    except Exception:
        widget.setWindowOpacity(1.0)   # 万一失败，至少保证窗口是可见的


def _nvidia_driver_present():
    """检测本机有没有 NVIDIA 显卡驱动（看 System32 里有没有 nvcuda.dll）。"""
    sr = os.environ.get('SystemRoot') or r'C:\Windows'
    return os.path.exists(os.path.join(sr, 'System32', 'nvcuda.dll'))


def _gpu_runtime_ready():
    """检测 CUDA 运行库是否已就位（放在 onnxruntime 的 capi 目录里）。"""
    capi = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi')
    return all(os.path.exists(os.path.join(capi, n)) for n in GPU_RUNTIME_DLLS)


def collect_images(root):
    """递归遍历文件夹，收集所有支持的图片的完整路径（返回 list[str]）。

    实现就是用 os.walk 一层层往下钻：每到一个目录，把文件名扩展名在
    SUPPORTED 白名单里的拼成完整路径收进来。
    """
    imgs = []                                  # 准备一个空列表装结果
    for dirpath, _, files in os.walk(root):    # dirpath=当前目录 files=里面的文件名
        for f in sorted(files):                # 排序后逐个看（保证顺序稳定）
            # splitext('a.jpg') -> ('a', '.jpg')，取 [1] 就是扩展名，转小写比大小写
            if os.path.splitext(f)[1].lower() in SUPPORTED:
                imgs.append(os.path.join(dirpath, f))   # 拼成完整路径放入列表
    return imgs


def read_image(path):
    """读一张图片（兼容中文路径），返回 OpenCV 的 numpy 数组；读不了返回 None。

    为什么不直接用 cv2.imread(path)？因为它在 Windows 上按 ANSI 编码打开文件，
    遇到中文文件名会失败。所以这里先用 Python 内置 open 读出字节（Python 走
    宽字符 API，中文没问题），再用 cv2.imdecode 从“内存里的字节”解码成图片。
    """
    with open(path, 'rb') as f:      # 'rb'=以二进制读，with 自动负责关文件
        data = f.read()              # 一次性读出整个文件的原始字节
    # frombuffer：把字节“零拷贝”看成一维数组；imdecode：从字节解码成彩色图
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


# ============================================================================
# ★ 人脸聚类算法（进阶内容，先看懂 smart_cluster 的整体思路即可）
#
# 一张脸 = 一个 512 维的特征向量。两个向量越“像”（余弦相似度越高），越可能是
# 同一个人。聚类就是：把互相很像的向量自动归成一组。
# 思路分三步，反复打磨几遍让结果更准：
#   ① DBSCAN 粗分 —— 先把明显同一人的脸拢成几堆；
#   ② 认老大     —— 每堆算一张“平均脸”当代表，让每张脸重新认领最像的老大；
#   ③ 认亲合并   —— 两个老大也太像（同一个人被拆成两堆时）就把两堆合并。
# ============================================================================

def _l2_normalize(emb):
    """把每一行向量都拉成长度=1（单位向量），这样点积=余弦相似度，比大小才公平。"""
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    # np.linalg.norm(emb, axis=1) = 每行求长度；keepdims 保住形状好做除法；+1e-9 防除0


def _group_centroids(emb, labels):
    """给每个群算“平均脸”当群代表(centroid)，返回 (代表dict, 群里成员dict)。

    平均脸 = 群里所有脸向量求平均再归一化。它比单张脸“稳”，能代表这个人。
    """
    groups = {}                                    # {群号: [成员下标...]}
    for i, lab in enumerate(labels):               # 遍历每张脸
        if lab < 0:                                # lab=-1 表示“散脸”，没归属
            continue
        groups.setdefault(int(lab), []).append(i)  # 按群号把人脸下标归类
    centroids = {}
    for lab, idxs in groups.items():               # 对每个群
        mean = emb[idxs].mean(axis=0)              # 群里所有向量求平均
        norm = np.linalg.norm(mean) + 1e-9         # 求长度
        centroids[lab] = mean / norm               # 归一化后当“老大”
    return centroids, groups


def _reassign_to_nearest(emb, labels, centroids, min_score, qualities=None):
    """“认老大”：让每张脸重新挑一个最像的老大；不够像的一律标成散脸(-1)。

    越模糊的脸(qualities 低)门槛自动放宽一点，避免被冤枉成散脸。
    """
    lab_list = list(centroids.keys())
    if not lab_list:                               # 一个群都没有，全当散脸
        return np.full(len(emb), -1, dtype=int)
    C = np.stack([centroids[l] for l in lab_list]) # 所有老大排成矩阵 C
    n_faces = len(emb)
    # ★ 分块算“每张脸 × 每个老大”的相似度：按 SIM_CHUNK 行一批批算，省内存
    sims = np.empty((n_faces, len(lab_list)), dtype=np.float32)
    for s in range(0, n_faces, SIM_CHUNK):
        e = min(s + SIM_CHUNK, n_faces)
        sims[s:e] = emb[s:e] @ C.T                # 矩阵乘法 = 一次算出一批相似度
    best = sims.argmax(axis=1)                    # 每张脸最像哪个老大（下标）
    best_score = sims[np.arange(len(emb)), best]  # 取“最像”的那个分数
    # 默认所有脸同一个门槛 min_score；模糊的脸门槛往下放，最多放宽 0.1
    thresholds = np.full(len(emb), min_score)
    if qualities is not None:
        relax = 0.10
        thresholds = min_score - (1.0 - np.asarray(qualities)) * relax
    # 够得着门槛的归到对应老大；够不着的一律 -1（散脸）
    new_labels = np.where(best_score >= thresholds,
                          np.array([lab_list[b] for b in best]), -1)
    return new_labels.astype(int)


def _merge_close_groups(emb, labels, centroids, merge_score):
    """“认亲合并”：两个群的老大长得太像(≥merge_score)就把两个群合成一个。

    场景：同一个人不同年代/光线的照片第一轮被拆成两堆，靠这步拼回来。
    """
    lab_list = list(centroids.keys())
    if len(lab_list) < 2:                          # 少于两个群没得合
        return labels
    C = np.stack([centroids[l] for l in lab_list])
    g = len(lab_list)
    sim = np.empty((g, g), dtype=np.float32)       # 群×群 相似度（也分块算，省内存）
    for a in range(0, g, SIM_CHUNK):
        a2 = min(a + SIM_CHUNK, g)
        sim[a:a2] = C[a:a2] @ C.T
    new_labels = np.array(labels)
    for a in range(len(lab_list)):                 # 两两比较每个群
        for b in range(a + 1, len(lab_list)):
            if sim[a][b] >= merge_score:           # 太像了
                big, small = lab_list[a], lab_list[b]
                new_labels[new_labels == small] = big   # b 群的人全部并入 a 群
    return new_labels.astype(int)


def _face_quality_score(image, bbox, det_score=1.0):
    """给一张脸打“清楚程度”分(0~1)。模糊脸特征不准，聚类时门槛要放宽。

    三个线索加权：锐利度(Laplacian 方差)占 4 成、脸在照片里的大小占 3 成、
    模型检测置信度占 3 成。
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1); y1 = max(0, y1)               # 把坐标夹到图片内，防止越界
    x2 = min(w, x2); y2 = min(h, y2)
    crop = image[y1:y2, x1:x2]                     # 抠出人脸那块
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # 转灰度才好算边缘
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # 边缘越锐利 → 越清楚
    sharpness = min(1.0, laplacian_var / 500.0)    # 归一化到 0~1
    face_area = (x2 - x1) * (y2 - y1)              # 脸占了多少像素
    img_area = max(1, h * w)
    area_ratio = face_area / img_area
    size_score = min(1.0, area_ratio / 0.01)       # 脸够大就给满分
    return max(0.0, min(1.0, 0.4 * sharpness + 0.3 * size_score + 0.3 * det_score))


def smart_cluster(emb, eps, min_cluster, max_iters=3, qualities=None):
    """聚类总入口：先 DBSCAN 粗分，再反复“认老大 + 认亲”几遍，返回标签数组。

    参数：
      emb          归一化的人脸特征矩阵，形状 (人脸数, 512)
      eps          用户填的阈值：认为“多像才算一家人”，越小越严
      min_cluster  一个组至少要几人才算人物
      qualities    每张脸的清楚分(0~1)，可选
    返回：
      labels       一维数组：labels[i]=第 i 张脸归的群号；-1 = 未分类散脸
    """
    # ① 粗分。因为已归一化，欧氏距离与余弦距离单调等价(d=√(2·dc))，
    #    用 euclidean 能让 DBSCAN 走 BallTree 树搜索，省下 N² 的内存。
    cl = DBSCAN(eps=np.sqrt(2.0 * eps), min_samples=min_cluster,
                metric='euclidean', algorithm='ball_tree')
    labels = cl.fit_predict(emb)                   # 得到第一版分组
    min_score = 1.0 - eps                          # 认老大要“至少像多少”
    merge_score = 1.0 - eps * 1.2                  # 认亲门槛更宽松(愿多合并)
    # ②③ 反复精修几遍：合并 → 重算老大 → 重新认老大
    for _ in range(max_iters):
        centroids, _ = _group_centroids(emb, labels)
        if not centroids:                          # 没群了，直接结束
            break
        labels = _merge_close_groups(emb, labels, centroids, merge_score)
        centroids, _ = _group_centroids(emb, labels)
        labels = _reassign_to_nearest(emb, labels, centroids, min_score, qualities)
    # 收尾：凑不够 min_cluster 张的小群不算人物，全部丢进散脸
    centroids, groups = _group_centroids(emb, labels)
    for lab, idxs in groups.items():
        if len(idxs) < min_cluster:
            labels[np.array(idxs)] = -1
    return labels


# ============================================================================
# 图像 → 界面能用的 QPixmap 工具
# ============================================================================
def _cv_to_qpixmap(img_bgr):
    """把 OpenCV 的 BGR 数组转成 QPixmap（Qt 图片对象，才能塞进界面）。

    注意 OpenCV 存图是 BGR 顺序，Qt 要 RGB，所以先 cvtColor 换一下通道。
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def make_face_thumb(image_path, bbox, size=(176, 176)):
    """按人脸框 bbox 从原图裁出人脸，做成正方形缩略图 QPixmap；失败返回 None。"""
    try:
        img = read_image(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]    # bbox=(左上x,左上y,右下x,右下y)
        pad = int((x2 - x1) * 0.35)                # 人脸框外再扩 35% 留点边
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)   # 夹到图片内
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        crop = img[y1:y2, x1:x2]                   # 切片 = 裁剪，就这么一行
        if crop.size == 0:
            crop = img
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop)                # numpy 数组 → Pillow 图片
        pil.thumbnail((size[0], size[1]), Image.Resampling.LANCZOS)  # 等比缩小(高画质)
        canvas = Image.new('RGB', (size[0], size[1]), (28, 30, 36))  # 空白画布(深底)
        canvas.paste(pil, ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2))
        data = canvas.tobytes('raw', 'RGB')        # Pillow 字节 → QImage → QPixmap
        qimg = QImage(data, size[0], size[1], size[0] * 3, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)
    except Exception:
        return None                                # 任何异常都返回空，不拖垮主流程


# ============================================================================
# 两张“背景图”加载
# ============================================================================
def load_background():
    """加载主窗口背景：读星空图，轻模糊 + 轻压暗。保留主体细节(能看清女孩/星河)。"""
    try:
        img = read_image(BG_IMG_PATH)
        if img is None:
            return None
        # 缩到 1/2 保持细节，模糊很轻（sigma≈9），主要让文字区不刺眼
        small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 9)
        small = cv2.addWeighted(small, 0.85, np.zeros_like(small), 0.15, 0)  # 压暗 15%
        return _cv_to_qpixmap(small)
    except Exception:
        return None


def load_background_center():
    """裁切星空图中心区域（完整的女孩 + 星河），给帮助子窗口当背景。

    女孩是居中的竖构图，所以纵向取 94%(几乎全高、保证头脚完整)；横向取到接近
    正方形(帮助窗口是正方形)，这样铺进窗口时几乎不会被裁。
    """
    try:
        img = read_image(BG_IMG_PATH)
        if img is None:
            return None
        h, w = img.shape[:2]
        ch = int(h * 0.94)                 # 裁切高度：原图 94%
        cw = int(ch * 1.02)                # 宽度取接近高度 → 裁切图近正方形
        cw = min(w, cw)
        y0 = (h - ch) // 2                 # 纵向从正中间开始
        x0 = (w - cw) // 2
        crop = img[y0:y0 + ch, x0:x0 + cw] # 切片裁出中心区域
        crop = cv2.addWeighted(crop, 0.7, np.zeros_like(crop), 0.3, 0)  # 压暗 30% 保文字可读
        return _cv_to_qpixmap(crop)
    except Exception:
        return None


# ============================================================================
# 界面控件区（每个 class 都是一个大名，看完这几个你就能看懂整个界面怎么拼的）
# ============================================================================

# ---------------------------------------------------------------------------
# GlassPanel —— 毛玻璃“容器面板”（工具栏/状态栏/对话框都用它打底）
# ---------------------------------------------------------------------------
class GlassPanel(QFrame):
    """半透明磨砂面板：低透明度底色 + 细边框 + 圆角，鼠标悬停时轻微变亮。

    paintEvent 是它自己“画自己”的入口：用 QPainter 画一个圆角矩形，填充
    半透明颜色，再描一圈细边框。半透明 → 下面的背景(星空图)能透上来 = 玻璃感。
    """

    def __init__(self, parent=None, tint=PANEL_TINT, border=PANEL_BORDER,
                 radius=16, hover_tint=None):
        super().__init__(parent)
        self._tint = tint                 # 平时底色
        self._border = border             # 描边颜色
        self._hover_tint = hover_tint or tint  # 悬停底色（没给就用平时色）
        self._radius = radius             # 圆角半径
        self._t = 0.0                     # 悬停进度 0~1（由动画驱动，见 set_hover）
        self.setAttribute(Qt.WA_TranslucentBackground)  # 允许本控件透明，露出父背景

    def set_hover(self, on, ms=160):
        """开/关悬停：把 _t 从当前值平滑动画到 0 或 1（on=True 变亮）。"""
        anim_value(self._on_t, self._t, 1.0 if on else 0.0, ms=ms, widget=self)

    def _on_t(self, v):
        """动画每帧回调：记下最新进度并重画。"""
        self._t = v
        self.update()                     # 请求重画 → 会触发 paintEvent

    # Qt 事件：鼠标进入/离开控件时自动调用（注意名字固定）
    def enterEvent(self, e):
        self.set_hover(True)              # 进入 → 开始变亮动画
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.set_hover(False)             # 离开 → 变回平时
        super().leaveEvent(e)

    def paintEvent(self, e):
        """自己画自己：圆角矩形，颜色按 _t 在平时色↔悬停色之间取当前值。"""
        p = QPainter(self)                        # 在本控件上“开画”
        p.setRenderHint(QPainter.Antialiasing)    # 打开抗锯齿 → 边缘不毛糙
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)  # 矩形四角缩半像素(描边不糊)
        path = QPainterPath()                     # 路径 = 要画的形状
        path.addRoundedRect(r, self._radius, self._radius)      # 圆角矩形
        base = QColor(self._tint)                 # 取平时色
        hov = QColor(self._hover_tint)            # 取悬停色
        # 按 _t 比例在两者间线性插值（红绿蓝透明度各自算）
        col = QColor(int(base.red() + (hov.red() - base.red()) * self._t),
                     int(base.green() + (hov.green() - base.green()) * self._t),
                     int(base.blue() + (hov.blue() - base.blue()) * self._t),
                     int(base.alpha() + (hov.alpha() - base.alpha()) * self._t))
        p.fillPath(path, col)                     # 填上颜色
        p.setPen(QPen(self._border, 1))           # 选一支 1px 描边笔
        p.drawPath(path)                          # 描出边框

# ---------------------------------------------------------------------------
# Avatar —— 圆形人脸头像（详情窗口里点开原图用）
#   做法：把一张方形缩略图，用“圆形裁剪”剪成圆形显示
# ---------------------------------------------------------------------------
class Avatar(QLabel):
    clicked = Signal()        # 自定义信号：头像被点击时向外广播（谁点谁知道）

    def __init__(self, pixmap, size=176, parent=None):
        super().__init__(parent)
        self._pix = pixmap          # 保存缩略图
        self._sz = size             # 圆形直径
        self.setFixedSize(size, size)
        self.setCursor(QCursor(Qt.PointingHandCursor))  # 悬停显示“小手”

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)  # 缩放图片时平滑
        rect = QRectF(0, 0, self._sz, self._sz)
        path = QPainterPath()
        path.addEllipse(rect)                 # 形状 = 椭圆(=正圆)
        p.setClipPath(path)                   # ★ 把画布“裁”成圆形，之后只画圆内
        if self._pix and not self._pix.isNull():
            p.drawPixmap(rect, self._pix, QRectF(0, 0, self._pix.width(), self._pix.height()))
        else:
            p.fillRect(rect, QColor(40, 42, 48))   # 没图时显示一个深色圆
        p.setClipping(False)                  # 取消裁剪
        p.setPen(QPen(QColor(255, 255, 255, 46), 1))   # 画一圈淡白描边更精致
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(rect)

    def mouseReleaseEvent(self, e):
        """松开左键且在头像内 → 发 clicked 信号（外面用 .connect 接住做动作）。"""
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


# ---------------------------------------------------------------------------
# GlassButton —— 透明无色按钮（开始聚类/浏览/导出等都是它）
#   默认完全透明不抢戏；悬停时浮现一层极淡的白色反馈；主按钮文字加粗
# ---------------------------------------------------------------------------
class GlassButton(QPushButton):
    def __init__(self, text='', primary=False, parent=None):
        super().__init__(text, parent)
        self._primary = primary   # True = 主按钮(如“开始聚类”)：文字纯白加粗
        self._hov = 0.0           # 悬停进度 0~1
        self._press = 0.0         # 按压进度 0~1
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFlat(True)        # 去掉按钮边框底纹
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setMinimumHeight(38)

    # ---- 动画：进入/离开变亮，按下/松开变深，全部平滑过渡 ----
    def enterEvent(self, e):
        anim_value(self._set_hov, self._hov, 1.0, ms=170, widget=self)
        super().enterEvent(e)

    def leaveEvent(self, e):
        anim_value(self._set_hov, self._hov, 0.0, ms=220, widget=self)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        anim_value(self._set_press, self._press, 1.0, ms=120, widget=self,
                   easing=QEasingCurve.OutQuad)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        anim_value(self._set_press, self._press, 0.0, ms=220, widget=self,
                   easing=QEasingCurve.OutBack)   # OutBack 带一点点回弹，手感更好
        super().mouseReleaseEvent(e)

    def _set_hov(self, v):
        self._hov = v
        self.update()

    def _set_press(self, v):
        self._press = v
        self.update()

    # ---- 绘制：平时只有文字；悬停/按下才出现淡淡的圆角白底 ----
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        disabled = not self.isEnabled()   # 禁用态(如聚类中“开始”按钮)
        r = QRectF(self.rect())
        hov = self._hov if not disabled else 0.0
        press = self._press if not disabled else 0.0

        # 悬停时画极淡的白色反馈(透明度很低，几乎看不出色块)，按下略加深
        if hov > 0.01 and not disabled:
            alpha = int(26 * hov + 14 * press)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, alpha))
            p.drawRoundedRect(r, 10, 10)          # 10 = 圆角半径
        elif press > 0.01 and not disabled:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, int(18 * press)))
            p.drawRoundedRect(r, 10, 10)

        # 文字：禁用偏灰；主按钮纯白加粗；普通按钮接近白
        alpha = 120 if disabled else (255 if self._primary else 235)
        f = QFont(FONT_FAMILY, 13)
        f.setBold(self._primary)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255, alpha))
        p.drawText(r, Qt.AlignCenter, self.text())   # 居中画按钮文字


# ---------------------------------------------------------------------------
# GlassProgressBar —— 细圆角进度条（聚类进度），数值变化带平滑动画
# ---------------------------------------------------------------------------
class GlassProgressBar(QProgressBar):
    """圆角进度条：轨道半透明白，已填充部分是蓝色渐变。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(False)    # 不显示“45%”这种文字
        self.setFixedHeight(8)        # 只留 8px 高的一条

    def animate_to(self, value, ms=320):
        """进度值“滑”过去而不是跳过去（QPropertyAnimation 直接改控件的 value）。"""
        a = QPropertyAnimation(self, b'value', self)   # 让 Qt 去动画 value 属性
        a.setStartValue(self.value())
        a.setEndValue(value)
        a.setDuration(ms)
        a.setEasingCurve(QEasingCurve.OutCubic)
        a.start(QAbstractAnimation.DeleteWhenStopped)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # 1) 底槽：整条半透明白的圆角条
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 30))
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
        # 2) 已填充部分：按 value/maximum 占宽，画一条蓝色渐变圆角条
        mx = self.maximum()
        frac = min(1.0, self.value() / mx) if mx > 0 else 0.0
        if frac > 0:
            fill = QRectF(0, 0, w * frac, h)
            grad = QLinearGradient(0, 0, w, 0)      # 从左到右渐变
            grad.setColorAt(0, QColor(ACCENT))
            grad.setColorAt(1, QColor('#5496d8'))
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(fill, h / 2, h / 2)


# ---------------------------------------------------------------------------
# QuoteCard —— 名言卡片（聚类等待时在状态栏下方轮播一句温暖的话）
#   动画要点：
#     · 出现：“模块下放”——高度从 0 展开到自然高度(OutBack 轻微回弹)
#     · 换句：先淡出旧字(fade→0) 再淡入新字(fade→1)
#     · 结束：高度收回到 0 隐藏
# ===========================================================================
class QuoteCard(QWidget):
    """透明卡片，只显示一行名言文字；本身没有背景色，直接浮在星空背景上。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._expand = 0.0          # 0~1：卡片“长出”的高度进度
        self._fade = 1.0            # 0~1：文字淡入淡出
        self._text = ''             # 当前名言
        self._natural_h = 68        # 完全展开后的高度
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedHeight(0)      # 平时高度 0 = 看不见
        self._anim = None           # 正在播的动画对象(需要手动管理，见 _kill_anim)

    def set_text(self, t):
        """换名言内容并重画。"""
        self._text = t
        self.update()

    def _kill_anim(self):
        """停掉上一次动画再开始新的。

        之前的动画设为“播完自动销毁”，所以 self._anim 可能指向一个已被 Qt
        删除的 C++ 对象；直接 .stop() 会报 RuntimeError，这里用 try 兜住。
        """
        if self._anim is not None:
            try:
                self._anim.stop()
            except RuntimeError:
                pass
            self._anim = None

    def show_animated(self):
        """名言“下放”出现：高度 0 → 自然高度。"""
        self._kill_anim()
        self._fade = 1.0
        self.setFixedHeight(0)
        a = QVariantAnimation(self)
        a.setStartValue(0.0)
        a.setEndValue(1.0)
        a.setDuration(460)
        a.setEasingCurve(QEasingCurve.OutBack)   # OutBack：最后带回弹，像放下来
        a.valueChanged.connect(self._set_expand)
        a.start()
        self._anim = a

    def hide_animated(self, on_done=None):
        """收起：高度归零（聚类结束 / 切换时用）。"""
        self._kill_anim()
        a = QVariantAnimation(self)
        a.setStartValue(self._expand)
        a.setEndValue(0.0)
        a.setDuration(320)
        a.setEasingCurve(QEasingCurve.InCubic)
        a.valueChanged.connect(self._set_expand)
        if on_done:
            a.finished.connect(on_done)          # 收完再通知外面收尾
        a.start()
        self._anim = a

    def fade_out(self, on_mid):
        """淡出当前句(透明度降到0)，完成后回调 on_mid 去换字。"""
        self._kill_anim()
        a = QVariantAnimation(self)
        a.setStartValue(self._fade)
        a.setEndValue(0.0)
        a.setDuration(240)
        a.setEasingCurve(QEasingCurve.InCubic)
        a.valueChanged.connect(self._set_fade)
        a.finished.connect(on_mid)
        a.start()
        self._anim = a

    def fade_in(self, on_done=None):
        """淡入新句(透明度回到1)。"""
        self._kill_anim()
        a = QVariantAnimation(self)
        a.setStartValue(0.0)
        a.setEndValue(1.0)
        a.setDuration(240)
        a.setEasingCurve(QEasingCurve.OutCubic)
        a.valueChanged.connect(self._set_fade)
        if on_done:
            a.finished.connect(on_done)
        a.start()
        self._anim = a

    def _set_fade(self, v):
        self._fade = v
        self.update()

    def _set_expand(self, v):
        """动画每帧回调：按进度设置实际高度并重画(展开动画本体)。"""
        self._expand = v
        self.setFixedHeight(int(self._natural_h * v))
        self.update()

    def paintEvent(self, e):
        """画文字。透明度 = 展开进度 × 淡入淡出，两者相乘保证动画连贯。"""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        a = self._expand
        if a <= 0.001:              # 完全收起就不画
            return
        f = self._fade
        r = QRectF(self.rect()).adjusted(20, 0, -20, 0)
        p.setPen(QColor(255, 255, 255, min(255, int(235 * a * f))))  # 白色半透文字
        p.setFont(QFont(FONT_FAMILY, 14))
        p.drawText(r, Qt.AlignCenter, f"❝ {self._text} ❞")


# ---------------------------------------------------------------------------
# FaceCard —— 主界面网格里的一张“人物卡片”
#   自绘(没有子控件)：圆角半透明卡片底 + 圆形人脸头像 + 名字 + 张数 + 命名按钮
#   悬停：上浮 + 变亮 + 出现阴影；按下：回弹动画
# ===========================================================================
class FaceCard(QWidget):
    clicked = Signal(object)     # 点卡片 → 发 clicked(group)，外面用来打开详情
    rename = Signal(object)      # 点“命名”按钮 → 发 rename(group)

    def __init__(self, group, idx, parent=None):
        """造一张卡片。group = 这一组的字典；idx = 它在 groups 列表里的位置。"""
        super().__init__(parent)
        self.group = group                 # 保存这一组数据，后面画图要用
        self.idx = idx
        self.setFixedSize(GRID_CARD_W, GRID_CARD_H)
        self.setCursor(QCursor(Qt.PointingHandCursor))   # 小手光标
        self.setMouseTracking(True)                      # 不开它 hover 动画不灵
        self._t = 0.0                      # 悬停进度(0~1)，动画驱动
        self._press = 0.0                  # 按压进度(0~1)，动画驱动
        # 用这组“第一张脸”做头像缩略图（我们只要一个人脸样例展示在卡片上）
        first = group['items'][0]
        self._thumb = make_face_thumb(first['image'], first['bbox'], size=(150, 150))
        # 卡片主标题：未分类固定叫“未分类”；人物组优先用用户起的名字，没有就“人物N”
        if group['type'] == 'unclassified':
            self._title = '未分类'
        else:
            self._title = group.get('name') or f'人物 {idx + 1}'
        self._sub = f"共 {len(group['items'])} 张"      # 副标题：照片数量
        self._rename_rect = QRect(0, 0, 92, 34)         # “命名”按钮区域(绘制时再更新)

    # ---- 动画 ----
    def set_hover(self, on, ms=180):
        """悬停开关：让 _t 平滑滑到 1(亮/浮起) 或 0(还原)。"""
        anim_value(self._set_t, self._t, 1.0 if on else 0.0, ms=ms, widget=self)

    def _set_t(self, v):
        self._t = v
        self.update()                     # 进度一变就重画

    def press_animate(self):
        """按下动画：先“陷下去”(_press→1)，播完自动回弹(_bounce_back)。"""
        anim_value(self._set_press, 0.0, 1.0, ms=170, widget=self,
                   easing=QEasingCurve.OutQuad, on_done=self._bounce_back)

    def _bounce_back(self):
        """回弹：_press 从 1 弹回 0，OutBack 曲线让卡片“嘣”地回弹一下(像按 iOS 图标)。"""
        anim_value(self._set_press, 1.0, 0.0, ms=280, widget=self,
                   easing=QEasingCurve.OutBack)

    def _set_press(self, v):
        self._press = v
        self.update()

    def enterEvent(self, e):
        self.set_hover(True)              # 鼠标进来 → 上浮
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.set_hover(False)             # 鼠标离开 → 还原
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.press_animate()          # 左键按下就播按压动画
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        """松开左键时判断点在哪：点命名按钮 → rename；点其它 → clicked。"""
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            if self.group['type'] == 'person' and \
               self._rename_rect.contains(e.position().toPoint()):
                self.rename.emit(self.group)      # 发“要命名”信号
            else:
                self.clicked.emit(self.group)     # 发“点卡片”信号(打开详情)
        super().mouseReleaseEvent(e)

    # ---- 绘制（一张卡片完全靠这里一笔笔画出来） ----
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        w = self.width()
        lift = int(6 * self._t)          # 悬停时整体上浮 6px
        press = int(3 * self._press)     # 按下时下压 3px
        dy = lift - press                # 综合偏移量，动画都体现在这

        # 悬停阴影（画在卡片下方一点的位置，营造“浮起来”的立体感）
        if self._t > 0.01:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, int(90 * self._t)))
            p.drawRoundedRect(QRectF(4, 8, w - 8, self.height() - 6), 20, 20)

        # 卡片主体：半透明白的圆角矩形(颜色随悬停 _t 在 CARD_TINT↔CARD_HOVER 之间变)
        body = QRectF(0, dy, w, self.height() - 6)
        path = QPainterPath()
        path.addRoundedRect(body, CARD_RADIUS, CARD_RADIUS)
        base = QColor(CARD_TINT)
        hov = QColor(CARD_HOVER)
        col = QColor(int(base.red() + (hov.red() - base.red()) * self._t),
                     int(base.green() + (hov.green() - base.green()) * self._t),
                     int(base.blue() + (hov.blue() - base.blue()) * self._t),
                     int(base.alpha() + (hov.alpha() - base.alpha()) * self._t))
        p.fillPath(path, col)                          # 填色
        p.setPen(QPen(QColor(255, 255, 255, int(36 + 30 * self._t)), 1))  # 白描边
        p.drawPath(path)

        # 圆形头像：先“裁剪成正圆”再把缩略图画进圆里
        av_sz = 140
        av_x = (w - av_sz) / 2
        av_y = 22 + dy
        av_rect = QRectF(av_x, av_y, av_sz, av_sz)
        av_path = QPainterPath()
        av_path.addEllipse(av_rect)
        p.save()
        p.setClipPath(av_path)
        if self._thumb and not self._thumb.isNull():
            p.drawPixmap(av_rect, self._thumb,
                         QRectF(0, 0, self._thumb.width(), self._thumb.height()))
        else:
            p.fillRect(av_rect, QColor(42, 44, 52))    # 没缩略图就画个深色圆
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(av_rect)                         # 圆外再勾一圈边

        # 名字（太长就省略号截断）
        f = QFont(FONT_FAMILY, 14)
        f.setBold(True)
        p.setFont(f)
        name_rect = QRectF(16, av_y + av_sz + 12, w - 32, 24)
        title = self._title
        if QFontMetrics(f).horizontalAdvance(title) > w - 32:   # 量字宽超没超
            title = QFontMetrics(f).elidedText(title, Qt.ElideRight, int(w - 32))
        p.setPen(TEXT)
        p.drawText(name_rect, Qt.AlignCenter, title)

        # 张数（次要文字）
        p.setPen(TEXT_SUB)
        p.setFont(QFont(FONT_FAMILY, 11))
        p.drawText(QRectF(16, av_y + av_sz + 38, w - 32, 20), Qt.AlignCenter, self._sub)

        # 命名按钮（只在人物组显示；未分类不允许命名）
        if self.group['type'] == 'person':
            bx = (w - 92) / 2
            by = self.height() - 46 + dy
            self._rename_rect = QRect(int(bx), int(by), 92, 34)  # 记录位置供点击判断
            bpath = QPainterPath()
            bpath.addRoundedRect(QRectF(bx, by, 92, 34), 10, 10)
            p.setPen(Qt.NoPen)
            p.fillPath(bpath, QColor(ACCENT))          # 蓝色胶囊
            p.drawPath(bpath)
            p.setPen(QColor('#ffffff'))
            p.setFont(QFont(FONT_FAMILY, 12))
            p.drawText(QRectF(bx, by, 92, 34), Qt.AlignCenter, '命名  ✏')


# ============================================================================
# MainWindow —— 整个程序的主窗口（最重要的一个类）
#   职责：搭界面 + 响应用户操作 + 起后台聚类线程 + 把结果渲染成卡片
# ============================================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('本地人脸聚类相册')
        self.resize(1240, 860)              # 初始窗口大小(像素)
        self.setMinimumSize(980, 640)       # 最小尺寸，防止被拖太小
        # 使用系统原生标题栏/边框（右上角三个按钮即系统原生）

        # ---- 业务状态（记录“当前程序处于什么情况”）----
        self.app = None              # 人脸识别模型对象。None=还没加载(懒加载)
        self.groups = []             # 聚类结果。元素形如 {'type':..., 'items':[...], 'name':...}
        self.use_gpu = False         # 是否启用 GPU
        self._busy = False           # 是不是正在聚类(防重复点“开始”)
        self._thumb_refs = []        # 保存缩略图引用，防被 Python 回收导致不显示
        self.msg_queue = queue.Queue()   # ★ 后台线程 → 界面 的消息队列
        self._confirm_event = threading.Event()  # “海量人脸确认”用的线程握手
        self._confirm_ok = False     # 用户是否点了“确定继续”
        self._quote_idx = 0          # 名言轮播下标
        self._quote_after = None     # 名言切换的定时器
        self._quote_showing = False  # 名言卡片是否正在显示
        self.export_mode = 'copy'    # 导出方式：'copy'=复制 / 'shortcut'=快捷方式
        self._last_cols = None       # 上次算出的列数(缓存，避免重复布局)

        # ---- 检测本机有没有 NVIDIA 显卡/CUDA（决定界面怎么提示）----
        self._nvidia_driver = _nvidia_driver_present()
        self._gpu_runtime = _gpu_runtime_ready()
        self._cuda_available = self._nvidia_driver and self._gpu_runtime
        self.use_gpu = self._cuda_available

        # 背景图（加载一次，之后缩放复用）
        self._bg_orig = load_background()   # 原始(轻模糊)背景
        self._bg_scaled = None              # 缩放到当前窗口尺寸的缓存

        self._build_ui()                    # 把整个界面搭起来(见下面)

        # 定时器：每 120ms 去队列里取一次后台消息(见 _poll_queue)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_queue)
        self._poll_timer.start(120)

    # ============ UI 构建 ============
    def _build_ui(self):
        """搭建整个界面。纵向从上到下分四块(见 root 的 addWidget 顺序)：

            ① header   顶部标题栏（标题+标语+帮助按钮）
            ② top      工具面板（文件夹选择 / 参数 / 导出）
            ③ status   状态面板（状态文字+进度条+名言卡）
            ④ scroll   可滚动的“人物卡片”大网格

        记住 Qt 排版的套路：外层用 QVBoxLayout 竖排，内部小块用 QHBoxLayout 横排；
        addWidget 从上往下/从左往右依次放；布局会自动处理缩放与对齐。
        """
        # ---- 最外层：一个竖排布局包住整窗 ----
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 20)   # 四周边距，让内容不贴边
        root.setSpacing(14)                        # 上下两块的间距

        # ---------------------------------------------------------------
        # ① 顶部标题栏：鸿蒙黑体大标题 + 行楷标语 + 右侧“?”帮助按钮
        # ---------------------------------------------------------------
        header = QWidget()
        header.setAttribute(Qt.WA_TranslucentBackground)   # 透明，露出下面星空背景
        hlay = QHBoxLayout(header)                 # header 内横向排列
        hlay.setContentsMargins(4, 0, 12, 0)
        tcol = QVBoxLayout()                       # 标题竖着叠两行：大标题在上标语在下
        tcol.setSpacing(2)
        title = QLabel('人脸聚类相册')             # 主标题
        title.setFont(QFont('HarmonyOS Sans SC', 26, QFont.Bold))  # 鸿蒙黑体、加粗
        title.setStyleSheet('color:white;background:transparent;')
        slogan = QLabel('时光会走远，但每一帧温暖的回忆，都替你记得。')  # 标语
        slogan.setFont(QFont('STXingkai', 17))     # 华文行楷字体
        slogan.setStyleSheet('color:rgba(255,255,255,175);background:transparent;')
        tcol.addWidget(title)
        tcol.addWidget(slogan)
        hlay.addLayout(tcol)                       # 标题组放最左
        hlay.addStretch(1)                         # 中间空出弹性空隙，把按钮推到最右
        self.btn_home_help = GlassButton('?')      # 帮助按钮(透明样式)
        self.btn_home_help.setFixedSize(40, 40)
        self.btn_home_help.clicked.connect(self._show_help)   # 点了→弹帮助窗口
        hlay.addWidget(self.btn_home_help)
        root.addWidget(header)

        # ---------------------------------------------------------------
        # ② 工具面板（GlassPanel=毛玻璃容器）：里面上下两行控件
        # ---------------------------------------------------------------
        top = GlassPanel(tint=QColor(28, 34, 44, 95), radius=14)
        root.addWidget(top)
        top_lay = QVBoxLayout(top)
        top_lay.setContentsMargins(22, 18, 22, 14)
        top_lay.setSpacing(12)

        # 第 1 行(横向)：照片文件夹输入框 + 浏览 + 开始聚类
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        lbl_dir = QLabel('照片文件夹')             # 左边一个小标签
        lbl_dir.setStyleSheet('color:white;font-size:13px;background:transparent;')
        row1.addWidget(lbl_dir)
        self.dir_edit = QLineEdit()                # 路径输入框
        self.dir_edit.setPlaceholderText('选择存放照片的文件夹…')  # 没内容时的灰色提示
        # setStyleSheet：给输入框化妆(半透明底、圆角；获得焦点时边框变主色)
        self.dir_edit.setStyleSheet(
            'QLineEdit{background:rgba(255,255,255,16);border:1px solid rgba(255,255,255,36);'
            'border-radius:10px;padding:8px 12px;color:white;font-size:13px;}'
            'QLineEdit:focus{border:1px solid ' + ACCENT + ';}')
        row1.addWidget(self.dir_edit, 1)           # 权重 1 = 让它把剩余宽度都吃掉
        self.btn_browse = GlassButton('浏览')      # 打开文件夹选择框
        self.btn_browse.clicked.connect(self._choose_dir)
        row1.addWidget(self.btn_browse)
        row1.addSpacing(12)
        self.btn_start = GlassButton('开始聚类', primary=True)  # 主按钮
        self.btn_start.clicked.connect(self._start_cluster)
        row1.addWidget(self.btn_start)
        top_lay.addLayout(row1)

        # 第 2 行(横向)：参数 + 分隔线 + 导出 + 提示
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        def caption(t):                            # 小工具：造一个“灰色小标签”
            lb = QLabel(t)
            lb.setStyleSheet('color:rgba(245,247,250,150);font-size:12px;'
                             'background:transparent;')
            return lb

        # 参数区：eps 阈值 + 最少张数 两个小输入框
        line_css = ('QLineEdit{background:rgba(255,255,255,16);'
                    'border:1px solid rgba(255,255,255,36);'
                    'border-radius:8px;padding:6px 8px;color:white;font-size:13px;}'
                    'QLineEdit:focus{border:1px solid ' + ACCENT + ';}')
        row2.addWidget(caption('参数'))
        row2.addWidget(caption('eps 阈值'))
        self.eps_edit = QLineEdit('0.43')          # 聚类敏感度，默认 0.43
        self.eps_edit.setFixedWidth(58)
        self.eps_edit.setStyleSheet(line_css)
        row2.addWidget(self.eps_edit)
        row2.addWidget(caption('最少张数'))
        self.min_edit = QLineEdit('2')             # 凑够几张才算一个人物
        self.min_edit.setFixedWidth(46)
        self.min_edit.setStyleSheet(line_css)
        row2.addWidget(self.min_edit)

        # GPU 勾选框：能加速就默认勾上；没显卡/缺运行库则显示不同提示
        self.gpu_check = QCheckBox('GPU 加速')
        self.gpu_check.setChecked(self.use_gpu)
        self.gpu_check.setStyleSheet(
            'QCheckBox{color:white;font-size:13px;background:transparent;spacing:7px;}'
            'QCheckBox::indicator{width:18px;height:18px;border-radius:5px;'
            'border:1px solid rgba(255,255,255,70);background:rgba(255,255,255,12);}'
            'QCheckBox::indicator:checked{background:' + ACCENT + ';border:1px solid ' + ACCENT + ';}')
        self.gpu_check.toggled.connect(self._gpu_toggled)
        if not self._nvidia_driver:
            self.gpu_check.setText('GPU 加速（本机无 NVIDIA 显卡）')
            self.gpu_check.setEnabled(False)       # 没显卡就禁止勾选
        elif not self._gpu_runtime:
            self.gpu_check.setText('GPU 加速（需联网下载运行库）')
        row2.addWidget(self.gpu_check)

        sep = QFrame()                             # 一条 1px 竖分隔线
        sep.setFixedWidth(1)
        sep.setStyleSheet('background:rgba(255,255,255,30);')
        row2.addWidget(sep)
        row2.addSpacing(6)

        # 导出区：复制 / 快捷方式 单选 + 导出按钮
        row2.addWidget(caption('导出'))
        self.radio_copy = QRadioButton('复制')     # 复制：原图不动，占双份空间
        self.radio_lnk = QRadioButton('快捷方式')  # 快捷方式：只建 .lnk 指向原图，省内存
        self.radio_copy.setChecked(True)
        radio_css = ('QRadioButton{color:white;font-size:13px;background:transparent;spacing:6px;}'
                     'QRadioButton::indicator{width:16px;height:16px;border-radius:8px;'
                     'border:1px solid rgba(255,255,255,70);background:rgba(255,255,255,12);}'
                     'QRadioButton::indicator:checked{background:' + ACCENT +
                     ';border:2px solid rgba(255,255,255,220);}')
        self.radio_copy.setStyleSheet(radio_css)
        self.radio_lnk.setStyleSheet(radio_css)
        # 两个单选要“互斥”：都绑到 export_mode 上，勾哪个就把模式设成哪个
        self.radio_copy.toggled.connect(
            lambda chk: setattr(self, 'export_mode', 'copy' if chk else 'shortcut'))
        self.radio_lnk.toggled.connect(
            lambda chk: setattr(self, 'export_mode', 'shortcut' if chk else 'copy'))
        row2.addWidget(self.radio_copy)
        row2.addWidget(self.radio_lnk)

        self.btn_export = GlassButton('导出相册')  # 按人物把照片写/链接到硬盘
        self.btn_export.clicked.connect(self._export_album)
        row2.addWidget(self.btn_export)
        row2.addWidget(caption('未命名的人物将导出为「人物N」'))
        row2.addStretch(1)                         # 右侧留弹性空隙，防止控件拉太开
        top_lay.addLayout(row2)

        tip = QLabel('提示：点击人物卡片可查看该人物的全部照片；点击照片可用系统默认程序打开原图。')
        tip.setStyleSheet('color:rgba(245,247,250,120);font-size:12px;background:transparent;')
        top_lay.addWidget(tip)

        # ---------------------------------------------------------------
        # ③ 状态面板：一行状态文字 + 圆角进度条 + 名言卡片(平时收起)
        # ---------------------------------------------------------------
        status = GlassPanel(tint=QColor(28, 34, 44, 95), radius=14)
        root.addWidget(status)
        st_lay = QVBoxLayout(status)
        st_lay.setContentsMargins(22, 12, 22, 14)
        st_lay.setSpacing(10)
        self.status_label = QLabel('就绪 —— 请选择照片文件夹')   # 底部状态文字
        self.status_label.setStyleSheet('color:white;font-size:13px;background:transparent;')
        st_lay.addWidget(self.status_label)
        self.progress = GlassProgressBar()         # 聚类进度条
        st_lay.addWidget(self.progress)
        self.quote_card = QuoteCard()              # 名言卡片(高度0=隐藏，播放时下放展开)
        st_lay.addWidget(self.quote_card)

        # ---------------------------------------------------------------
        # ④ 主区域：QScrollArea 可滚动区域，里面放人物卡片网格
        #    （QScrollArea = 一个能上下滚的“框”，内容多了出现滚动条）
        # ---------------------------------------------------------------
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)       # 内容宽度跟随区域自动伸缩
        self.scroll.setFrameShape(QFrame.NoFrame)  # 去掉边框
        # QSS 给滚动条化妆：细条、半透明滑块(Windows 默认的灰色滚动条很突兀)
        self.scroll.setStyleSheet('QScrollArea{background:transparent;border:none;}'
                                  'QScrollBar:vertical{background:transparent;width:8px;margin:0;}'
                                  'QScrollBar::handle:vertical{background:rgba(255,255,255,50);'
                                  'border-radius:4px;min-height:30px;}'
                                  'QScrollBar::add-line:vertical,'
                                  'QScrollBar::sub-line:vertical{height:0;}')
        self.scroll.viewport().setAutoFillBackground(False)   # 视口不自动填底色→保持透明
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 不要横向滚动条
        root.addWidget(self.scroll, 1)             # 权重 1：占满剩余所有高度

        # 网格承载控件 + 网格布局（网格布局按 行/列 摆放卡片，像 Excel 表格）
        self.grid_host = QWidget()
        self.grid_host.setStyleSheet('background:transparent;')
        self.grid_host.setAttribute(Qt.WA_TranslucentBackground)
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(16)                   # 卡片间距
        self.grid.setContentsMargins(4, 4, 4, 8)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignHCenter)   # 靠上、水平居中
        self.scroll.setWidget(self.grid_host)      # 把这个网格塞进滚动区域

        # 空状态提示：还没聚类时，网格中央显示两行引导文字(见 _show_empty 控制显隐)
        self.empty_host = QWidget(self.grid_host)
        self.empty_host.setAttribute(Qt.WA_TranslucentBackground)
        ev = QVBoxLayout(self.empty_host)
        ev.setAlignment(Qt.AlignCenter)
        e1 = QLabel('还没有聚类结果')
        e1.setStyleSheet('color:rgba(245,247,250,150);font-size:26px;font-weight:700;'
                         'background:transparent;')
        e1.setAlignment(Qt.AlignCenter)
        e2 = QLabel('选择照片文件夹，点击「开始聚类」\n程序会自动识别照片里的人脸，并按人物智能分组')
        e2.setStyleSheet('color:rgba(245,247,250,90);font-size:15px;background:transparent;')
        e2.setAlignment(Qt.AlignCenter)
        ev.addWidget(e1)
        ev.addWidget(e2)
        self._show_empty()                         # 初始显示空状态

    # ============ 背景 & 布局 ============
    def _update_bg(self):
        if self._bg_orig is None:
            self._bg_scaled = None
            return
        s = self.size()
        if s.width() <= 1 or s.height() <= 1:
            return
        p = self._bg_orig.scaled(s, Qt.KeepAspectRatioByExpanding,
                                 Qt.SmoothTransformation)
        x = (p.width() - s.width()) // 2
        y = (p.height() - s.height()) // 2
        self._bg_scaled = p.copy(max(0, x), max(0, y), s.width(), s.height())

    def _grid_cols(self):
        w = self.width() - 60
        if w < 220:
            w = 1100
        return max(1, w // (GRID_CARD_W + 24))

    def _show_empty(self):
        if self.empty_host.parent() is not self.grid_host:
            self.empty_host.setParent(self.grid_host)
        self.empty_host.setGeometry(self.grid_host.rect())
        self.empty_host.raise_()

    def _hide_empty(self):
        if self.empty_host.parent() is self.grid_host:
            self.empty_host.setParent(None)

    def _relayout_empty(self):
        if self.empty_host.parent() is self.grid_host:
            self.empty_host.setGeometry(self.grid_host.rect())

    def showEvent(self, e):
        """窗口第一次显示时：更新背景并摆放空状态。"""
        super().showEvent(e)
        self._update_bg()
        self._relayout_empty()

    def resizeEvent(self, e):
        """窗口尺寸一变：重新缩放背景铺满 + 让空状态提示保持居中。"""
        super().resizeEvent(e)
        self._update_bg()
        self._relayout_empty()

    def paintEvent(self, e):
        """主窗口“画背景”：把背景图铺满整个窗口；没背景图就画个深色渐变兜底。"""
        p = QPainter(self)
        if self._bg_scaled:
            p.drawPixmap(self.rect(), self._bg_scaled)
        else:
            grad = QLinearGradient(0, 0, 0, self.height())
            grad.setColorAt(0, QColor(52, 60, 74))
            grad.setColorAt(1, QColor(26, 30, 38))
            p.fillRect(self.rect(), QBrush(grad))

    # ============ 交互 ============
    def _choose_dir(self):
        """弹系统文件夹选择框，把选中的路径写进输入框。"""
        d = QFileDialog.getExistingDirectory(self, '选择照片文件夹')
        if d:                               # 点“取消”返回空串，不处理
            self.dir_edit.setText(d)

    def _gpu_toggled(self, on):
        """GPU 勾选框变化时，记下用户的选择。"""
        self.use_gpu = on

    def _show_help(self):
        """打开帮助子窗口(阻塞到它关掉)。"""
        dlg = HelpDialog(self)
        dlg.exec()

    # ============ 聚类 ============
    def _start_cluster(self):
        """点“开始聚类”：校验输入 → 清旧结果 → 起后台线程(真正的活在线程里干)。"""
        d = self.dir_edit.text().strip()            # 取输入框路径并去掉首尾空格
        if not d or not os.path.isdir(d):           # 空 or 不是文件夹
            show_glass_info(self, '提示', '请先选择有效的照片文件夹')
            return
        if self._busy:                              # 上一轮没跑完就拒绝再点
            show_glass_info(self, '提示', '上一轮聚类还在进行中，请等待它完成')
            return
        try:                                        # 校验参数能不能转成数字
            eps = float(self.eps_edit.text())       # 用户可能填错 → 抛异常被接住
            mc = int(self.min_edit.text())
        except Exception:
            show_glass_info(self, '提示', 'eps 或最少张数格式不正确')
            return

        # ---- 准备好“新的一轮”的界面状态 ----
        self._clear_grid()                          # 清掉上一轮的卡片
        self.groups = []                            # 结果清空
        self.progress.setValue(0)                   # 进度条归零
        self._busy = True                           # 标记忙碌(防止重复点)
        self.btn_start.setEnabled(False)            # 禁用“开始”按钮
        self.status_label.setText('正在加载模型...')

        QTimer.singleShot(600, self._show_quote)    # 0.6 秒后开始播名言(加载模型抢镜完)

        # ★ 启动后台线程。target=线程里要运行的函数；args=传进去的参数。
        #   daemon=True：主程序退出时线程自动结束，不会卡住程序关闭。
        threading.Thread(target=self._cluster_worker, args=(d, eps, mc),
                         daemon=True).start()
        # .start() 一开线程，本函数立刻返回 → 界面继续流畅响应；
        # 耗时的活都在新线程里做，做完通过消息队列通知回来(见 _poll_queue)。

    def _clear_grid(self):
        """清空网格里的所有旧卡片(删控件)。"""
        while self.grid.count():                    # 只要网格里还有东西
            it = self.grid.takeAt(0)                # 从布局里取出第一项
            w = it.widget()
            if w:
                w.deleteLater()                     # 稍后删除(交给事件循环，安全)
        self._thumb_refs.clear()                    # 缩略图引用一并释放
        self._hide_empty()

    def _cluster_finished(self):
        """一轮聚类结束(成功/出错/取消)时调用：恢复按钮、收起名言卡。"""
        self._busy = False
        self.btn_start.setEnabled(True)             # 恢复“开始”可点
        if self._quote_after is not None:           # 取消名言轮播定时器
            self._quote_after.stop()
            self._quote_after = None
        self._quote_showing = False
        self.quote_card.hide_animated(lambda: (self.quote_card.set_text(''),
                                               self.quote_card.update()))  # 名言卡收起来

    # ---- 名言：模块下放动画 + 4 秒轮播 ----
    def _show_quote(self):
        """开始播名言：先“下放”展开卡片，再定时 4 秒换下一句。"""
        if not self._busy:                          # 若已结束就不再播
            return
        self._quote_showing = True
        self._set_quote_text()                      # 取一句填进去
        self.quote_card.show_animated()             # 高度从 0 展开(下放动画)
        self._schedule_next_quote()                 # 安排 4 秒后换句

    def _set_quote_text(self):
        """从 QUOTES 里按顺序取一句(取完从头再来)。"""
        q = QUOTES[self._quote_idx % len(QUOTES)]   # % 取余实现“循环播放”
        self._quote_idx += 1
        self.quote_card.set_text(q)

    def _schedule_next_quote(self):
        """设一个 4 秒后触发 _next_quote 的单次定时器。"""
        if not self._busy:
            return
        if self._quote_after is not None:           # 有旧的先停掉，避免叠出多个
            self._quote_after.stop()
        t = QTimer(self)
        t.setSingleShot(True)                       # 只响一次
        t.timeout.connect(self._next_quote)
        t.start(4000)                               # 4000 毫秒 = 4 秒
        self._quote_after = t

    def _next_quote(self):
        """4 秒到：淡出旧句(播完会走到 _fade_in_next 换新句)。"""
        self._quote_after = None
        if not self._busy:
            return
        self.quote_card.fade_out(on_mid=self._fade_in_next)   # 淡出完自动去换字

    def _fade_in_next(self):
        """承接上一步：换上新名言，再淡入。"""
        if not self._busy:
            return
        self._set_quote_text()                      # 换上下一句
        self.quote_card.fade_in(on_done=self._schedule_next_quote)  # 淡入完再计 4 秒

    # ============ 联网下载 ============
    def _stream_download(self, url, dest, label):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get('Content-Length') or 0)
            done = 0
            last_pct = -1
            with open(dest, 'wb') as f:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = min(100, int(done * 100 / total))
                        if pct != last_pct:
                            last_pct = pct
                            self.msg_queue.put(('progress', (done, total,
                                                             f"{label} {pct}%")))

    def _ensure_model_ready(self):
        model_dir = os.path.join(MODEL_ROOT, 'models', 'buffalo_l')
        try:
            if os.path.isdir(model_dir) and any(f.endswith('.onnx')
                                                for f in os.listdir(model_dir)):
                return True
            os.makedirs(model_dir, exist_ok=True)
            zip_path = os.path.join(os.path.dirname(model_dir), 'buffalo_l.zip')
            self.msg_queue.put(('status', '首次使用，正在联网下载人脸识别模型（约 300MB）...'))
            self._stream_download(MODEL_DL_URL, zip_path, '正在下载人脸识别模型')
            with zipfile.ZipFile(zip_path) as zf:
                if any(n.startswith('buffalo_l/') for n in zf.namelist()):
                    zf.extractall(os.path.join(MODEL_ROOT, 'models'))
                else:
                    zf.extractall(model_dir)
            try:
                os.remove(zip_path)
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _gpu_wheel_url(self):
        if GPU_RUNTIME_URL:
            return GPU_RUNTIME_URL
        ver = onnxruntime.__version__
        py_tag = 'cp%d%d' % sys.version_info[:2]
        api = 'https://pypi.org/pypi/onnxruntime-gpu/%s/json' % ver
        with urllib.request.urlopen(api, timeout=30) as resp:
            data = json.load(resp)
        for f in data['urls']:
            fn = f['filename']
            if fn.endswith('win_amd64.whl') and ('-%s-' % py_tag) in fn:
                return f['url']
        raise RuntimeError('未找到 onnxruntime-gpu %s 的 Windows 安装包' % ver)

    def _pypi_wheel_url(self, pkg, major=None):
        api = 'https://pypi.org/pypi/%s/json' % pkg
        with urllib.request.urlopen(api, timeout=30) as resp:
            data = json.load(resp)

        def ver_key(v):
            return [int(x) for x in re.split(r'[.\-+]', v) if x.isdigit()] or [0]

        for v in sorted(data['releases'].keys(), key=ver_key, reverse=True):
            if major is not None and not v.split('.')[0] == str(major):
                continue
            for f in data['releases'][v]:
                if f['filename'].endswith('win_amd64.whl'):
                    return f['url']
        raise RuntimeError('找不到 %s 的 Windows 安装包' % pkg)

    @staticmethod
    def _extract_dlls(wheel_path, dest, need):
        with zipfile.ZipFile(wheel_path) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base.lower().endswith('.dll') and base.lower().startswith(need):
                    with zf.open(name) as src, open(os.path.join(dest, base), 'wb') as dst:
                        shutil.copyfileobj(src, dst)

    def _try_enable_gpu_on_demand(self):
        try:
            self.msg_queue.put(('status', '正在联网下载 GPU 加速组件（约 1.2GB，只下一次）...'))
            capi = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi')
            if not os.path.isdir(capi) or not os.access(capi, os.W_OK):
                return False
            tmp = os.path.join(capi, '_gpu_tmp')
            os.makedirs(tmp, exist_ok=True)
            if GPU_RUNTIME_URL:
                zip_path = os.path.join(tmp, 'gpu_runtime.zip')
                self._stream_download(GPU_RUNTIME_URL, zip_path, '正在下载 GPU 加速组件')
                self._extract_dlls(zip_path, tmp, ('cublas', 'cudnn', 'cudart', 'cufft',
                                                   'onnxruntime_providers_cuda',
                                                   'onnxruntime_providers_shared'))
            else:
                wheel = os.path.join(tmp, 'ort_gpu.whl')
                self._stream_download(self._gpu_wheel_url(), wheel, '正在下载 GPU 加速组件')
                self._extract_dlls(wheel, tmp, ('onnxruntime_providers_cuda',
                                                'onnxruntime_providers_shared'))
                os.remove(wheel)
                for pkg, major in (('nvidia-cublas-cu12', 12), ('nvidia-cuda-runtime-cu12', 12),
                                   ('nvidia-cudnn-cu12', 9), ('nvidia-cufft-cu12', None)):
                    w = os.path.join(tmp, pkg + '.whl')
                    self._stream_download(self._pypi_wheel_url(pkg, major), w,
                                          '正在下载 GPU 加速组件')
                    self._extract_dlls(w, tmp, ('cublas', 'cudnn', 'cudart', 'cufft'))
                    os.remove(w)
            for f in os.listdir(tmp):
                if f.lower().endswith('.dll'):
                    shutil.move(os.path.join(tmp, f), os.path.join(capi, f))
            shutil.rmtree(tmp, ignore_errors=True)
            self._gpu_runtime = _gpu_runtime_ready()
            self._cuda_available = self._nvidia_driver and self._gpu_runtime
            return self._cuda_available
        except Exception:
            return False

    # ============ 后台聚类 ============
    def _cluster_worker(self, d, eps, min_cluster):
        """★ 后台线程里跑完整聚类流程（用户点“开始聚类”后由新线程调用这里）。

        铁律：这个函数运行在【后台线程】，绝不能直接碰界面控件（Qt 只允许主线程
        改界面）。想通知界面，就把消息塞进 self.msg_queue，界面会定时来取。
        消息就是一个小元组，第 0 个元素是类型：
          ('status', 文字)  /  ('progress',(当前,总数,文字))  /  ('confirm',...)
          ('cancelled',文字) /  ('error', 文字)  /  ('done', groups)
        """
        try:
            # ---- ① 收集图片 ----
            images = collect_images(d)              # 递归找文件夹里所有图片
            if not images:
                self.msg_queue.put(('error', '该文件夹下没有找到任何图片'))
                return
            self.msg_queue.put(('status', f"共 {len(images)} 张图片，正在加载人脸模型..."))

            # ---- ② 懒加载人脸模型(只用第一次，之后复用 self.app) ----
            if self.app is None:
                if not self._ensure_model_ready():  # 模型没下载就先去下载
                    self.msg_queue.put(('error',
                        '人脸识别模型没有就绪。首次使用需要联网下载模型（约 300MB），\n'
                        '当前可能没有网络或下载失败，请检查网络后重新开始聚类。'))
                    return
                # 决定是否用 GPU：勾了且 CUDA 齐全 → GPU；缺运行库 → 试着联网装；
                # 都不行 → CPU(功能不变)
                want_gpu = False
                if self.use_gpu:
                    if self._cuda_available:
                        want_gpu = True
                    elif self._try_enable_gpu_on_demand():
                        want_gpu = True
                    else:
                        self.msg_queue.put(('status',
                                            '未启用 GPU：本机无 NVIDIA 显卡或无法联网下载加速组件，已使用 CPU'))
                providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if want_gpu
                             else ['CPUExecutionProvider'])
                self.msg_queue.put(('status',
                                    '正在加载人脸模型（' + ('GPU 加速' if want_gpu else 'CPU')
                                    + '）...'))
                try:
                    self.app = FaceAnalysis(name='buffalo_l', root=MODEL_ROOT,
                                            providers=providers)
                    self.app.prepare(ctx_id=0 if want_gpu else -1, det_size=(640, 640))
                except Exception:                   # 加载失败(驱动问题等)自动退回 CPU
                    self.app = FaceAnalysis(name='buffalo_l', root=MODEL_ROOT,
                                            providers=['CPUExecutionProvider'])
                    self.app.prepare(ctx_id=-1, det_size=(640, 640))
                    self.msg_queue.put(('status', '显卡不可用，已自动改用 CPU 运行'))

            # ---- ③ 逐张图片做人脸检测 + 特征提取 ----
            embeddings = []     # 每个人脸的 512 维特征向量
            qualities = []      # 与上面一一对应：每张脸的“清楚分”
            metas = []          # 与上面一一对应：这张脸来自哪张图、人脸框在哪
            total = len(images)
            use_half = False    # 人脸太多时特征改存 float16(省一半内存)
            for i, p in enumerate(images):
                img = read_image(p)                 # 读图(兼容中文路径)
                if img is None:
                    continue                        # 读不了就跳过
                faces = self.app.get(img)           # ★ 检测这一张图里的所有人脸
                # 跨过“中等量级”门槛 → 把已存特征转成半精度，之后也都用半精度
                if not use_half and len(embeddings) + len(faces) > FACE_TIER_MID:
                    use_half = True
                    embeddings = [e.astype(np.float16) for e in embeddings]
                for face in faces:                  # 一张合照可能有多张脸，逐个处理
                    embeddings.append(face.embedding.astype(np.float16 if use_half
                                                            else np.float32))
                    metas.append({'image': p, 'bbox': face.bbox.tolist()})
                    qualities.append(_face_quality_score(img, face.bbox,
                                                         float(face.det_score)))
                # 每处理 3 张(或最后一张)上报一次进度，别太频繁塞队列
                if (i + 1) % 3 == 0 or i + 1 == total:
                    self.msg_queue.put(('progress', (i + 1, total,
                                                     f"已处理 {i + 1}/{total} 张，累计人脸 {len(embeddings)} 个")))

            if not embeddings:
                self.msg_queue.put(('error', '所有图片中都没有检测到人脸'))
                return

            # ---- ④ 海量人脸保护：太多时先弹窗问用户同不同意继续 ----
            if len(embeddings) > FACE_TIER_HUGE:
                est_mb = len(embeddings) * 2 // (1024 * 1024)
                self._confirm_event.clear()
                self._confirm_ok = False
                self.msg_queue.put(('confirm', (len(embeddings), est_mb)))  # 请主线程弹窗
                self._confirm_event.wait()          # ★ 阻塞等主线程把回答写回(线程握手)
                if not self._confirm_ok:
                    self.msg_queue.put(('cancelled', '人脸数量过多，已取消本轮聚类'))
                    return

            # ---- ⑤ 归一化 + 智能聚类 ----
            emb = np.array(embeddings)              # 列表 → 二维矩阵 (人脸数, 512)
            if emb.dtype == np.float16:             # 存半精度、算全精度
                emb = emb.astype(np.float32)
            emb = _l2_normalize(emb)                # 每行变单位向量 → 点积即相似度

            self.msg_queue.put(('status', '正在聚类，请耐心等待...'))
            labels = smart_cluster(emb, eps, min_cluster, qualities=qualities)
            # labels[i] = 第 i 张脸归的组号；-1 表示谁都不像(散脸)

            # ---- ⑥ 按标签分组，整理成界面要的格式 ----
            persons = {}                            # {组号: [这张脸的元信息,...]}
            unclassified = []                       # 散脸单独放
            for idx, lab in enumerate(labels):
                if lab == -1:
                    unclassified.append(metas[idx])
                else:
                    persons.setdefault(int(lab), []).append(metas[idx])

            sorted_persons = sorted(persons.values(), key=lambda x: -len(x))  # 人多在前
            groups = [{'type': 'person', 'name': '', 'items': g} for g in sorted_persons]
            if unclassified:                        # 有散脸才加一个“未分类”组
                groups.append({'type': 'unclassified', 'items': unclassified})

            self.msg_queue.put(('done', groups))    # ★ 把最终结果发给界面线程渲染

        except Exception as e:                      # 兜底：任何异常都回传给界面显示
            self.msg_queue.put(('error', f"聚类出错：{e}\n\n{traceback.format_exc()}"))

    # ============ 队列轮询（界面主线程，每 120ms 跑一次） ============
    def _poll_queue(self):
        """把后台线程塞进队列的消息，一条条取出来落实到界面上。

        消息种类：
          status      只改一句状态文字
          progress    更新进度条和文字
          confirm     海量人脸：弹窗问用户(阻塞)，把回答写回并唤醒后台线程
          cancelled   用户取消：复位状态
          error       出错：复位状态 + 弹错误窗
          done        聚类成功：统计人数、渲染人物卡片网格
        """
        try:
            while True:                             # 一口气把队列里现有消息取完
                msg = self.msg_queue.get_nowait()   # 取一条；没消息了会抛 queue.Empty
                kind = msg[0]                       # 消息的第 0 个元素 = 类型
                if kind == 'status':
                    self.status_label.setText(msg[1])
                elif kind == 'progress':
                    cur, total, text = msg[1]       # 解包：(当前, 总数, 文字)
                    self.progress.setMaximum(total)
                    self.progress.animate_to(cur)   # 进度条平滑滑到当前值
                    self.status_label.setText(text)
                elif kind == 'confirm':             # 海量人脸确认闸门
                    n, est_mb = msg[1]
                    self._confirm_ok = show_glass_question(   # 主线程弹窗(不会卡界面线程规则)
                        self, '大量人脸确认',
                        f"检测到约 {n} 张人脸，继续聚类预计占用内存约 {est_mb} MB、\n"
                        f"耗时也会明显变长。\n\n确定继续吗？")
                    self._confirm_event.set()       # 唤醒在 wait 的后台线程
                elif kind == 'cancelled':
                    self.status_label.setText(msg[1])
                    self._cluster_finished()        # 复位按钮/收起名言
                elif kind == 'error':
                    self.status_label.setText('出错')
                    self._cluster_finished()
                    show_glass_info(self, '错误', msg[1], w=600)
                elif kind == 'done':                # ★ 聚类完成：显示结果
                    self.groups = msg[1]            # 保存结果
                    n_person = len([g for g in self.groups if g['type'] == 'person'])
                    n_unc = sum(len(g['items']) for g in self.groups
                                if g['type'] == 'unclassified')
                    self.progress.animate_to(self.progress.maximum() or 1)  # 进度拉满
                    self.status_label.setText(
                        f"完成 —— 识别出 {n_person} 个人物"
                        + (f"，{n_unc} 张单张脸归入未分类" if n_unc else ""))
                    self._cluster_finished()
                    self._render_groups()           # 把结果画成卡片网格
        except queue.Empty:                         # 队列取空就结束本次轮询
            pass

    # ============ 渲染人物网格 ============
    def _render_groups(self):
        """把 self.groups 渲染成网格里的一张张 FaceCard。"""
        self._clear_grid()                          # 先清掉旧卡片
        cols = self._grid_cols()                    # 按窗口宽度算每行几列
        for c in range(cols):
            self.grid.setColumnStretch(c, 1)        # 让每一列等宽铺满
        self._show_empty()                          # 兜底显示空状态(有卡片会立刻盖掉)
        for i, group in enumerate(self.groups):     # 每组一张卡片
            card = FaceCard(group, i)
            # 用默认参数 c=card 提前“拍照”，避免循环变量被改(闭包经典坑)
            card.clicked.connect(lambda g, c=card: self._show_person(g, c.idx))  # 点卡片→详情
            card.rename.connect(self._rename_group)                             # 点命名→改名
            r, c = divmod(i, cols)                  # 第 i 张放第几行第几列
            self.grid.addWidget(card, r, c, Qt.AlignCenter)
        self._hide_empty()                          # 有卡片了就藏起空状态

    # ============ 命名人物 ============
    def _rename_group(self, group):
        """弹输入框让用户给这组人物起名，起完刷新卡片。"""
        new_name = show_glass_input(
            self, '命名人物',
            '给这位人物起个名字（导出相册时子文件夹就用这个名字）：\n'
            '留空或点取消 = 不改名。',
            initial=group.get('name', ''))
        if new_name:
            group['name'] = new_name                # 名字存进这组字典
            self._render_groups()                   # 重画卡片让名字显示出来

    # ============ 导出相册 ============
    def _export_album(self):
        """把已命名/可归属的人物，按“一人一个文件夹”导出到用户选的目录。

        两种写入方式(界面单选)：
          copy     复制一份原图过去(原图保留，占双份空间)
          shortcut 只建指向原图的 .lnk 快捷方式(几乎不占空间；一张合照有多个
                   人脸时，能同时出现在几个人的文件夹里而不重复存图)
        """
        if self._busy:
            show_glass_info(self, '提示', '聚类还在跑，等它完成再导出。')
            return
        persons = [g for g in self.groups if g['type'] == 'person']  # 未分类不导出
        if not persons:
            show_glass_info(self, '提示', '还没有可导出的人物，先聚类试试。')
            return
        out_root = QFileDialog.getExistingDirectory(
            self, '选择相册输出文件夹（将在此目录下按人物建子文件夹）')
        if not out_root:
            return

        mode = self.export_mode            # 'copy' 或 'shortcut'
        copied = 0                         # 成功处理了几张
        skipped = 0                        # 跳过几张(文件不存在/失败)
        shortcut_items = []                # 攒着的快捷方式任务：(原图, .lnk目标)

        # 逐个处理每个人物
        for i, group in enumerate(persons):
            # 文件夹名：优先用户起的名字；没有就用“人物N”
            folder = group.get('name') or f"人物{i + 1}"
            # 文件夹名不能含 Windows 禁用的 < > : " / \ | ? *，先清掉
            folder = re.sub(r'[<>:"/\\|?*]', '', folder).strip() or f"人物{i + 1}"
            folder_path = os.path.join(out_root, folder)
            try:
                os.makedirs(folder_path, exist_ok=True)   # 建子文件夹(已存在就跳过)
            except Exception:
                show_glass_info(self, '错误', f"无法创建文件夹：{folder_path}")
                return

            # 处理这组里的每一张照片
            for item in group['items']:
                src = item['image']                    # 原图路径
                if not os.path.isfile(src):            # 原图已不存在 → 跳过
                    skipped += 1
                    continue
                base = os.path.basename(src)
                if mode == 'shortcut':
                    base += '.lnk'
                dst = os.path.join(folder_path, base)
                n = 2
                while os.path.exists(dst):
                    stem, ext = os.path.splitext(base)
                    dst = os.path.join(folder_path, f"{stem}_{n}{ext}")
                    n += 1
                if mode == 'shortcut':
                    shortcut_items.append((src, dst))
                else:
                    try:
                        shutil.copy2(src, dst)
                        copied += 1
                    except Exception:
                        skipped += 1

        if mode == 'shortcut' and shortcut_items:
            made = self._make_shortcuts(shortcut_items)
            copied = made
            skipped += len(shortcut_items) - made

        if mode == 'shortcut':
            head = f"已为 {copied} 张照片创建快捷方式到：\n{out_root}\n"
        else:
            head = f"已复制 {copied} 张照片到：\n{out_root}\n"
        show_glass_info(
            self, '导出完成',
            head
            + (f"（{skipped} 张因原图缺失或写入失败被跳过）" if skipped else "")
            + "\n\n未分类的照片没有导出。")

    def _make_shortcuts(self, items):
        """用系统自带的 PowerShell 一次性批量创建 .lnk 快捷方式。

        items = [(原图完整路径, 快捷方式目标路径), ...]
        返回成功创建的个数；失败(如 PowerShell 异常)返回 0。
        返回 0 的坑都处理了：脚本写进临时 .ps1 再 -File 执行，避免“命令行太长”。
        """
        if not items:
            return 0
        # 拼 PowerShell 脚本：每行建一个快捷方式(WScript.Shell 是 Windows 自带接口)
        lines = ["$ws = New-Object -ComObject WScript.Shell"]
        for src, dst in items:
            lines.append(
                "$sc = $ws.CreateShortcut('{0}'); $sc.TargetPath = '{1}'; $sc.Save()"
                .format(dst.replace("'", "''"), src.replace("'", "''")))  # 单引号翻倍转义
        script = "\n".join(lines)
        ps1 = os.path.join(tempfile.gettempdir(), 'face_album_lnk.ps1')
        try:
            # utf-8-sig(带 BOM)：保证脚本里含中文路径时 PowerShell 能正确读
            with open(ps1, 'w', encoding='utf-8-sig') as f:
                f.write(script)
            # 调 powershell 执行；CREATE_NO_WINDOW = 不弹黑窗口
            subprocess.run(
                ['powershell', '-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden',
                 '-ExecutionPolicy', 'Bypass', '-File', ps1],
                check=True, timeout=180,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            return len(items)
        except Exception:
            return 0
        finally:
            if os.path.isfile(ps1):
                try:
                    os.remove(ps1)
                except OSError:
                    pass

    # ============ 人物详情窗口 ============
    def _show_person(self, group, idx):
        """打开某个分组的所有照片的详情窗口。"""
        dlg = PersonDialog(self, group, idx)
        dlg.exec()


# ============================================================================
# 弹窗区：提示 / 询问 / 输入 / 帮助 / 人物详情
#   全部做成“毛玻璃无边框”风格，和主界面统一；打开淡入、关闭淡出。
# ============================================================================

# ---------------------------------------------------------------------------
# _FadeCloseMixin —— 给对话框的关闭加淡出动画的“混入类”
#   Python 混入：一个小类，专门给别的类“加能力”。这里让 accept/reject
#   (确定/取消)先播一段透明度降到 0 的动画，播完再真正关闭。
# ---------------------------------------------------------------------------
class _FadeCloseMixin:
    """给对话框加关闭淡出：accept/reject 前先把窗口透明度降到 0。"""

    def _fade_close(self, act):
        """act = 真正要执行的关闭动作(QDialog.accept 或 QDialog.reject)。"""
        if getattr(self, '_closing', False):
            # 已在淡出中又来一次关闭请求（如双击✕）：直接真关闭，绝不能让 exec 卡死
            act()
            return
        self._closing = True
        try:
            a = QPropertyAnimation(self, b'windowOpacity', self)  # 动画窗口透明度
            a.setStartValue(self.windowOpacity())
            a.setEndValue(0.0)            # 渐渐变透明
            a.setDuration(170)
            a.setEasingCurve(QEasingCurve.InCubic)
            a.finished.connect(act)       # 淡出完 → 才真正 accept/reject
            a.start()
        except Exception:
            act()

    def accept(self):
        self._fade_close(QDialog.accept.__get__(self, type(self)))

    def reject(self):
        self._fade_close(QDialog.reject.__get__(self, type(self)))


# ---------------------------------------------------------------------------
# GlassDialog —— 毛玻璃对话框基座（提示/询问/输入都基于它）
#   里面预留了两个“插槽”：body_lay 放内容，btns 放按钮
# ---------------------------------------------------------------------------
class GlassDialog(_FadeCloseMixin, QDialog):
    def __init__(self, title, parent=None, w=520):
        super().__init__(parent)
        # 无边框 + 背景透明：自己画圆角毛玻璃，才能和主界面一个风格
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)               # 模态：挡住父窗口，必须处理完才放行
        self.setFixedWidth(w)             # 只固定宽度，高度按内容自适应
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        # 一个毛玻璃面板当对话框的“身体”
        self.panel = GlassPanel(tint=QColor(28, 34, 44, 225),
                                border=QColor(186, 200, 216, 70), radius=14)
        outer.addWidget(self.panel)
        lay = QVBoxLayout(self.panel)
        lay.setContentsMargins(28, 24, 28, 22)
        lay.setSpacing(14)

        # 顶部一行：标题(左) + ✕关闭(右)
        head = QHBoxLayout()
        t = QLabel(title)
        t.setStyleSheet('color:white;font-size:17px;font-weight:700;background:transparent;')
        head.addWidget(t)
        head.addStretch(1)
        x = QPushButton('✕')
        x.setFixedSize(26, 26)
        x.setCursor(QCursor(Qt.PointingHandCursor))
        x.setStyleSheet('QPushButton{background:rgba(255,255,255,18);color:white;'
                        'border:none;border-radius:13px;}'
                        'QPushButton:hover{background:rgba(255,255,255,40);}')
        x.clicked.connect(self.reject)
        head.addWidget(x)
        lay.addLayout(head)

        self.body_lay = QVBoxLayout()
        self.body_lay.setSpacing(12)
        lay.addLayout(self.body_lay)

        self.btns = QHBoxLayout()
        self.btns.setSpacing(10)
        lay.addLayout(self.btns)

    def add_button(self, text, primary=False, on_click=None):
        b = GlassButton(text, primary=primary)
        if on_click:
            b.clicked.connect(on_click)
        self.btns.addWidget(b)
        return b

    def showEvent(self, e):
        super().showEvent(e)
        _fade_in_window(self)


def _dlg_label(text):
    """造一个“可换行、可选中复制”的正文标签，供各种弹窗复用。"""
    lbl = QLabel(text)
    lbl.setWordWrap(True)                                  # 太长自动换行
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 文字能用鼠标选中复制
    lbl.setStyleSheet('color:rgba(245,247,250,230);font-size:13px;background:transparent;')
    return lbl


def show_glass_info(parent, title, text, w=520):
    """毛玻璃“提示框”：只有一个【好】按钮。"""
    dlg = GlassDialog(title, parent, w)
    dlg.body_lay.addWidget(_dlg_label(text))
    dlg.btns.addStretch(1)                 # 把按钮推到右边
    dlg.add_button('好', primary=True, on_click=dlg.accept)
    dlg.exec()


def show_glass_question(parent, title, text, w=560):
    """毛玻璃“询问框”：返回 True=用户点确定 / False=点取消或关窗。"""
    dlg = GlassDialog(title, parent, w)
    dlg.body_lay.addWidget(_dlg_label(text))
    dlg.btns.addStretch(1)
    dlg.add_button('取消', primary=False, on_click=dlg.reject)
    dlg.add_button('确定', primary=True, on_click=dlg.accept)
    return dlg.exec() == QDialog.Accepted   # exec 返回 Accepted 才代表“确定”


def show_glass_input(parent, title, prompt, initial='', w=520):
    """毛玻璃“输入框”：返回用户输入的文本(点取消则返回 None)。"""
    dlg = GlassDialog(title, parent, w)
    dlg.body_lay.addWidget(_dlg_label(prompt))
    edit = QLineEdit(initial)              # 输入框，默认填上次的内容
    edit.setStyleSheet(
        'QLineEdit{background:rgba(255,255,255,20);border:1px solid rgba(255,255,255,40);'
        'border-radius:10px;padding:9px 12px;color:white;font-size:14px;}'
        'QLineEdit:focus{border:1px solid ' + ACCENT + ';}')
    edit.selectAll()
    dlg.body_lay.addWidget(edit)
    result = {'text': None}

    def ok():
        result['text'] = edit.text().strip()
        dlg.accept()

    edit.returnPressed.connect(ok)
    dlg.btns.addStretch(1)
    dlg.add_button('取消', primary=False, on_click=dlg.reject)
    dlg.add_button('确定', primary=True, on_click=ok)
    dlg.exec()
    return result['text']


# ---------------------------------------------------------------------------
# 帮助子窗口：与父窗口同样的毛玻璃样式，背景用父背景图裁切出的中心
# （画面中央的女孩与星河），介绍功能与操作。
# ---------------------------------------------------------------------------
class HelpDialog(_FadeCloseMixin, QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._bg = load_background_center()
        self._drag = False
        self._off = None
        self.resize(720, 720)
        self.setMinimumSize(620, 620)

        # 首次显示时限制不超过父窗口并居中
        self._positioned = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(36, 28, 36, 28)
        outer.setSpacing(16)

        # 顶部标题 + 关闭
        head = QHBoxLayout()
        t = QLabel('使用帮助')
        t.setFont(QFont('HarmonyOS Sans SC', 24, QFont.Bold))
        t.setStyleSheet('color:white;background:transparent;')
        head.addWidget(t)
        head.addStretch(1)
        x = QPushButton('✕')
        x.setFixedSize(30, 30)
        x.setCursor(QCursor(Qt.PointingHandCursor))
        x.setStyleSheet('QPushButton{background:rgba(255,255,255,18);color:white;'
                        'border:none;border-radius:15px;}'
                        'QPushButton:hover{background:rgba(255,69,58,200);}')
        x.clicked.connect(self.accept)
        head.addWidget(x)
        outer.addLayout(head)

        # 功能与操作介绍
        func = QLabel(
            '■  功 能 介 绍\n'
            '· 人脸识别：自动检测照片里的人脸并提取特征\n'
            '· 自动分组：同一人物自动聚成一组，不用一张张翻\n'
            '· 查看详情：点击人物卡片，查看该人物的全部照片\n'
            '· 命名人物：给人物起名字，导出时自动建同名文件夹\n'
            '· 一键导出：按人物把照片复制或建快捷方式归档')
        func.setStyleSheet('color:rgba(255,255,255,240);font-size:17px;'
                           'background:transparent;')
        func.setWordWrap(True)
        outer.addWidget(func)

        op = QLabel(
            '■  操 作 介 绍\n'
            '1. 点击「浏览」选择照片文件夹\n'
            '2. 按需调整 eps 阈值 / 最少张数 / GPU 加速\n'
            '3. 点击「开始聚类」等待识别完成\n'
            '4. 完成后点击人物卡片查看全部照片\n'
            '5. 点卡片上的「命名」给人物起名字\n'
            '6. 选「复制」或「快捷方式」，点「导出相册」归档')
        op.setStyleSheet('color:rgba(255,255,255,240);font-size:17px;'
                         'background:transparent;')
        op.setWordWrap(True)
        outer.addWidget(op)

        outer.addStretch(1)
        bottom = QLabel('照片全程在本机处理，不会上传，请放心使用。')
        bottom.setStyleSheet('color:rgba(255,255,255,150);font-size:14px;'
                             'background:transparent;')
        outer.addWidget(bottom)

    # ---- 首次显示时：限制不超出父窗口并居中；然后淡入 ----
    def showEvent(self, e):
        super().showEvent(e)
        if not self._positioned:                 # 只在第一次显示时定位
            self._positioned = True
            parent = self.parentWidget()
            if parent is not None:
                pw, ph = parent.width(), parent.height()
                # 帮助窗口不能比父窗口还大(留 40px 边)
                self.resize(min(self.width(), pw - 40), min(self.height(), ph - 40))
                pg = parent.frameGeometry()
                self.move(pg.center().x() - self.width() // 2,    # 水平居中
                          pg.center().y() - self.height() // 2)   # 垂直居中
        _fade_in_window(self)                    # 打开淡入动画

    def paintEvent(self, e):
        """画背景：圆角裁出窗口形状，把中心裁切图(女孩星河)cover 铺满。"""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 16, 16)   # 圆角形状
        p.setClipPath(path)                      # 只画圆角内
        if self._bg:
            s = self.size()
            # KeepAspectRatioByExpanding：等比放大到“盖满”窗口再居中裁边(cover 效果)
            pm = self._bg.scaled(s, Qt.KeepAspectRatioByExpanding,
                                 Qt.SmoothTransformation)
            x = (pm.width() - s.width()) // 2
            y = (pm.height() - s.height()) // 2
            p.drawPixmap(QRectF(0, 0, s.width(), s.height()), pm,
                         QRectF(max(0, x), max(0, y), s.width(), s.height()))
        else:
            p.fillRect(self.rect(), QColor(26, 30, 38))   # 没背景就深色兜底
        p.setClipping(False)
        p.setPen(QPen(QColor(186, 200, 216, 90), 1))     # 一圈细边更像玻璃窗
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 16, 16)

    # ---- 支持用鼠标把整个窗口拖走(无边框窗口没有系统标题栏) ----
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = True
            # 记下“鼠标当前位置 - 窗口左上角”的偏移，拖动时保持相对位置不变
            self._off = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag and e.buttons() & Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self._off)   # 跟着鼠标走
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag = False                       # 松开鼠标停止拖动
        super().mouseReleaseEvent(e)


# ---------------------------------------------------------------------------
# PersonDialog —— 人物详情窗口
#   点主界面的卡片后打开：毛玻璃面板里放这一组所有照片的头像网格，
#   点某个头像 → 用系统默认程序打开原图。
# ===========================================================================
class PersonDialog(_FadeCloseMixin, QDialog):
    """毛玻璃详情窗口：网格展示某个人物的所有照片，点击打开原图。"""

    def __init__(self, parent, group, idx):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._group = group                 # 保存这一组数据
        self.setModal(False)
        self.resize(1140, 780)
        self.setMinimumSize(760, 540)

        # 窗口标题：人物名(或“人物N”) + 照片数量
        if group['type'] == 'unclassified':
            title = f"未分类 · 共 {len(group['items'])} 张照片"
        else:
            gname = group.get('name') or f"人物 {idx + 1}"
            title = f"{gname} · 共 {len(group['items'])} 张照片"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)

        # 整个窗口 = 一块毛玻璃面板
        panel = GlassPanel(tint=QColor(36, 42, 54, 225), radius=14)
        outer.addWidget(panel)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(12)

        head = QHBoxLayout()
        t = QLabel(title)
        t.setStyleSheet('color:white;font-size:18px;font-weight:700;background:transparent;')
        head.addWidget(t)
        head.addStretch(1)
        tip = QLabel('点击任意照片用系统默认程序打开原图')
        tip.setStyleSheet('color:rgba(245,247,250,130);font-size:12px;background:transparent;')
        head.addWidget(tip)
        x = QPushButton('✕')
        x.setFixedSize(28, 28)
        x.setCursor(QCursor(Qt.PointingHandCursor))
        x.setStyleSheet('QPushButton{background:rgba(255,255,255,18);color:white;'
                        'border:none;border-radius:14px;}'
                        'QPushButton:hover{background:rgba(255,255,255,40);}')
        x.clicked.connect(self.accept)
        head.addWidget(x)
        lay.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet('QScrollArea{background:transparent;border:none;}'
                             'QScrollBar:vertical{background:transparent;width:8px;}'
                             'QScrollBar::handle:vertical{background:rgba(255,255,255,50);'
                             'border-radius:4px;min-height:30px;}'
                             'QScrollBar::add-line:vertical,'
                             'QScrollBar::sub-line:vertical{height:0;}')
        scroll.viewport().setAutoFillBackground(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(scroll, 1)

        host = QWidget()
        host.setAttribute(Qt.WA_TranslucentBackground)
        host.setStyleSheet('background:transparent;')
        grid = QGridLayout(host)
        grid.setSpacing(14)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        scroll.setWidget(host)

        cols = 4
        for c in range(cols):
            grid.setColumnStretch(c, 1)
        for i, item in enumerate(group['items']):
            cell = QWidget()
            cell.setAttribute(Qt.WA_TranslucentBackground)
            cell.setFixedSize(236, 236)
            cv = QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(6)
            thumb = make_face_thumb(item['image'], item['bbox'], size=(190, 190))
            av = Avatar(thumb, size=190)
            cv.addWidget(av, 1, Qt.AlignHCenter)
            name = os.path.basename(item['image'])
            if len(name) > 20:
                name = name[:18] + '...'
            nl = QLabel(name)
            nl.setStyleSheet('color:rgba(245,247,250,200);font-size:12px;'
                             'background:transparent;')
            nl.setAlignment(Qt.AlignCenter)
            cv.addWidget(nl)
            r, c = divmod(i, cols)
            grid.addWidget(cell, r, c)
            av.clicked.connect(lambda p=item['image']: self._open_full(p))

    def showEvent(self, e):
        super().showEvent(e)
        _fade_in_window(self)

    @staticmethod
    def _open_full(path):
        try:
            os.startfile(path)
        except Exception:
            show_glass_info(None, '照片路径', path)


# ============================================================================
# 程序入口
# ============================================================================
def main():
    """启动整个程序：建 QApplication(每个 Qt 程序有且只有一个) → 建主窗口 → 进入事件循环。

    app.exec() 这一行会“卡住”程序不退出，一直处理点击/重绘/定时器等事件，
    直到用户关掉主窗口才返回。所以它要放在最后。
    """
    app = QApplication(sys.argv)     # 全局唯一的应用对象(管理事件循环)
    app.setApplicationName('人脸聚类相册')
    w = MainWindow()                 # 创建主窗口(构造里已经把界面搭好)
    w.show()                         # 让窗口显示出来
    sys.exit(app.exec())             # 进入事件循环；退出时把 Qt 的返回值交给系统


# __name__ == '__main__' 表示“本文件被直接运行”(而不是被 import)。
# 只有被直接运行时才启动界面；被别的文件 import 时不会弹窗口。
if __name__ == '__main__':
    main()
