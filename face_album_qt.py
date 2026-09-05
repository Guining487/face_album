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

# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
import os            # 路径、遍历、打开文件
import sys           # 解释器、版本、frozen 判断
import re            # 正则：清洗文件夹名
import shutil        # 复制原图 / 移动 DLL
import subprocess    # 调 PowerShell 批量建快捷方式
import tempfile      # 临时 .ps1 脚本
import threading     # 后台聚类线程
import queue         # 线程安全消息队列
import urllib.request
import json
import zipfile
import traceback     # 打印完整错误栈

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import DBSCAN
import onnxruntime
from insightface.app import FaceAnalysis

from PySide6.QtCore import (Qt, QRectF, QTimer, QPropertyAnimation,
                            QVariantAnimation, QEasingCurve, QAbstractAnimation,
                            QRect, Signal)
from PySide6.QtGui import (QPainter, QPainterPath, QColor, QPixmap, QImage, QFont,
                           QPen, QCursor, QLinearGradient, QBrush, QFontMetrics)
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QLineEdit,
                               QCheckBox, QRadioButton, QFrame, QScrollArea,
                               QGridLayout, QHBoxLayout, QVBoxLayout, QFileDialog,
                               QProgressBar, QDialog)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 模型根目录：优先程序目录，回退用户目录
_CANDIDATE_ROOTS = [
    os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
                 else os.path.dirname(os.path.abspath(__file__)), '.insightface'),
    os.path.join(os.path.expanduser('~'), '.insightface'),
]
MODEL_ROOT = next((r for r in _CANDIDATE_ROOTS
                   if os.path.isdir(os.path.join(r, 'models', 'buffalo_l'))),
                  _CANDIDATE_ROOTS[-1])

SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MODEL_DL_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
GPU_RUNTIME_URL = ''
GPU_RUNTIME_DLLS = ('onnxruntime_providers_cuda.dll', 'cublas64_12.dll',
                    'cublasLt64_12.dll', 'cudart64_12.dll', 'cudnn64_9.dll')

# ---- 视觉常量：Win7 Aero 毛玻璃风格 ----
BG_IMG_PATH = r'E:\vscode_background\star.png'     # VSCode 当前背景图（星空）
WINDOW_RADIUS = 8                                  # 主窗口圆角（Aero 的小圆角）
PANEL_TINT = QColor(28, 34, 44, 95)               # 面板底色：Aero 深蓝灰半透明玻璃
PANEL_BORDER = QColor(186, 200, 216, 70)           # 面板细边框（浅蓝灰）
CARD_TINT = QColor(255, 255, 255, 26)              # 卡片底色：更透的玻璃
CARD_HOVER = QColor(255, 255, 255, 46)             # 卡片悬停亮一档
CARD_RADIUS = 14
ACCENT = '#2f7bc4'                                 # Win7 Aero 蓝（主色）
ACCENT_HOVER = '#5496d8'
ACCENT_PRESS = '#1d5fa0'
CLOSE_HOVER = '#c42b1c'                            # 关闭按钮悬停红

TEXT = QColor(245, 247, 250)
TEXT_SUB = QColor(245, 247, 250, 160)
TEXT_DIM = QColor(245, 247, 250, 95)

FONT_FAMILY = 'Microsoft YaHei'

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

FACE_TIER_MID = 30000
FACE_TIER_HUGE = 150000
SIM_CHUNK = 4096
GRID_CARD_W = 236       # 卡片固定宽
GRID_CARD_H = 300       # 卡片固定高


def qss_color(c):
    """把 QColor 转成 QSS 可用的 rgba(...) 字符串（保留透明度）。"""
    if isinstance(c, str):
        return c
    return f'rgba({c.red()},{c.green()},{c.blue()},{c.alpha()})'


def anim_value(fn, c1, c2, ms=180, widget=None, easing=QEasingCurve.OutCubic,
               on_done=None):
    """对任意值做插值动画：每帧回调 fn(当前值)。"""
    a = QVariantAnimation(widget)
    a.setStartValue(c1)
    a.setEndValue(c2)
    a.setDuration(ms)
    a.setEasingCurve(easing)
    a.valueChanged.connect(fn)
    if on_done:
        a.finished.connect(on_done)
    a.start(QAbstractAnimation.DeleteWhenStopped)
    return a


# ---------------------------------------------------------------------------
# 基础业务函数（与 tkinter 版完全一致）
# ---------------------------------------------------------------------------
def _nvidia_driver_present():
    sr = os.environ.get('SystemRoot') or r'C:\Windows'
    return os.path.exists(os.path.join(sr, 'System32', 'nvcuda.dll'))


def _gpu_runtime_ready():
    capi = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi')
    return all(os.path.exists(os.path.join(capi, n)) for n in GPU_RUNTIME_DLLS)


def collect_images(root):
    imgs = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in SUPPORTED:
                imgs.append(os.path.join(dirpath, f))
    return imgs


def read_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _l2_normalize(emb):
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


def _group_centroids(emb, labels):
    groups = {}
    for i, lab in enumerate(labels):
        if lab < 0:
            continue
        groups.setdefault(int(lab), []).append(i)
    centroids = {}
    for lab, idxs in groups.items():
        mean = emb[idxs].mean(axis=0)
        norm = np.linalg.norm(mean) + 1e-9
        centroids[lab] = mean / norm
    return centroids, groups


def _reassign_to_nearest(emb, labels, centroids, min_score, qualities=None):
    lab_list = list(centroids.keys())
    if not lab_list:
        return np.full(len(emb), -1, dtype=int)
    C = np.stack([centroids[l] for l in lab_list])
    n_faces = len(emb)
    sims = np.empty((n_faces, len(lab_list)), dtype=np.float32)
    for s in range(0, n_faces, SIM_CHUNK):
        e = min(s + SIM_CHUNK, n_faces)
        sims[s:e] = emb[s:e] @ C.T
    best = sims.argmax(axis=1)
    best_score = sims[np.arange(len(emb)), best]
    thresholds = np.full(len(emb), min_score)
    if qualities is not None:
        relax = 0.10
        thresholds = min_score - (1.0 - np.asarray(qualities)) * relax
    new_labels = np.where(best_score >= thresholds,
                          np.array([lab_list[b] for b in best]), -1)
    return new_labels.astype(int)


def _merge_close_groups(emb, labels, centroids, merge_score):
    lab_list = list(centroids.keys())
    if len(lab_list) < 2:
        return labels
    C = np.stack([centroids[l] for l in lab_list])
    g = len(lab_list)
    sim = np.empty((g, g), dtype=np.float32)
    for a in range(0, g, SIM_CHUNK):
        a2 = min(a + SIM_CHUNK, g)
        sim[a:a2] = C[a:a2] @ C.T
    new_labels = np.array(labels)
    for a in range(len(lab_list)):
        for b in range(a + 1, len(lab_list)):
            if sim[a][b] >= merge_score:
                big, small = lab_list[a], lab_list[b]
                new_labels[new_labels == small] = big
    return new_labels.astype(int)


def _face_quality_score(image, bbox, det_score=1.0):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    crop = image[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(1.0, laplacian_var / 500.0)
    face_area = (x2 - x1) * (y2 - y1)
    img_area = max(1, h * w)
    area_ratio = face_area / img_area
    size_score = min(1.0, area_ratio / 0.01)
    return max(0.0, min(1.0, 0.4 * sharpness + 0.3 * size_score + 0.3 * det_score))


def smart_cluster(emb, eps, min_cluster, max_iters=3, qualities=None):
    cl = DBSCAN(eps=np.sqrt(2.0 * eps), min_samples=min_cluster,
                metric='euclidean', algorithm='ball_tree')
    labels = cl.fit_predict(emb)
    min_score = 1.0 - eps
    merge_score = 1.0 - eps * 1.2
    for _ in range(max_iters):
        centroids, _ = _group_centroids(emb, labels)
        if not centroids:
            break
        labels = _merge_close_groups(emb, labels, centroids, merge_score)
        centroids, _ = _group_centroids(emb, labels)
        labels = _reassign_to_nearest(emb, labels, centroids, min_score, qualities)
    centroids, groups = _group_centroids(emb, labels)
    for lab, idxs in groups.items():
        if len(idxs) < min_cluster:
            labels[np.array(idxs)] = -1
    return labels


# ---------------------------------------------------------------------------
# 图像 -> QPixmap 工具
# ---------------------------------------------------------------------------
def _cv_to_qpixmap(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def make_face_thumb(image_path, bbox, size=(176, 176)):
    """按人脸框裁剪出正方形缩略图，返回 QPixmap；失败返回 None。"""
    try:
        img = read_image(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = int((x2 - x1) * 0.35)
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop)
        pil.thumbnail((size[0], size[1]), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (size[0], size[1]), (28, 30, 36))
        canvas.paste(pil, ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2))
        data = canvas.tobytes('raw', 'RGB')
        qimg = QImage(data, size[0], size[1], size[0] * 3, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)
    except Exception:
        return None


def load_background():
    """加载并预处理全局背景：轻微模糊 + 轻微压暗，保留主体细节（能看清女孩/星河）。"""
    try:
        img = read_image(BG_IMG_PATH)
        if img is None:
            return None
        # 缩到 1/2 保持细节，模糊很轻（sigma≈9），主要保证文字区不刺眼
        small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 9)
        small = cv2.addWeighted(small, 0.85, np.zeros_like(small), 0.15, 0)
        return _cv_to_qpixmap(small)
    except Exception:
        return None


def load_background_center():
    """裁切原始背景图的中心区域（画面中央的女孩与星河），供帮助子窗口当背景。"""
    try:
        img = read_image(BG_IMG_PATH)
        if img is None:
            return None
        h, w = img.shape[:2]
        ch, cw = int(h * 0.72), int(w * 0.62)
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        crop = img[y0:y0 + ch, x0:x0 + cw]
        # 轻微压暗，保证窗口里的文字可读
        crop = cv2.addWeighted(crop, 0.72, np.zeros_like(crop), 0.28, 0)
        return _cv_to_qpixmap(crop)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 毛玻璃面板
# ---------------------------------------------------------------------------
class GlassPanel(QFrame):
    """半透明磨砂面板：低透明度底色 + 细边框 + 圆角，悬停轻微变亮。"""

    def __init__(self, parent=None, tint=PANEL_TINT, border=PANEL_BORDER,
                 radius=16, hover_tint=None):
        super().__init__(parent)
        self._tint = tint
        self._border = border
        self._hover_tint = hover_tint or tint
        self._radius = radius
        self._t = 0.0
        self.setAttribute(Qt.WA_TranslucentBackground)

    def set_hover(self, on, ms=160):
        anim_value(self._on_t, self._t, 1.0 if on else 0.0, ms=ms, widget=self)

    def _on_t(self, v):
        self._t = v
        self.update()

    def enterEvent(self, e):
        self.set_hover(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.set_hover(False)
        super().leaveEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, self._radius, self._radius)
        base = QColor(self._tint)
        hov = QColor(self._hover_tint)
        col = QColor(int(base.red() + (hov.red() - base.red()) * self._t),
                     int(base.green() + (hov.green() - base.green()) * self._t),
                     int(base.blue() + (hov.blue() - base.blue()) * self._t),
                     int(base.alpha() + (hov.alpha() - base.alpha()) * self._t))
        p.fillPath(path, col)
        p.setPen(QPen(self._border, 1))
        p.drawPath(path)


# ---------------------------------------------------------------------------
# 圆角头像标签（自绘圆形头像，点击发信号）
# ---------------------------------------------------------------------------
class Avatar(QLabel):
    clicked = Signal()

    def __init__(self, pixmap, size=176, parent=None):
        super().__init__(parent)
        self._pix = pixmap
        self._sz = size
        self.setFixedSize(size, size)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = QRectF(0, 0, self._sz, self._sz)
        path = QPainterPath()
        path.addEllipse(rect)
        p.setClipPath(path)
        if self._pix and not self._pix.isNull():
            p.drawPixmap(rect, self._pix, QRectF(0, 0, self._pix.width(), self._pix.height()))
        else:
            p.fillRect(rect, QColor(40, 42, 48))
        p.setClipping(False)
        p.setPen(QPen(QColor(255, 255, 255, 46), 1))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(rect)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


# ---------------------------------------------------------------------------
# 透明无色按钮（无填充，仅文字 + 悬停极淡反馈）
# ---------------------------------------------------------------------------
class GlassButton(QPushButton):
    def __init__(self, text='', primary=False, parent=None):
        super().__init__(text, parent)
        self._primary = primary
        self._hov = 0.0
        self._press = 0.0
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFlat(True)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumHeight(38)

    # ---- 动画 ----
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
                   easing=QEasingCurve.OutBack)
        super().mouseReleaseEvent(e)

    def _set_hov(self, v):
        self._hov = v
        self.update()

    def _set_press(self, v):
        self._press = v
        self.update()

    # ---- 绘制 ----
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        disabled = not self.isEnabled()
        r = QRectF(self.rect())
        hov = self._hov if not disabled else 0.0
        press = self._press if not disabled else 0.0

        # 悬停时画一层极淡的白色反馈（几乎无色），按下略微加深
        if hov > 0.01 and not disabled:
            alpha = int(26 * hov + 14 * press)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, alpha))
            p.drawRoundedRect(r, 10, 10)
        elif press > 0.01 and not disabled:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, int(18 * press)))
            p.drawRoundedRect(r, 10, 10)

        # 文字
        alpha = 120 if disabled else (255 if self._primary else 235)
        f = QFont(FONT_FAMILY, 13)
        f.setBold(self._primary)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255, alpha))
        p.drawText(r, Qt.AlignCenter, self.text())


# ---------------------------------------------------------------------------
# 圆角进度条（数值丝滑过渡）
# ---------------------------------------------------------------------------
class GlassProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(False)
        self.setFixedHeight(8)

    def animate_to(self, value, ms=320):
        a = QPropertyAnimation(self, b'value', self)
        a.setStartValue(self.value())
        a.setEndValue(value)
        a.setDuration(ms)
        a.setEasingCurve(QEasingCurve.OutCubic)
        a.start(QAbstractAnimation.DeleteWhenStopped)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 30))
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
        mx = self.maximum()
        frac = min(1.0, self.value() / mx) if mx > 0 else 0.0
        if frac > 0:
            fill = QRectF(0, 0, w * frac, h)
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0, QColor(ACCENT))
            grad.setColorAt(1, QColor('#5496d8'))
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(fill, h / 2, h / 2)


# ---------------------------------------------------------------------------
# 名言卡片：带「模块下放」展开动画
# ---------------------------------------------------------------------------
class QuoteCard(QWidget):
    """聚类等待时的名言卡片。平时高度 0；播放时从进度条下方「下放」滑出展开。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._expand = 0.0
        self._fade = 1.0          # 0~1：名言切换时的淡出/淡入
        self._text = ''
        self._natural_h = 68
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedHeight(0)
        self._anim = None

    def set_text(self, t):
        self._text = t
        self.update()

    def _kill_anim(self):
        """停止并丢弃旧动画。旧动画用完后会被 Qt 自动销毁，这里要防访问已删除对象。"""
        if self._anim is not None:
            try:
                self._anim.stop()
            except RuntimeError:
                pass
            self._anim = None

    def show_animated(self):
        """模块下放展开（开始播放名言时）。"""
        self._kill_anim()
        self._fade = 1.0
        self.setFixedHeight(0)
        a = QVariantAnimation(self)
        a.setStartValue(0.0)
        a.setEndValue(1.0)
        a.setDuration(460)
        a.setEasingCurve(QEasingCurve.OutBack)
        a.valueChanged.connect(self._set_expand)
        a.start()
        self._anim = a

    def hide_animated(self, on_done=None):
        """收起（高度归零）。"""
        self._kill_anim()
        a = QVariantAnimation(self)
        a.setStartValue(self._expand)
        a.setEndValue(0.0)
        a.setDuration(320)
        a.setEasingCurve(QEasingCurve.InCubic)
        a.valueChanged.connect(self._set_expand)
        if on_done:
            a.finished.connect(on_done)
        a.start()
        self._anim = a

    def fade_out(self, on_mid):
        """淡出当前句，完成后回调 on_mid（用于换字后再淡入）。"""
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
        self._expand = v
        self.setFixedHeight(int(self._natural_h * v))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        a = self._expand
        if a <= 0.001:
            return
        f = self._fade
        # 透明背景，只有文字——名言直接浮在背景上，与背景融为一体
        r = QRectF(self.rect()).adjusted(20, 0, -20, 0)
        p.setPen(QColor(255, 255, 255, min(255, int(235 * a * f))))
        p.setFont(QFont(FONT_FAMILY, 14))
        p.drawText(r, Qt.AlignCenter, f"❝ {self._text} ❞")


# ---------------------------------------------------------------------------
# 人物卡片：自绘毛玻璃卡片（圆形头像 + 命名按钮 + 悬停/按压动画）
# ---------------------------------------------------------------------------
class FaceCard(QWidget):
    clicked = Signal(object)     # emit(group)
    rename = Signal(object)      # emit(group)

    def __init__(self, group, idx, parent=None):
        super().__init__(parent)
        self.group = group
        self.idx = idx
        self.setFixedSize(GRID_CARD_W, GRID_CARD_H)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setMouseTracking(True)
        self._t = 0.0
        self._press = 0.0
        first = group['items'][0]
        self._thumb = make_face_thumb(first['image'], first['bbox'], size=(150, 150))
        if group['type'] == 'unclassified':
            self._title = '未分类'
        else:
            self._title = group.get('name') or f'人物 {idx + 1}'
        self._sub = f"共 {len(group['items'])} 张"
        self._rename_rect = QRect(0, 0, 92, 34)

    # ---- 动画 ----
    def set_hover(self, on, ms=180):
        anim_value(self._set_t, self._t, 1.0 if on else 0.0, ms=ms, widget=self)

    def _set_t(self, v):
        self._t = v
        self.update()

    def press_animate(self):
        anim_value(self._set_press, 0.0, 1.0, ms=170, widget=self,
                   easing=QEasingCurve.OutQuad, on_done=self._bounce_back)

    def _bounce_back(self):
        anim_value(self._set_press, 1.0, 0.0, ms=280, widget=self,
                   easing=QEasingCurve.OutBack)

    def _set_press(self, v):
        self._press = v
        self.update()

    def enterEvent(self, e):
        self.set_hover(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.set_hover(False)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.press_animate()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            if self.group['type'] == 'person' and \
               self._rename_rect.contains(e.position().toPoint()):
                self.rename.emit(self.group)
            else:
                self.clicked.emit(self.group)
        super().mouseReleaseEvent(e)

    # ---- 绘制 ----
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        w = self.width()
        lift = int(6 * self._t)
        press = int(3 * self._press)
        dy = lift - press

        # 悬停阴影（浮起感）
        if self._t > 0.01:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, int(90 * self._t)))
            p.drawRoundedRect(QRectF(4, 8, w - 8, self.height() - 6), 20, 20)

        body = QRectF(0, dy, w, self.height() - 6)
        path = QPainterPath()
        path.addRoundedRect(body, CARD_RADIUS, CARD_RADIUS)
        base = QColor(CARD_TINT)
        hov = QColor(CARD_HOVER)
        col = QColor(int(base.red() + (hov.red() - base.red()) * self._t),
                     int(base.green() + (hov.green() - base.green()) * self._t),
                     int(base.blue() + (hov.blue() - base.blue()) * self._t),
                     int(base.alpha() + (hov.alpha() - base.alpha()) * self._t))
        p.fillPath(path, col)
        p.setPen(QPen(QColor(255, 255, 255, int(36 + 30 * self._t)), 1))
        p.drawPath(path)

        # 圆形头像
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
            p.fillRect(av_rect, QColor(42, 44, 52))
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(av_rect)

        # 名字
        f = QFont(FONT_FAMILY, 14)
        f.setBold(True)
        p.setFont(f)
        name_rect = QRectF(16, av_y + av_sz + 12, w - 32, 24)
        title = self._title
        if QFontMetrics(f).horizontalAdvance(title) > w - 32:
            title = QFontMetrics(f).elidedText(title, Qt.ElideRight, int(w - 32))
        p.setPen(TEXT)
        p.drawText(name_rect, Qt.AlignCenter, title)

        # 张数
        p.setPen(TEXT_SUB)
        p.setFont(QFont(FONT_FAMILY, 11))
        p.drawText(QRectF(16, av_y + av_sz + 38, w - 32, 20), Qt.AlignCenter, self._sub)

        # 命名按钮（仅人物组）
        if self.group['type'] == 'person':
            bx = (w - 92) / 2
            by = self.height() - 46 + dy
            self._rename_rect = QRect(int(bx), int(by), 92, 34)
            bpath = QPainterPath()
            bpath.addRoundedRect(QRectF(bx, by, 92, 34), 10, 10)
            p.setPen(Qt.NoPen)
            p.fillPath(bpath, QColor(ACCENT))
            p.drawPath(bpath)
            p.setPen(QColor('#ffffff'))
            p.setFont(QFont(FONT_FAMILY, 12))
            p.drawText(QRectF(bx, by, 92, 34), Qt.AlignCenter, '命名  ✏')

# ---------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('本地人脸聚类相册')
        self.resize(1240, 860)
        self.setMinimumSize(980, 640)
        # 使用系统原生标题栏/边框（右上角三个按钮即系统原生）

        # 业务状态
        self.app = None
        self.groups = []
        self.use_gpu = False
        self._busy = False
        self._thumb_refs = []
        self.msg_queue = queue.Queue()
        self._confirm_event = threading.Event()
        self._confirm_ok = False
        self._quote_idx = 0
        self._quote_after = None
        self._quote_showing = False
        self.export_mode = 'copy'
        self._last_cols = None

        # 硬件检测
        self._nvidia_driver = _nvidia_driver_present()
        self._gpu_runtime = _gpu_runtime_ready()
        self._cuda_available = self._nvidia_driver and self._gpu_runtime
        self.use_gpu = self._cuda_available

        # 背景
        self._bg_orig = load_background()
        self._bg_scaled = None

        self._build_ui()

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_queue)
        self._poll_timer.start(120)

    # ============ UI 构建 ============
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 20)
        root.setSpacing(14)

        # ---- 顶部标题栏：鸿蒙黑体标题 + 行楷标语 + 帮助按钮 ----
        header = QWidget()
        header.setAttribute(Qt.WA_TranslucentBackground)
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(4, 0, 12, 0)
        tcol = QVBoxLayout()
        tcol.setSpacing(2)
        title = QLabel('人脸聚类相册')
        title.setFont(QFont('HarmonyOS Sans SC', 26, QFont.Bold))
        title.setStyleSheet('color:white;background:transparent;')
        slogan = QLabel('时光会走远，但每一帧温暖的回忆，都替你记得。')
        slogan.setFont(QFont('STXingkai', 17))
        slogan.setStyleSheet('color:rgba(255,255,255,175);background:transparent;')
        tcol.addWidget(title)
        tcol.addWidget(slogan)
        hlay.addLayout(tcol)
        hlay.addStretch(1)
        self.btn_home_help = GlassButton('?')
        self.btn_home_help.setFixedSize(40, 40)
        self.btn_home_help.clicked.connect(self._show_help)
        hlay.addWidget(self.btn_home_help)
        root.addWidget(header)

        # ---- 工具面板 ----
        top = GlassPanel(tint=QColor(28, 34, 44, 95), radius=14)
        root.addWidget(top)
        top_lay = QVBoxLayout(top)
        top_lay.setContentsMargins(22, 18, 22, 14)
        top_lay.setSpacing(12)

        row1 = QHBoxLayout()
        row1.setSpacing(10)
        lbl_dir = QLabel('照片文件夹')
        lbl_dir.setStyleSheet('color:white;font-size:13px;background:transparent;')
        row1.addWidget(lbl_dir)
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText('选择存放照片的文件夹…')
        self.dir_edit.setStyleSheet(
            'QLineEdit{background:rgba(255,255,255,16);border:1px solid rgba(255,255,255,36);'
            'border-radius:10px;padding:8px 12px;color:white;font-size:13px;}'
            'QLineEdit:focus{border:1px solid ' + ACCENT + ';}')
        row1.addWidget(self.dir_edit, 1)
        self.btn_browse = GlassButton('浏览')
        self.btn_browse.clicked.connect(self._choose_dir)
        row1.addWidget(self.btn_browse)
        row1.addSpacing(12)
        self.btn_start = GlassButton('开始聚类', primary=True)
        self.btn_start.clicked.connect(self._start_cluster)
        row1.addWidget(self.btn_start)
        top_lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)

        def caption(t):
            lb = QLabel(t)
            lb.setStyleSheet('color:rgba(245,247,250,150);font-size:12px;'
                             'background:transparent;')
            return lb

        line_css = ('QLineEdit{background:rgba(255,255,255,16);'
                    'border:1px solid rgba(255,255,255,36);'
                    'border-radius:8px;padding:6px 8px;color:white;font-size:13px;}'
                    'QLineEdit:focus{border:1px solid ' + ACCENT + ';}')

        row2.addWidget(caption('参数'))
        row2.addWidget(caption('eps 阈值'))
        self.eps_edit = QLineEdit('0.43')
        self.eps_edit.setFixedWidth(58)
        self.eps_edit.setStyleSheet(line_css)
        row2.addWidget(self.eps_edit)
        row2.addWidget(caption('最少张数'))
        self.min_edit = QLineEdit('2')
        self.min_edit.setFixedWidth(46)
        self.min_edit.setStyleSheet(line_css)
        row2.addWidget(self.min_edit)

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
            self.gpu_check.setEnabled(False)
        elif not self._gpu_runtime:
            self.gpu_check.setText('GPU 加速（需联网下载运行库）')
        row2.addWidget(self.gpu_check)

        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setStyleSheet('background:rgba(255,255,255,30);')
        row2.addWidget(sep)
        row2.addSpacing(6)

        row2.addWidget(caption('导出'))
        self.radio_copy = QRadioButton('复制')
        self.radio_lnk = QRadioButton('快捷方式')
        self.radio_copy.setChecked(True)
        radio_css = ('QRadioButton{color:white;font-size:13px;background:transparent;spacing:6px;}'
                     'QRadioButton::indicator{width:16px;height:16px;border-radius:8px;'
                     'border:1px solid rgba(255,255,255,70);background:rgba(255,255,255,12);}'
                     'QRadioButton::indicator:checked{background:' + ACCENT +
                     ';border:2px solid rgba(255,255,255,220);}')
        self.radio_copy.setStyleSheet(radio_css)
        self.radio_lnk.setStyleSheet(radio_css)
        self.radio_copy.toggled.connect(
            lambda chk: setattr(self, 'export_mode', 'copy' if chk else 'shortcut'))
        self.radio_lnk.toggled.connect(
            lambda chk: setattr(self, 'export_mode', 'shortcut' if chk else 'copy'))
        row2.addWidget(self.radio_copy)
        row2.addWidget(self.radio_lnk)

        self.btn_export = GlassButton('导出相册')
        self.btn_export.clicked.connect(self._export_album)
        row2.addWidget(self.btn_export)
        row2.addWidget(caption('未命名的人物将导出为「人物N」'))
        row2.addStretch(1)
        top_lay.addLayout(row2)

        tip = QLabel('提示：点击人物卡片可查看该人物的全部照片；点击照片可用系统默认程序打开原图。')
        tip.setStyleSheet('color:rgba(245,247,250,120);font-size:12px;background:transparent;')
        top_lay.addWidget(tip)

        # ---- 状态面板：状态文字 + 进度条 + 名言卡片 ----
        status = GlassPanel(tint=QColor(28, 34, 44, 95), radius=14)
        root.addWidget(status)
        st_lay = QVBoxLayout(status)
        st_lay.setContentsMargins(22, 12, 22, 14)
        st_lay.setSpacing(10)
        self.status_label = QLabel('就绪 —— 请选择照片文件夹')
        self.status_label.setStyleSheet('color:white;font-size:13px;background:transparent;')
        st_lay.addWidget(self.status_label)
        self.progress = GlassProgressBar()
        st_lay.addWidget(self.progress)
        self.quote_card = QuoteCard()
        st_lay.addWidget(self.quote_card)

        # ---- 主区域：可滚动网格 ----
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet('QScrollArea{background:transparent;border:none;}'
                                  'QScrollBar:vertical{background:transparent;width:8px;margin:0;}'
                                  'QScrollBar::handle:vertical{background:rgba(255,255,255,50);'
                                  'border-radius:4px;min-height:30px;}'
                                  'QScrollBar::add-line:vertical,'
                                  'QScrollBar::sub-line:vertical{height:0;}')
        self.scroll.viewport().setAutoFillBackground(False)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(self.scroll, 1)

        self.grid_host = QWidget()
        self.grid_host.setStyleSheet('background:transparent;')
        self.grid_host.setAttribute(Qt.WA_TranslucentBackground)
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(16)
        self.grid.setContentsMargins(4, 4, 4, 8)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.scroll.setWidget(self.grid_host)

        # 空状态占位（绝对定位在网格上方居中）
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
        self._show_empty()

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
        super().showEvent(e)
        self._update_bg()
        self._relayout_empty()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_bg()
        self._relayout_empty()

    def paintEvent(self, e):
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
        d = QFileDialog.getExistingDirectory(self, '选择照片文件夹')
        if d:
            self.dir_edit.setText(d)

    def _gpu_toggled(self, on):
        self.use_gpu = on

    def _show_help(self):
        dlg = HelpDialog(self)
        dlg.exec()

    # ============ 聚类 ============
    def _start_cluster(self):
        d = self.dir_edit.text().strip()
        if not d or not os.path.isdir(d):
            show_glass_info(self, '提示', '请先选择有效的照片文件夹')
            return
        if self._busy:
            show_glass_info(self, '提示', '上一轮聚类还在进行中，请等待它完成')
            return
        try:
            eps = float(self.eps_edit.text())
            mc = int(self.min_edit.text())
        except Exception:
            show_glass_info(self, '提示', 'eps 或最少张数格式不正确')
            return

        self._clear_grid()
        self.groups = []
        self.progress.setValue(0)
        self._busy = True
        self.btn_start.setEnabled(False)
        self.status_label.setText('正在加载模型...')

        QTimer.singleShot(600, self._show_quote)

        threading.Thread(target=self._cluster_worker, args=(d, eps, mc),
                         daemon=True).start()

    def _clear_grid(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self._thumb_refs.clear()
        self._hide_empty()

    def _cluster_finished(self):
        self._busy = False
        self.btn_start.setEnabled(True)
        if self._quote_after is not None:
            self._quote_after.stop()
            self._quote_after = None
        self._quote_showing = False
        self.quote_card.hide_animated(lambda: (self.quote_card.set_text(''),
                                               self.quote_card.update()))

    # ---- 名言：模块下放动画 + 轮播 ----
    def _show_quote(self):
        if not self._busy:
            return
        self._quote_showing = True
        self._set_quote_text()
        self.quote_card.show_animated()
        self._schedule_next_quote()

    def _set_quote_text(self):
        q = QUOTES[self._quote_idx % len(QUOTES)]
        self._quote_idx += 1
        self.quote_card.set_text(q)

    def _schedule_next_quote(self):
        if not self._busy:
            return
        if self._quote_after is not None:
            self._quote_after.stop()
        t = QTimer(self)
        t.setSingleShot(True)
        t.timeout.connect(self._next_quote)
        t.start(4000)
        self._quote_after = t

    def _next_quote(self):
        self._quote_after = None
        if not self._busy:
            return
        # 名言切换动画：先淡出当前句 → 换新句 → 再淡入
        self.quote_card.fade_out(on_mid=self._fade_in_next)

    def _fade_in_next(self):
        if not self._busy:
            return
        self._set_quote_text()
        self.quote_card.fade_in(on_done=self._schedule_next_quote)

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
        try:
            images = collect_images(d)
            if not images:
                self.msg_queue.put(('error', '该文件夹下没有找到任何图片'))
                return
            self.msg_queue.put(('status', f"共 {len(images)} 张图片，正在加载人脸模型..."))

            if self.app is None:
                if not self._ensure_model_ready():
                    self.msg_queue.put(('error',
                        '人脸识别模型没有就绪。首次使用需要联网下载模型（约 300MB），\n'
                        '当前可能没有网络或下载失败，请检查网络后重新开始聚类。'))
                    return
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
                except Exception:
                    self.app = FaceAnalysis(name='buffalo_l', root=MODEL_ROOT,
                                            providers=['CPUExecutionProvider'])
                    self.app.prepare(ctx_id=-1, det_size=(640, 640))
                    self.msg_queue.put(('status', '显卡不可用，已自动改用 CPU 运行'))

            embeddings = []
            qualities = []
            metas = []
            total = len(images)
            use_half = False
            for i, p in enumerate(images):
                img = read_image(p)
                if img is None:
                    continue
                faces = self.app.get(img)
                if not use_half and len(embeddings) + len(faces) > FACE_TIER_MID:
                    use_half = True
                    embeddings = [e.astype(np.float16) for e in embeddings]
                for face in faces:
                    embeddings.append(face.embedding.astype(np.float16 if use_half
                                                            else np.float32))
                    metas.append({'image': p, 'bbox': face.bbox.tolist()})
                    qualities.append(_face_quality_score(img, face.bbox,
                                                         float(face.det_score)))
                if (i + 1) % 3 == 0 or i + 1 == total:
                    self.msg_queue.put(('progress', (i + 1, total,
                                                     f"已处理 {i + 1}/{total} 张，累计人脸 {len(embeddings)} 个")))

            if not embeddings:
                self.msg_queue.put(('error', '所有图片中都没有检测到人脸'))
                return

            if len(embeddings) > FACE_TIER_HUGE:
                est_mb = len(embeddings) * 2 // (1024 * 1024)
                self._confirm_event.clear()
                self._confirm_ok = False
                self.msg_queue.put(('confirm', (len(embeddings), est_mb)))
                self._confirm_event.wait()
                if not self._confirm_ok:
                    self.msg_queue.put(('cancelled', '人脸数量过多，已取消本轮聚类'))
                    return

            emb = np.array(embeddings)
            if emb.dtype == np.float16:
                emb = emb.astype(np.float32)
            emb = _l2_normalize(emb)

            self.msg_queue.put(('status', '正在聚类，请耐心等待...'))
            labels = smart_cluster(emb, eps, min_cluster, qualities=qualities)

            persons = {}
            unclassified = []
            for idx, lab in enumerate(labels):
                if lab == -1:
                    unclassified.append(metas[idx])
                else:
                    persons.setdefault(int(lab), []).append(metas[idx])

            sorted_persons = sorted(persons.values(), key=lambda x: -len(x))
            groups = [{'type': 'person', 'name': '', 'items': g} for g in sorted_persons]
            if unclassified:
                groups.append({'type': 'unclassified', 'items': unclassified})

            self.msg_queue.put(('done', groups))

        except Exception as e:
            self.msg_queue.put(('error', f"聚类出错：{e}\n\n{traceback.format_exc()}"))

    # ============ 队列轮询 ============
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                kind = msg[0]
                if kind == 'status':
                    self.status_label.setText(msg[1])
                elif kind == 'progress':
                    cur, total, text = msg[1]
                    self.progress.setMaximum(total)
                    self.progress.animate_to(cur)
                    self.status_label.setText(text)
                elif kind == 'confirm':
                    n, est_mb = msg[1]
                    self._confirm_ok = show_glass_question(
                        self, '大量人脸确认',
                        f"检测到约 {n} 张人脸，继续聚类预计占用内存约 {est_mb} MB、\n"
                        f"耗时也会明显变长。\n\n确定继续吗？")
                    self._confirm_event.set()
                elif kind == 'cancelled':
                    self.status_label.setText(msg[1])
                    self._cluster_finished()
                elif kind == 'error':
                    self.status_label.setText('出错')
                    self._cluster_finished()
                    show_glass_info(self, '错误', msg[1], w=600)
                elif kind == 'done':
                    self.groups = msg[1]
                    n_person = len([g for g in self.groups if g['type'] == 'person'])
                    n_unc = sum(len(g['items']) for g in self.groups
                                if g['type'] == 'unclassified')
                    self.progress.animate_to(self.progress.maximum() or 1)
                    self.status_label.setText(
                        f"完成 —— 识别出 {n_person} 个人物"
                        + (f"，{n_unc} 张单张脸归入未分类" if n_unc else ""))
                    self._cluster_finished()
                    self._render_groups()
        except queue.Empty:
            pass

    # ============ 渲染人物网格 ============
    def _render_groups(self):
        self._clear_grid()
        cols = self._grid_cols()
        for c in range(cols):
            self.grid.setColumnStretch(c, 1)
        self._show_empty()
        for i, group in enumerate(self.groups):
            card = FaceCard(group, i)
            card.clicked.connect(lambda g, c=card: self._show_person(g, c.idx))
            card.rename.connect(self._rename_group)
            r, c = divmod(i, cols)
            self.grid.addWidget(card, r, c, Qt.AlignCenter)
        self._hide_empty()

    # ============ 命名人物 ============
    def _rename_group(self, group):
        new_name = show_glass_input(
            self, '命名人物',
            '给这位人物起个名字（导出相册时子文件夹就用这个名字）：\n'
            '留空或点取消 = 不改名。',
            initial=group.get('name', ''))
        if new_name:
            group['name'] = new_name
            self._render_groups()

    # ============ 导出相册 ============
    def _export_album(self):
        if self._busy:
            show_glass_info(self, '提示', '聚类还在跑，等它完成再导出。')
            return
        persons = [g for g in self.groups if g['type'] == 'person']
        if not persons:
            show_glass_info(self, '提示', '还没有可导出的人物，先聚类试试。')
            return
        out_root = QFileDialog.getExistingDirectory(
            self, '选择相册输出文件夹（将在此目录下按人物建子文件夹）')
        if not out_root:
            return

        mode = self.export_mode
        copied = 0
        skipped = 0
        shortcut_items = []

        for i, group in enumerate(persons):
            folder = group.get('name') or f"人物{i + 1}"
            folder = re.sub(r'[<>:"/\\|?*]', '', folder).strip() or f"人物{i + 1}"
            folder_path = os.path.join(out_root, folder)
            try:
                os.makedirs(folder_path, exist_ok=True)
            except Exception:
                show_glass_info(self, '错误', f"无法创建文件夹：{folder_path}")
                return

            for item in group['items']:
                src = item['image']
                if not os.path.isfile(src):
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
        if not items:
            return 0
        lines = ["$ws = New-Object -ComObject WScript.Shell"]
        for src, dst in items:
            lines.append(
                "$sc = $ws.CreateShortcut('{0}'); $sc.TargetPath = '{1}'; $sc.Save()"
                .format(dst.replace("'", "''"), src.replace("'", "''")))
        script = "\n".join(lines)
        ps1 = os.path.join(tempfile.gettempdir(), 'face_album_lnk.ps1')
        try:
            with open(ps1, 'w', encoding='utf-8-sig') as f:
                f.write(script)
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
        dlg = PersonDialog(self, group, idx)
        dlg.exec()


# ---------------------------------------------------------------------------
# 毛玻璃对话框（替代原生 MessageBox / InputDialog，风格统一）
# ---------------------------------------------------------------------------
class GlassDialog(QDialog):
    def __init__(self, title, parent=None, w=520):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedWidth(w)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        self.panel = GlassPanel(tint=QColor(28, 34, 44, 225),
                                border=QColor(186, 200, 216, 70), radius=14)
        outer.addWidget(self.panel)
        lay = QVBoxLayout(self.panel)
        lay.setContentsMargins(28, 24, 28, 22)
        lay.setSpacing(14)

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


def _dlg_label(text):
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lbl.setStyleSheet('color:rgba(245,247,250,230);font-size:13px;background:transparent;')
    return lbl


def show_glass_info(parent, title, text, w=520):
    dlg = GlassDialog(title, parent, w)
    dlg.body_lay.addWidget(_dlg_label(text))
    dlg.btns.addStretch(1)
    dlg.add_button('好', primary=True, on_click=dlg.accept)
    dlg.exec()


def show_glass_question(parent, title, text, w=560):
    dlg = GlassDialog(title, parent, w)
    dlg.body_lay.addWidget(_dlg_label(text))
    dlg.btns.addStretch(1)
    dlg.add_button('取消', primary=False, on_click=dlg.reject)
    dlg.add_button('确定', primary=True, on_click=dlg.accept)
    return dlg.exec() == QDialog.Accepted


def show_glass_input(parent, title, prompt, initial='', w=520):
    dlg = GlassDialog(title, parent, w)
    dlg.body_lay.addWidget(_dlg_label(prompt))
    edit = QLineEdit(initial)
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
class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._bg = load_background_center()
        self._drag = False
        self._off = None
        self.resize(800, 640)
        self.setMinimumSize(620, 500)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 24, 32, 26)
        outer.setSpacing(14)

        # 顶部标题 + 关闭
        head = QHBoxLayout()
        t = QLabel('使用帮助')
        t.setFont(QFont('HarmonyOS Sans SC', 20, QFont.Bold))
        t.setStyleSheet('color:white;background:transparent;')
        head.addWidget(t)
        head.addStretch(1)
        x = QPushButton('✕')
        x.setFixedSize(28, 28)
        x.setCursor(QCursor(Qt.PointingHandCursor))
        x.setStyleSheet('QPushButton{background:rgba(255,255,255,18);color:white;'
                        'border:none;border-radius:14px;}'
                        'QPushButton:hover{background:rgba(255,69,58,200);}')
        x.clicked.connect(self.accept)
        head.addWidget(x)
        outer.addLayout(head)

        # 功能与操作介绍
        func = QLabel(
            '■  功 能 介 绍\n'
            '· 人脸识别：自动检测照片里的人脸并提取特征；\n'
            '· 自动分组：同一人物自动聚成一组，不用一张张翻；\n'
            '· 查看详情：点击人物卡片，查看该人物的全部照片；\n'
            '· 命名人物：给人物起名字，导出时自动建同名文件夹；\n'
            '· 一键导出：按人物把照片复制或建快捷方式归档。')
        func.setStyleSheet('color:rgba(255,255,255,235);font-size:14px;'
                           'line-height:1.8;background:transparent;')
        func.setWordWrap(True)
        outer.addWidget(func)

        op = QLabel(
            '■  操 作 介 绍\n'
            '1. 点击「浏览」选择照片文件夹；\n'
            '2. 按需调整 eps 阈值 / 最少张数 / GPU 加速；\n'
            '3. 点击「开始聚类」等待识别完成；\n'
            '4. 完成后点击人物卡片查看全部照片；\n'
            '5. 点卡片上的「命名」给人物起名字；\n'
            '6. 选「复制」或「快捷方式」，点「导出相册」归档。')
        op.setStyleSheet('color:rgba(255,255,255,235);font-size:14px;'
                         'line-height:1.8;background:transparent;')
        op.setWordWrap(True)
        outer.addWidget(op)

        outer.addStretch(1)
        bottom = QLabel('照片全程在本机处理，不会上传，请放心使用。')
        bottom.setStyleSheet('color:rgba(255,255,255,140);font-size:12px;'
                             'background:transparent;')
        outer.addWidget(bottom)

    # 背景：中心裁切图（女孩与星河），cover 铺满圆角窗口
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 16, 16)
        p.setClipPath(path)
        if self._bg:
            s = self.size()
            pm = self._bg.scaled(s, Qt.KeepAspectRatioByExpanding,
                                 Qt.SmoothTransformation)
            x = (pm.width() - s.width()) // 2
            y = (pm.height() - s.height()) // 2
            p.drawPixmap(QRectF(0, 0, s.width(), s.height()), pm,
                         QRectF(max(0, x), max(0, y), s.width(), s.height()))
        else:
            p.fillRect(self.rect(), QColor(26, 30, 38))
        p.setClipping(False)
        p.setPen(QPen(QColor(186, 200, 216, 90), 1))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 16, 16)

    # 支持拖动
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = True
            self._off = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag and e.buttons() & Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self._off)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag = False
        super().mouseReleaseEvent(e)


class PersonDialog(QDialog):
    """毛玻璃详情窗口：网格展示某个人物的所有照片，点击打开原图。"""

    def __init__(self, parent, group, idx):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._group = group
        self.setModal(False)
        self.resize(1140, 780)
        self.setMinimumSize(760, 540)

        if group['type'] == 'unclassified':
            title = f"未分类 · 共 {len(group['items'])} 张照片"
        else:
            gname = group.get('name') or f"人物 {idx + 1}"
            title = f"{gname} · 共 {len(group['items'])} 张照片"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)

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

    @staticmethod
    def _open_full(path):
        try:
            os.startfile(path)
        except Exception:
            show_glass_info(None, '照片路径', path)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName('人脸聚类相册')
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
