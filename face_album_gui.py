# -*- coding: utf-8 -*-
# 上面这行叫“编码声明”：告诉 Python 这个源文件本身是用 UTF-8 写的（才能正确显示中文字符串）。
# 在 Python 3 里其实默认就是 UTF-8，写不写都行，老习惯保留。

# ============================================================================
# 【给只学过 C/C++ 的读者的总导读】
#
#  Python 和 C/C++ 的几个核心区别（先记住这些，读代码会顺很多）：
#
#  1. 没有类型声明。变量不需要“先声明再使用”，写了就存在：
#         int x = 5;            // C++
#         x = 5                 # Python —— 变量本身就是个“名字”，指向一个对象
#     同一个名字之后可以再指向别的类型：x = "hello" 也完全合法（动态类型）。
#
#  2. 代码块靠“缩进”而不是 {}。同一缩进 = 同一层，别混用空格和 Tab。
#
#  3. 不需要头文件。C++ 用 #include <xxx>，Python 用 import 模块名 引入别人写好的功能
#     （模块 ≈ 一个 .py 文件；包 ≈ 一个目录）。下面 import 一大堆就是“引入库”。
#
#  4. 最常用的“容器”不需要指针/迭代器手动管理：
#        list  [1,2,3]    ≈ 可动态增长的数组（像 std::vector，但啥类型都能塞）
#        tuple (1,2,3)    ≈ 不可变的“结构体打包”，常用来一次返回/传多个值
#        dict  {"name":"张三","age":18} ≈ 哈希表 / std::map
#        set   {'.jpg','.png'}         ≈ 哈希集合 / std::unordered_set
#     访问都是容器[下标或key]，例如 d["name"]、imgs[0]。索引从 0 开始，同 C。
#
#  5. 一切皆对象、且大多数都是“引用语义”。可以理解成：普通 Python 变量更像 C++ 的
#     智能指针/引用，赋值(=)基本是让两个名字指向同一个东西，真正“复制数据”
#     往往要 .copy() / 切片 / copy 库。函数传参也是传“引用”，改 list/dict 内容会
#     影响外部，但“重新给参数名赋值”不会（这一点最容易踩坑）。
#
#  6. 错误处理用 try/except，等价于 C++ 的 try/catch，但不用声明抛出什么。
#
#  7. for 循环是“遍历”：for x in 容器 逐个取出元素，而不是 C 的 for(i=0;...)。
#
#  8. 函数是“一等公民”，函数名可以直接当值传来传去（下面到处都是：按钮的回调、
#     线程的 target、排序的 key…… 全部是在“传函数”）。这有点像 C++ 的函数指针，
#     只是写法更自然。
#
#  9. 类用 class 定义，构造函数叫 __init__，第一个参数必须写 self（≈ C++ 的 this），
#     但调用时 Python 会自动帮你传，不用写。
#
#  10. “双下划线包起来”的成员名(__init__ / __main__ 等)不是随便命名，
#     它们是 Python 预留的“魔法方法/特殊变量”，看到就当关键字理解。
#
#  图像处理/识别会反复用到的三座大山：
#     numpy   —— 数组与矩阵计算的库，图像在 OpenCV 里本质就是一个 numpy 二维/三维数组
#     cv2     —— OpenCV 的 Python 接口，图像读入、裁剪、缩放、人脸框绘制等
#     PIL     —— Pillow 库，负责把 numpy 数组变成能塞进 Tk 图形界面的图片对象
#     sklearn / insightface —— 机器学习与现成人脸模型
# ============================================================================

# ---------------------------------------------------------------------------
# import：相当于 include，逐个说明本文件用到了谁、拿来干嘛
# ---------------------------------------------------------------------------
import os            # 操作系统功能：路径拼接、遍历目录、打开文件(startfile)等
import sys           # 解释器相关：这里没直接用，属常见标配导入
import re            # 正则表达式：导出相册时把不能当文件夹名的字符清掉
import shutil        # 文件操作：复制(shutil.copy2) / 移动(shutil.move) 原图
import threading     # 线程库。创建后台线程去跑耗时的人脸检测，避免界面“假死”
import queue         # 线程安全的消息队列：后台线程算完把结果塞进来，主线程取走更新界面
import urllib.request  # 联网下载：首次使用下载人脸模型、按需下载 GPU 加速组件
import json            # 解析 PyPI 接口，自动定位 onnxruntime-gpu 安装包地址
import zipfile       # 解压：把下载的模型/组件压缩包解开
import tkinter as tk  # Python 自带的 GUI 库(Tk)。下面所有窗口控件都是它提供的
from tkinter import ttk, filedialog, messagebox, simpledialog
# “from 模块 import 名字A, 名字B” 意思是：只把 A/B 直接拿进来用。
# 区别：
#   import tkinter as tk        -> 要用 tk.Tk()，前缀不能省（as 是给模块起个短别名）
#   from tkinter import ttk     -> 可以直接写 ttk.Button(...)
#   ttk   = 带样式的“高级控件”  filedialog = 系统“选文件夹”对话框  messagebox = 弹窗提示
import numpy as np   # np 是 numpy 的惯用别名。numpy 提供高性能数组与数学函数
import cv2           # OpenCV。注意 cv2.imread 读进来的图 = 一个 numpy 数组
from PIL import Image, ImageTk   # Pillow 图像库：缩略图缩放；ImageTk 把图转成 Tk 能显示的
from sklearn.cluster import DBSCAN   # 聚类算法：把“相似的人脸特征”自动归成同一个人
import onnxruntime   # onnxruntime：用来检测本机有没有 CUDA（NVIDIA 显卡）可用
from insightface.app import FaceAnalysis   # 现成人脸识别模型封装：检测脸+提取512维特征

# ---------- 模型根目录：优先 E 盘，回退用户目录 ----------
# 以下属于“模块级全局变量”，写在这层的缩进是 0（顶格），整个文件都能访问。
# C++ 里你写 const std::string kModelRoot = ...; Python 没有 const，全靠自觉不改它。
# 名字带前导下划线 _xxx 是一种约定俗成：告诉别人“这是模块内部用的，别在外面碰”。
_CANDIDATE_ROOTS = [          # 这是一个 list（列表）
    # 优先使用“程序所在目录”下的 .insightface（随软件走，方便移植）。
    # 打包成 exe 后 __file__ 指向临时解压目录，这里改用 exe 所在目录(sys.executable)。
    os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
                 else os.path.dirname(os.path.abspath(__file__)), '.insightface'),
    # 其次回退到当前用户主目录下的 .insightface（insightface 默认下载位置）
    os.path.join(os.path.expanduser('~'), '.insightface'),
    # os.path.expanduser('~') 会得到当前用户主目录，如 C:\Users\guigu
    # os.path.join(a,b) 是“跨平台拼路径”，自动加对的正斜杠/反斜杠，别自己写 '\\'
]
# next(迭代器, 默认值) 取出迭代器的第一个元素。整行意思：
#   在 _CANDIDATE_ROOTS 里找第一个“其下 models/buffalo_l 目录已存在”的路径，
#   找到了就用作模型根目录(MODEL_ROOT)；都找不到就用最后一个候选(即用户目录)。
# 这种“(表达式 for x in 列表 if 条件)”叫【生成器表达式】，可以理解成懒惰版的
# for 循环 + if，只在你真正“取”的时候才逐个算。等价 C++ 大概像把结果一个个 yield 出来。
MODEL_ROOT = next((r for r in _CANDIDATE_ROOTS
                   if os.path.isdir(os.path.join(r, 'models', 'buffalo_l'))),
                  _CANDIDATE_ROOTS[-1])
# os.path.isdir(...) == 判断是不是一个存在的目录  [-1] 是列表“倒数第一个”元素
# 下标支持负数！-1 末位，-2 倒数第二…… 这是 Python 特有，C++ 没有。

# 图片格式白名单：一个 set（集合）。用 in 判断“在不在里面”是 O(1)，天然去重。
SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ---------------------------------------------------------------------------
# ★ 联网下载相关（精简 exe 的关键：能下载的都不打包进 exe）：
#   · 人脸模型：不打包，首次使用时自动联网下载到本地（insightface 官方源）；
#   · GPU 加速组件：不打包，用户勾选 GPU 且本机没装 CUDA 时按需联网下载。
# 没网时程序会给出明确提示，不会假装能用。
# ---------------------------------------------------------------------------
MODEL_DL_URL = ('https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip')
# GPU 加速组件的下载来源：
#   · 留空 ''（默认）= 自动从 PyPI 官方下载与 exe 同版本的 onnxruntime-gpu wheel，
#     再下载 NVIDIA 官方的 CUDA/cuDNN 运行库（cublas / cudnn / cudart / cufft），
#     抽出 DLL 放进 onnxruntime 目录即可启用 GPU；
#   · 也可以填一个你自己打包好的“GPU 组件 zip”（内含全部所需 DLL），会优先用它。
GPU_RUNTIME_URL = ''

# onnxruntime 的 CUDA 运行库需要这几个 DLL 都齐全，GPU 才能用（对应 CUDA 12 / cuDNN 9）。
# 注意：cufft 不算在这——它可能装在系统其他位置，只在下载包里补上即可。
GPU_RUNTIME_DLLS = ('onnxruntime_providers_cuda.dll', 'cublas64_12.dll',
                    'cublasLt64_12.dll', 'cudart64_12.dll', 'cudnn64_9.dll')


def _nvidia_driver_present():
    """本机有没有 NVIDIA 显卡驱动：看 System32 里有没有 nvcuda.dll（驱动自带的加载器）。"""
    sr = os.environ.get('SystemRoot') or r'C:\Windows'
    return os.path.exists(os.path.join(sr, 'System32', 'nvcuda.dll'))


def _gpu_runtime_ready():
    """CUDA 运行库 DLL 是否已经齐全（放在 onnxruntime 的 capi 目录里）。"""
    capi = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi')
    return all(os.path.exists(os.path.join(capi, n)) for n in GPU_RUNTIME_DLLS)

# ---------------------------------------------------------------------------
# ★★★ 界面主题 —— 把“颜色”和“字体”集中定义成“命名常量”，统一样式、方便修改 ★★★
# ---------------------------------------------------------------------------
# 界面的“颜值”全靠下面这些变量。想改配色，只动这里几行，全界面一起变。
# 取名全大写 = 约定俗成地表示“这是常量，别在程序运行中改它”。
# ---------------------------------------------------------------------------
# 颜色用“十六进制”表示，格式是 #红红绿绿蓝蓝，每两位一个 0~255 的十六进制数：
#   #ff0000 红  #00ff00 绿  #0000ff 蓝  #ffffff 白  #000000 黑
# 可以用在线“取色器”自己挑喜欢的颜色，把值填进来即可。
# 配色走“温暖回忆”路线（暖杏/奶油色），这是一直定下的主题，不能换：
#   白奶油面板 + 温暖杏色主调，让人想起旧照片和老相册的氛围。
# 在“温暖”基础上尽量做干净、克制，去掉花哨装饰，贴近成熟相册软件。
C_BG          = '#faf7f2'   # 窗口/画布背景：极浅米色，比刺眼的纯白柔和
C_BG_DEEP     = '#f3eee8'   # 画布背景的“深一档”：浅暖灰，让白卡片浮起来
C_PANEL       = '#fffdf9'   # 面板/卡片底色：极浅奶油白
C_ACCENT      = '#c98d5e'   # 主色调（温暖杏色）：按钮、标题、进度条、悬停高亮
C_ACCENT_LT   = '#ddab80'   # 主色调的“浅一档”：鼠标悬停在按钮上时用
C_ACCENT_DK   = '#b37445'   # 主色调的“深一档”：按钮按下那一刻用
C_ACCENT_SOFT = '#f6ece2'   # 主色调的“极浅打底”：悬停背景、名言卡片底色
C_BORDER      = '#e6dbd0'   # 卡片/输入框的“淡边框”颜色
C_BORDER_DK   = '#d6c9bb'   # 稍深一点的边框/分隔线颜色
C_TEXT        = '#5c4a40'   # 主文字颜色：温暖深棕，比纯黑更耐看
C_TEXT_TITLE  = '#4a3830'   # 标题/大号文字：比正文更深，更有分量
C_TEXT_SUB    = '#9b8b7e'   # 次要文字颜色：暖灰棕，用于说明、提示
C_TEXT_DIS    = '#bfb0a6'   # 置灰文字：浅棕灰，用于禁用/占位提示
C_OK          = '#6ba383'   # 状态灯·绿：就绪/完成
C_RUN         = '#c98d5e'   # 状态灯·橙：正在处理
C_ERR         = '#d97a66'   # 状态灯·红：出错
C_WARN        = '#e8a76e'   # 状态灯·琥珀：警告
C_CARD_HOV    = '#c98d5e'   # 鼠标悬停在人物卡片上时，边框变成主色调（提示“可以点”）
# ---------------------------------------------------------------------------
# 字体：tuple (字体名, 字号, 可选样式)。ttk 控件按这个显示文字。
# 字号参照 VSCode：正文 13px 起，不费眼；小号说明也保持可读。
FONT      = ('Microsoft YaHei', 13)            # 默认正文：微软雅黑 13 号
FONT_BOLD = ('Microsoft YaHei', 13, 'bold')    # 加粗：卡片标题等
FONT_SMALL = ('Microsoft YaHei', 11)           # 小号：文件名等次要文字
FONT_TITLE = ('Microsoft YaHei', 22, 'bold')   # 顶栏大标题
FONT_SUB   = ('Microsoft YaHei', 11)           # 顶栏副标题
# 名言用的“艺术字体”：华文行楷是行书笔画的毛笔感字体，写名言很有味道。
# 系统没有这个字体时 Tk 会自动退回默认字体，不影响运行。
FONT_QUOTE = ('华文行楷', 20)

# 网格每行放几张卡片（两个窗口共用）
GRID_COLS = 4

# 聚类等待时在状态栏下方轮播的“名言”（像游戏加载界面那样提神）
# 主题：珍视回忆、珍视友谊。每句轮着显示，聚类结束自动消失。
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

# 按“人脸总数 N”分三档的内存保护阶梯（N 越多内存压力越大，保护层层叠加）：
#   量少  N ≤ FACE_TIER_MID    只做分块计算（零精度损失，恒开）
#   中等  FACE_TIER_MID < N ≤ FACE_TIER_HUGE
#                            再加 float16 半精度存特征（内存减半，近无损）
#   巨大  N > FACE_TIER_HUGE   再加“确认闸门”：先弹窗问用户，同意才继续
FACE_TIER_MID   = 30000     # 超过它，特征改用 float16 存（每张脸 2KB→1KB）
FACE_TIER_HUGE  = 150000    # 超过它，聚类前必须先让用户确认，防止内存爆掉
SIM_CHUNK       = 4096      # 相似度矩阵分块的行数：分块算，省内存且结果不变


# ============================================================================
# 函数 1：收集文件夹里所有图片路径
# ============================================================================
def collect_images(root):
    """递归遍历文件夹，收集所有受支持格式的图片路径"""
    # 三引号字符串紧跟在函数/类后面的，叫 docstring（文档注释），可用 help() 查看，
    # 同时也是给读者看的说明。C++ 里没有对应物，就当规范化注释。
    imgs = []                    # 空 list，准备往里装结果（类似 vector<string>）
    # os.walk(root) 是一个“生成器”：每次吐出一个三元组 (当前目录, 子目录列表, 文件列表)，
    # 它会自动一层层往下钻（递归遍历目录树）。
    # for dirpath, _, files in ... 这种叫“解包赋值”：把三元组一次性拆给三个变量。
    # 变量名叫 _ 是约定俗成：表示“这个位置的值我不需要”。
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):            # sorted() 返回排好序的新列表（字符串默认按字母）
            # os.path.splitext(f) -> (不含扩展名的部分, 扩展名) 比如 ('a','.jpg')
            # [1] 取第二个元素也就是扩展名；.lower() 转小写，这样 .JPG 也能命中白名单
            if os.path.splitext(f)[1].lower() in SUPPORTED:  # 在集合里吗？
                imgs.append(os.path.join(dirpath, f))        # 是 -> 把完整路径收进列表
    return imgs                  # 返回装满路径的 list


# ============================================================================
# 函数 2：专门“能读中文路径”的读图函数
# ============================================================================
def read_image(path):
    """兼容中文/含特殊字符路径的图片读取。

    cv2.imread 在 Windows 上内部用 ANSI 编码的 fopen 打开文件，
    遇到含中文的文件名或目录（UTF-8）会打开失败，打印
    "loadsave.cpp cv::findDecoder ... can't open/read" 警告并返回 None，
    导致整批中文名的照片都被当成"读不了/没脸"。
    这里先用 Python 内置 open 读字节（底层走宽字符 API，中文没问题），
    再交给 cv2.imdecode 解码。
    """
    # with ... as f: 叫“上下文管理器”，作用 ≈ C++ 的 RAII：进块自动 open，
    # 出块(无论是否异常)自动 close 关文件。省得你手写 close，也不怕忘。
    # 'rb' = read binary，以“二进制”方式读取。图片就是一堆字节，读成 bytes。
    with open(path, 'rb') as f:
        data = f.read()      # 一次性把整个文件内容读到内存（bytes，类似 C 的 char 数组）
    # np.frombuffer(data, dtype=np.uint8)：
    #   把那一坨字节“零拷贝”地看成 numpy 的 uint8 一维数组（不复制，只换个视角）。
    #   dtype=np.uint8 说明每个元素是 8 位无符号整数(0~255)，对应 C++ 的 unsigned char。
    # cv2.imdecode(数据, cv2.IMREAD_COLOR)：从“内存里的字节”解码成图片(numpy 数组)。
    #   绕开按“文件名”打开，文件名本身就不参与解码，所以中文路径没问题。
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img
    # 图片在 OpenCV 里 = 一个 numpy 三维数组，形状是 (高, 宽, 3)：
    #   img[h, w, 0] = 蓝色通道, [1] 绿色, [2] 红色。注意是 BGR 不是 RGB！OpenCV 反着排。


# ============================================================================
# 函数 3：根据人脸框，从原图里裁剪出人脸区域并做成正方形缩略图
# ============================================================================
def make_face_thumb(image_path, bbox, size=(180, 180)):
    """根据人脸框裁剪并生成缩略图（居中到固定画布），失败返回 None"""
    # 形参 size 有默认值 (180,180)：调用时不传就用默认。C++ 的默认参数差不多。
    try:                       # try = C++ 的 try。可能出错的代码放这里
        img = read_image(image_path)   # 读整张原图（numpy 数组，3维：高H、宽W、3通道）
        if img is None:        # None ≈ C 的 NULL/空指针，表示“没读到东西”
            return None  # 图片无法读取
        h, w = img.shape[:2]   # img.shape 返回 (高,宽,通道数)；[:2] 是“切片”取前两个
        # 切片 list[a:b]：取下标 a(含) 到 b(不含)。a 可省=从头，b 可省=到尾。
        # 这里 shape[:2] == 取 shape[0], shape[1]，即高 h 和宽 w。
        # 人脸框 bbox 是 4 个数的数组 (x1, y1, x2, y2)，分别是框的左上角、右下角坐标。
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # [int(v) for v in bbox] 叫【列表推导式】≈ 一行写成的 for+append：
        #   等价于：
        #     tmp = []
        #     for v in bbox: tmp.append(int(v))
        #   int(v) 把可能的小数转成整数下标（C++ 的 static_cast<int>）。
        pad = int((x2 - x1) * 0.35)   # 把人脸框四周再外扩 35%，把人脸“周围留点边”
        # max/min 与 C++ 相同，但参数是(值1,值2)。这里用它们做“夹取”，防止越界：
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)   # 不能小于 0（图片左上角）
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)   # 不能大于宽/高（图片右下角）
        # Python 允许一行写多句，用分号 ; 隔开。不推荐但偶尔能看到。
        crop = img[y1:y2, x1:x2]
        # ★ 图像裁剪就是这么一行：img[行范围, 列范围]。
        #   C++ 你得写三层 for 循环搬像素；numpy 直接切片拿到一块“视图”(相当于子数组引用)。
        #   注意顺序是先“行”(竖直方向 y) 后“列”(水平方向 x)，和数学坐标习惯相反。
        if crop.size == 0:      # size = 元素总数。宽或高为 0 时这里会是 0
            crop = img          # 极端情况：裁剪区域为空就退回整张图
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # 颜色空间转换：OpenCV 内部是 BGR，Pillow/Tk 要 RGB。这里把 BGR 通道重排成 RGB。
        pil = Image.fromarray(crop)
        # 把 numpy 数组转成 Pillow 的 Image 对象（不复制，共享同一块内存，只是换解释方式）
        pil.thumbnail(size, Image.Resampling.LANCZOS)
        # thumbnail = 等比缩小图片，使它能放进 size 这个框内。LANCZOS 是高画质缩放算法。
        canvas = Image.new('RGB', size, (235, 235, 235))
        # 新建一块“空画布”，统一尺寸(如180x180)，浅灰背景(235,235,235)。缩放后图居中贴上面。
        canvas.paste(pil, ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2))
        # paste = 把缩略图贴到画布的 (x偏移, y偏移) 处；// 是整除(向下取整)，用来居中。
        return ImageTk.PhotoImage(canvas)
        # 再转成 Tk 的 PhotoImage 才能放进图形界面的 Label 里显示。这就是函数返回的“成品”。
    except Exception:            # except = C++ 的 catch；Exception 是“所有异常的基类”
        return None   # 任何异常(文件损坏、格式不对…)都返回空，不影响主流程


# ============================================================================
# ★ 聚类升级：从“一把尺子量到底”变成“先粗分 + 认老大 + 认亲合并” ★
#
# 老办法是：给一个 eps 阈值，一把尺子量所有脸，够近就归一组（DBSCAN）。
# 问题：有人照片多、有人照片少，光线角度还五花八门，一把尺子容易误判。
#
# 新办法学华为相册的思路，分三步走：
#   1) 先粗分：还是用 DBSCAN 大概分一下，把明显是同一个人的先拢成几堆；
#   2) 认老大：每堆照片算一张“平均脸”当这堆的“老大”，
#      然后每张脸重新跟最像的老大认亲——不够像的踢出去，串堆的纠正回来；
#   3) 认亲合并：两个堆的老大长得太像（比如同一个人不同年龄段被拆成两堆），
#      就把这两堆合成一堆。
# 以上 2、3 两步反复做几遍，结果越修越稳。
# ============================================================================

def _l2_normalize(emb):
    """把每一行向量都拉成“长度=1”的单位向量。

    两个单位向量的点积 = 它们的余弦相似度，也就是“这俩脸像不像”的分数，
    范围在 -1~1 之间，越接近 1 越像。后面到处都要用相似度，先归一化省事。
    """
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


def _group_centroids(emb, labels):
    """给每个群算一张“平均脸”（centroid），当这群的“老大”。

    平均脸 = 群里所有人脸特征向量求平均，再归一化。
    它比单张照片“稳”：一个人的平均脸基本就代表这个人长啥样。
    返回两个东西：
      centroids: {群号: 平均脸向量}
      groups:    {群号: [群里每个人脸的下标, ...]}
    """
    groups = {}
    for i, lab in enumerate(labels):
        if lab < 0:                      # -1 是没归属的“散脸”，先不拿它当老大
            continue
        groups.setdefault(int(lab), []).append(i)
    centroids = {}
    for lab, idxs in groups.items():
        mean = emb[idxs].mean(axis=0)                 # 取平均
        norm = np.linalg.norm(mean) + 1e-9            # 防止除以 0
        centroids[lab] = mean / norm                  # 再归一化
    return centroids, groups


def _reassign_to_nearest(emb, labels, centroids, min_score, qualities=None):
    """“认老大”这一步：让每张脸重新认领一个最像的群。

    对每张脸：
      - 算出它和每个老大（平均脸）的相似度，挑分数最高的那个；
      - 如果最高分还够不到门槛（“像到一定程度才算一家人”），
        就把它标成 -1（散脸），最后进“未分类”；
      - 够得着就归到那个老大名下（包括把 DBSCAN 漏掉的散脸也捡回来）。
    这样能纠正 DBSCAN 错分、漏分的脸。

    门槛不是一把尺子量到底：如果给了每张脸的“清楚分”（qualities），
    越模糊的脸门槛就越放宽一点——因为模糊脸的特征本身就不准，
    还拿和清晰脸一样的标准要求它，很容易把它冤枉成“散脸”。
    """
    lab_list = list(centroids.keys())
    if not lab_list:
        return np.full(len(emb), -1, dtype=int)
    # 把所有老大排成矩阵 C：每行一个老大，和 emb 一乘就一次算出所有相似度
    C = np.stack([centroids[l] for l in lab_list])        # (群数, 512)
    # ★分块算相似度：不一次生成 (脸数, 群数) 的大矩阵，按 SIM_CHUNK 行一批批算。
    #   数学结果和一次性 @ 完全一样，只是峰值内存从“整张脸×群”降到“一块×群”。
    n_faces = len(emb)
    sims = np.empty((n_faces, len(lab_list)), dtype=np.float32)
    for s in range(0, n_faces, SIM_CHUNK):
        e = min(s + SIM_CHUNK, n_faces)
        sims[s:e] = emb[s:e] @ C.T
    best = sims.argmax(axis=1)                            # 每张脸最像哪个老大
    best_score = sims[np.arange(len(emb)), best]          # 那个老大给的分数

    # 默认所有人用同一个门槛 min_score
    thresholds = np.full(len(emb), min_score)
    if qualities is not None:
        # 清楚分越低的（越模糊），门槛往下放得越多，最多放宽 0.1
        # 例：清楚分 1.0 的门槛不变；清楚分 0.5 的门槛放宽 0.05
        relax = 0.10
        thresholds = min_score - (1.0 - np.asarray(qualities)) * relax

    # 够得着各自门槛的归老大，够不着的一律当散脸
    new_labels = np.where(best_score >= thresholds,
                          np.array([lab_list[b] for b in best]), -1)
    return new_labels.astype(int)


def _merge_close_groups(emb, labels, centroids, merge_score):
    """“认亲合并”这一步：两个群的老大长得太像，就合成一个群。

    为什么需要？同一个人不同年龄段的照片、或光线角度差很多的照片，
    第一轮可能被拆成两堆。但两堆的平均脸依然很接近——
    只要接近到 merge_score 这个程度，就认定是同一个人，并堆。
    注意：merge_score 比认老大用的 min_score 更宽松（更愿意合并），
    专门用来“认亲”，把被拆开的同一人拼回来。
    """
    lab_list = list(centroids.keys())
    if len(lab_list) < 2:
        return labels
    C = np.stack([centroids[l] for l in lab_list])        # (群数, 512)
    # ★分块算“群×群”相似度：群很多时也不一次生成 (群数, 群数) 大矩阵，
    #   按 SIM_CHUNK 行一批批算，峰值内存从 O(群²) 降到 O(块×群)。
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
                # 把 b 群里的人全部并进 a 群
                new_labels[new_labels == small] = big
    return new_labels.astype(int)


def _face_quality_score(image, bbox, det_score=1.0):
    """给一张人脸打一个“清楚程度”分，0~1，越接近 1 越清楚。

    为什么要打这个分？因为照片有清晰有模糊：清晰的脸特征很准，
    模糊的脸（对焦差/手抖/像素低）特征本身就“毛”了。
    如果两种脸用同一把尺子去比，模糊脸很容易认错人或者没人要。
    所以我们要给每张脸估个清楚分，越模糊后面聚类时标准越放宽。

    怎么算？三个线索加在一起（都压到 0~1）：
      1) 锐利度：把脸抠出来用 Laplacian 算子看“边缘锐利度”，
         照片越清楚边缘越锐利，这个数值越大；
      2) 人脸像素占比：脸在整张照片里占多大（宽×高 / 原图宽×高）。
         脸越大，能用的有效细节越多；脸小到快贴不上，信息天生不足；
      3) 检测置信度：人脸检测给的 det_score，模型对这张脸越有把握越高。

    权重：锐利度最要紧占 4 成，人脸大小其次占 3 成，置信度垫底占 3 成。
    """
    h, w = image.shape[:2]                 # 原图高、宽
    x1, y1, x2, y2 = [int(v) for v in bbox]  # 人脸框四角坐标
    x1 = max(0, x1); y1 = max(0, y1)          # 防止坐标越界，夹到图内
    x2 = min(w, x2); y2 = min(h, y2)
    crop = image[y1:y2, x1:x2]                # 把人脸那块抠出来
    if crop is None or crop.size == 0:
        return 0.0                            # 抠空了就当最模糊

    # 1) 锐利度：Laplacian 方差，越大越清楚。没有上限，经验上 500 已算很清晰
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)   # 转成灰度图才好算边缘
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(1.0, laplacian_var / 500.0)

    # 2) 人脸像素占比：脸面积 / 整张图面积。一般占 1% 以上就是很清晰的大脸了
    face_area = (x2 - x1) * (y2 - y1)              # 脸框有多少像素
    img_area = max(1, h * w)                       # 整张图有多少像素
    area_ratio = face_area / img_area
    size_score = min(1.0, area_ratio / 0.01)       # 占比 1% 就算满分

    # 3) 三合一：锐利度 4 成 + 人脸大小 3 成 + 检测置信度 3 成
    return max(0.0, min(1.0, 0.4 * sharpness + 0.3 * size_score + 0.3 * det_score))


def smart_cluster(emb, eps, min_cluster, max_iters=3, qualities=None):
    """聚类总入口：先粗分，再反复“认老大 + 认亲”几遍。

    参数：
      emb         已经归一化的人脸特征矩阵 (脸数, 512)
      eps         用户填的阈值（距离），越小越严
      min_cluster 最少几张同脸才算一个“人物”
      qualities   每张脸的“清楚分”（0~1），可选。给了的话，
                  模糊的脸认老大时门槛自动放宽，不容易被冤枉成散脸。
    返回：
      labels      一维数组，第 i 个元素是第 i 张脸归的群号；-1 = 未分类
    """
    # 第一轮粗分：DBSCAN。因为输入已归一化，metric='cosine' 时
    # 距离 = 1 - 相似度，所以 eps 正好等于“相似度多少才算一家人”的临界。
    # ★内存保护：cosine 不能进 BallTree，sklearn 会退化出“全脸×全脸”的 N² 大矩阵
    #   （10 万张脸 ≈ 40GB，直接爆内存）。对归一化向量，欧氏距离与余弦距离
    #   单调等价：d_e = √(2·d_c)。所以换成 metric='euclidean' + eps'=√(2·eps)，
    #   聚类结果【完全一样】，但能走 BallTree 树搜索，内存降到 O(N·logN)。
    cl = DBSCAN(eps=np.sqrt(2.0 * eps), min_samples=min_cluster,
                metric='euclidean', algorithm='ball_tree')
    labels = cl.fit_predict(emb)

    # 把距离阈值换算成相似度阈值：
    #   min_score  = 认老大要“至少像多少”；
    #   merge_score = 认亲要“至少像多少”，比 min_score 更宽松（更愿意合并）。
    min_score = 1.0 - eps
    merge_score = 1.0 - eps * 1.2        # 默认比 eps 再放宽 20%，方便认亲

    # 反复“认老大 → 认亲 → 再认老大”，每轮都会把结果修得更准一点
    for _ in range(max_iters):
        centroids, _ = _group_centroids(emb, labels)      # 先算出各群老大
        if not centroids:
            break
        labels = _merge_close_groups(emb, labels, centroids, merge_score)
        centroids, _ = _group_centroids(emb, labels)      # 合并完重算老大
        labels = _reassign_to_nearest(emb, labels, centroids,
                                      min_score, qualities)

    # 最后一道保险：群太小（没凑够 min_cluster 张）的不算人物，全丢进散脸
    centroids, groups = _group_centroids(emb, labels)
    for lab, idxs in groups.items():
        if len(idxs) < min_cluster:
            labels[np.array(idxs)] = -1
    return labels


# ============================================================================
# 类 FaceAlbumGUI：整个图形界面的“主控制器”
# ============================================================================
# class ≈ C++ 的 class。成员变量/成员函数都写在它里面。
# 没有 public/private 关键字；所有成员默认“公开”，靠下划线前缀约定私密程度。
#   self._xxx / self.__xxx：单下划线=内部用；双下划线=更私密(Python会自动改名)。
class FaceAlbumGUI:
    # __init__ 是“构造函数”，创建对象时自动被调用。参数 self 就是“这个对象自己”(≈this)。
    def __init__(self, root):
        # 后面所有 self.xxx 都是“这个对象的成员变量”，离开对象就共享给所有方法用。
        # C++ 你在构造函数里初始化成员，这里一样。
        self.root = root                  # 保存主窗口对象，方便后续操作窗口
        root.title("本地人脸聚类相册")     # 设置窗口标题（方法调用，不用写 return）
        root.geometry("1240x840")         # 窗口初始大小：宽x高，单位像素
        root.minsize(980, 640)            # 窗口最小尺寸，防止被拖太小
        root.configure(bg=C_BG)           # ★美化：把整个窗口背景刷成我们的主题浅色

        # ------------------------------------------------------------------
        # 状态：Tk 的“变量对象”。作用是把“输入框的内容/勾选框的状态”和界面双向绑定。
        # 你改 .set(值)，输入框立即显示；用户在输入框打字，.get() 就能取到。
        # （普通 Python 变量没这个“通知界面”的能力，所以 GUI 里要用这些 *Var）
        # ------------------------------------------------------------------
        self.input_dir = tk.StringVar()                 # 装“照片文件夹路径”的文本框内容
        self.eps = tk.DoubleVar(value=0.43)             # 聚类阈值 eps，浮点，默认 0.43
        self.min_cluster = tk.IntVar(value=2)           # 最少照片数，整数，默认 2
        # ★ 开机探测显卡情况（不靠 get_available_providers，那只看编译选项不可靠）：
        #   有没有 NVIDIA 驱动(nvcuda.dll) + CUDA 运行库 DLL 是否齐全。
        #   A 卡/核显/无独显 = 没驱动 → 自动禁用 GPU 选项，全程 CPU，绝不报错。
        self._nvidia_driver = _nvidia_driver_present()
        self._gpu_runtime = _gpu_runtime_ready()
        self._cuda_available = self._nvidia_driver and self._gpu_runtime
        self.use_gpu = tk.BooleanVar(value=self._cuda_available)  # “GPU加速”勾选框：有显卡才默认勾
        self.status_text = tk.StringVar(value="就绪 —— 请选择照片文件夹")  # 底部状态栏文字
        self.export_mode = tk.StringVar(value='copy')   # 写入方式：'copy'=复制 / 'move'=移动原图
        self.app = None              # insightface 模型对象。None 表示“还没加载”
        #                               空值占位，第一次真正用之前再建(见 _cluster_worker 懒加载)
        self.groups = []             # 聚类结果。每个元素是 {'type':..., 'items':[...], 'name':...}
        #                               这是个 list，将来装 dict。dict 以 {键:值} 存取
        #                               'name' 是用户给人物起的名字（未命名则为空）
        self._busy = False           # 是否正有一轮聚类在后台跑。
        #                              防止用户反复点“开始聚类”叠加出多个后台线程——
        #                              多个线程同时读高清大图 + 并发调用模型，内存成倍叠加，
        #                              最终被系统杀掉闪退。这个开关保证“同时只有一轮”。
        self._thumb_refs = []        # 保存缩略图引用，防被垃圾回收(GC)提前清掉(见下)
        self._empty = None           # 画布上的“空状态”占位提示（无结果时居中显示）
        self._quote_idx = 0          # 名言轮播的下标（等待聚类时在状态栏下方滚动显示）
        self._quote_label = None     # 名言标签控件；空闲/结束时隐藏（见 _build_ui、_cluster_finished）
        self._quote_after_id = None  # 名言轮播定时器的编号，结束时要取消它防止残留
        self._quote_seq = 0          # 名言动画的“世代”：每次新动画 +1，用来作废旧动画帧
        self._confirm_event = threading.Event()   # “大量人脸确认闸门”的唤醒开关（线程间握手）
        self._confirm_ok = False      # 用户对确认闸门的回答：True=同意继续 / False=取消
        self.msg_queue = queue.Queue()   # 线程安全队列：后台线程->界面线程传消息用
        # queue.Queue() 内部带锁，多线程往里 put / get 都不会乱，不用像 C++ 那样自己加 mutex。

        self._apply_theme()                   # ★美化：先统一给所有控件“换皮肤”（见下面方法）
        self._build_ui()                      # 搭界面(下面定义的方法)
        self.root.after(120, self._poll_queue)
        # root.after(毫秒, 函数)：让 Tk 在 120ms 后调用 _poll_queue。
        # GUI 是“事件循环”模型：不是顺序执行到底，而是空闲时不断处理事件。
        # 这行相当于给自己定了个“每 120ms 就来一次”的闹钟(因为 _poll_queue 末尾又重复 after 自己)

    # ---------------- ★美化：主题/皮肤设置（整个界面好不好看的关键） ----------------
    # 为什么 ttk 控件默认那么“土”？
    #   因为 Windows 上 ttk 默认用系统自带的 vista 主题，长什么样是操作系统说了算，
    #   我们很难改它的颜色。解决办法是 theme_use('clam') —— clam 是一个“可自由
    #   上色”的主题（跨平台都有），换过去之后，所有颜色就都能由我们控制了。
    def _apply_theme(self):
        """统一配置全局配色：字体、背景、按钮、输入框、进度条、滚动条、卡片边框。"""
        style = ttk.Style(self.root)          # 创建“样式管理器”，负责给控件换皮
        style.theme_use('clam')               # 换成可上色的 clam 主题（关键一步）

        # 给“所有 ttk 控件”('.' 表示全局默认)统一字体 + 前景/背景色。
        # 这样大部分控件不用一个个去设置，风格天然统一。
        style.configure('.', font=FONT,
                        background=C_BG, foreground=C_TEXT)

        # --- 普通按钮：白底、淡灰边框，悬停变浅，克制不花哨 ---
        style.configure('TButton',
                        background=C_PANEL,      # 底色白色
                        foreground=C_TEXT,       # 文字近黑
                        bordercolor=C_BORDER,    # 边框淡灰
                        borderwidth=1,
                        padding=(16, 7),         # (左右, 上下) 内边距，够大不挤
                        focuscolor=C_ACCENT)     # 键盘焦点时的高亮色
        # style.map(样式名, 状态=颜色列表)：定义“不同状态”下的颜色。
        style.map('TButton',
                  background=[('active', C_BG), ('pressed', C_BG_DEEP)],
                  bordercolor=[('active', C_ACCENT), ('focus', C_ACCENT)])

        # --- 主按钮（“开始聚类”）：主色填充白字，整个界面唯一的重色块 ---
        style.configure('Primary.TButton',
                        background=C_ACCENT,     # 主色蓝底
                        foreground='#ffffff',    # 白字
                        bordercolor=C_ACCENT,
                        borderwidth=0,           # 无边框，纯色块更现代
                        padding=(22, 8),
                        font=FONT_BOLD)          # 加粗，突出它是主要操作
        style.map('Primary.TButton',
                  background=[('active', C_ACCENT_LT),   # 悬停变浅
                              ('pressed', C_ACCENT_DK),  # 按下变深
                              ('disabled', C_BORDER_DK)],  # 禁用时变灰
                  foreground=[('disabled', '#ffffff')])

        # --- 输入框：白底、淡边框、聚焦时边框变主色（给人“我在编辑它”的提示） ---
        style.configure('TEntry',
                        fieldbackground='#ffffff',  # 输入区底色
                        foreground=C_TEXT,
                        bordercolor=C_BORDER,
                        lightcolor=C_BORDER,        # clam 主题里边框由这俩控制
                        darkcolor=C_BORDER,
                        padding=5)
        style.map('TEntry',
                  bordercolor=[('focus', C_ACCENT)],   # 获得焦点 -> 边框变主色
                  lightcolor=[('focus', C_ACCENT)],
                  darkcolor=[('focus', C_ACCENT)])

        # --- 勾选框 / 单选框：文字颜色对齐主题 ---
        style.configure('TCheckbutton', background=C_PANEL, foreground=C_TEXT)
        style.map('TCheckbutton',
                  background=[('active', C_PANEL)],   # 悬停时别变色，保持白色面板
                  foreground=[('active', C_ACCENT)])  # 但文字变成主色，仍然有反馈
        style.configure('TRadiobutton', background=C_PANEL, foreground=C_TEXT)
        style.map('TRadiobutton',
                  background=[('active', C_PANEL)],
                  foreground=[('active', C_ACCENT)])

        # --- 面板：白色圆角面板样式，承载各操作区 ---
        style.configure('Panel.TFrame', background=C_PANEL)
        style.configure('Panel.TLabel', background=C_PANEL, foreground=C_TEXT)

        # --- 进度条：满格主色，轨道浅灰，细一点更精致 ---
        style.configure('Horizontal.TProgressbar',
                        background=C_ACCENT,          # 已填充颜色
                        troughcolor=C_BG_DEEP,        # 轨道(底槽)颜色
                        bordercolor=C_BG_DEEP,
                        thickness=10)                 # 条的高度(像素)

        # --- 滚动条：浅灰滑块，和页面融为一体 ---
        style.configure('Vertical.TScrollbar',
                        background=C_BORDER_DK,       # 滑块颜色
                        troughcolor=C_BG,             # 轨道颜色
                        bordercolor=C_BG,
                        arrowcolor=C_PANEL,
                        width=12)

    # ---------------- UI 构建 ----------------
    # 方法名以 _ 开头表示“内部方法，外界别直接调”。self 永远在第一位。
    def _build_ui(self):
        """搭建窗口的全部控件：顶栏、参数栏、状态栏/进度条、可滚动网格区"""
        # ------------------------------------------------------------------
        # 布局三兄弟：pack / grid / place。
        # 本文件只用 pack(顺序摆放) 和 grid(表格摆放)。控件先创建，再 pack/grid 到父容器。
        # 例如 ttk.Button(...).pack(...)：先造按钮，立刻 pack 到父容器并指定靠边/内边距。
        # ------------------------------------------------------------------
        self._build_header()     # 顶部的标题横幅（单独一个方法，保持 _build_ui 清爽）

        # 工具条：一个白色面板容器，把各种参数控件收在一起
        top = ttk.Frame(self.root, style='Panel.TFrame', padding=(18, 14))
        top.pack(fill=tk.X, padx=16, pady=(0, 10))   # padx/pady 让它四周留点边，像浮起的卡片

        # 第一行：文件夹选择（左） + 开始聚类（右，主操作）
        row1 = ttk.Frame(top, style='Panel.TFrame')
        row1.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(row1, text="照片文件夹", style='Panel.TLabel').pack(side=tk.LEFT)
        # Entry = 单行输入框。textvariable 绑定到 self.input_dir，双向同步
        ttk.Entry(row1, textvariable=self.input_dir, width=46).pack(side=tk.LEFT, padx=8)
        ttk.Button(row1, text="浏览", command=self._choose_dir).pack(side=tk.LEFT)
        # 主按钮放右侧：现代应用的习惯是“主要操作靠右”，一眼能看到
        self._start_btn = ttk.Button(row1, text="开始聚类", style='Primary.TButton',
                                     command=self._start_cluster)
        self._start_btn.pack(side=tk.RIGHT)

        # 第二行：左边参数设置，中间分隔线，右边导出设置，最右帮助按钮
        row2 = ttk.Frame(top, style='Panel.TFrame')
        row2.pack(fill=tk.X)
        # -- 参数组 --
        g1 = ttk.Frame(row2, style='Panel.TFrame')
        g1.pack(side=tk.LEFT)
        ttk.Label(g1, text="参数", style='Panel.TLabel', font=FONT_SMALL,
                  foreground=C_TEXT_SUB).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(g1, text="eps 阈值", style='Panel.TLabel').pack(side=tk.LEFT)
        ttk.Entry(g1, textvariable=self.eps, width=5).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(g1, text="最少张数", style='Panel.TLabel').pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(g1, textvariable=self.min_cluster, width=4).pack(side=tk.LEFT, padx=(4, 0))
        # Checkbutton = 勾选框。用 tk 原生控件，勾选显示标准对勾 ✓，一目了然。
        # 三种情况给出不同提示：能用 / 有 NVIDIA 但缺运行库（可联网下载）/ 没有 NVIDIA。
        self._gpu_check = tk.Checkbutton(g1, text="GPU 加速", variable=self.use_gpu,
                                         bg=C_PANEL, fg=C_TEXT, activebackground=C_PANEL,
                                         activeforeground=C_TEXT, selectcolor=C_PANEL,
                                         font=FONT)
        if self._cuda_available:
            pass
        elif self._nvidia_driver:
            self._gpu_check.config(fg=C_TEXT_SUB, text="GPU 加速（需联网下载运行库）")
        else:
            self._gpu_check.config(fg=C_TEXT_SUB, text="GPU 加速（本机无 NVIDIA 显卡）")
        self._gpu_check.pack(side=tk.LEFT, padx=(10, 0))
        # -- 分隔线 --
        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=14, pady=2)
        # -- 导出组 --
        g2 = ttk.Frame(row2, style='Panel.TFrame')
        g2.pack(side=tk.LEFT)
        ttk.Label(g2, text="导出", style='Panel.TLabel', font=FONT_SMALL,
                  foreground=C_TEXT_SUB).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Radiobutton(g2, text="复制", value='copy',
                        variable=self.export_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(g2, text="移动原图", value='move',
                        variable=self.export_mode).pack(side=tk.LEFT, padx=(0, 8))
        self._export_btn = ttk.Button(g2, text="导出相册", command=self._export_album)
        self._export_btn.pack(side=tk.LEFT)
        ttk.Label(g2, text="未命名的人物将导出为“人物N”", style='Panel.TLabel',
                  font=FONT_SMALL, foreground=C_TEXT_SUB).pack(side=tk.LEFT, padx=(8, 0))
        # -- 帮助按钮，最右侧 --
        ttk.Button(row2, text="?", width=3,
                   command=self._show_help).pack(side=tk.RIGHT)

        # 一行小字提示：点击卡片可查看该人物全部照片，弱化显示不抢戏
        ttk.Label(top, text="提示：点击人物卡片可查看该人物的全部照片；点击照片可用系统默认程序打开原图。",
                  style='Panel.TLabel', font=FONT_SMALL,
                  foreground=C_TEXT_SUB).pack(fill=tk.X, pady=(8, 0))

        # 状态栏(一行文字)+进度条。anchor=tk.W 文字左对齐；progress 是横条进度条
        status_panel = ttk.Frame(self.root, style='Panel.TFrame', padding=(18, 12))
        status_panel.pack(fill=tk.X, padx=16, pady=(0, 10))
        ttk.Label(status_panel, textvariable=self.status_text, anchor=tk.W,
                  style='Panel.TLabel', foreground=C_TEXT).pack(fill=tk.X)
        self.progress = ttk.Progressbar(status_panel, mode='determinate')
        # mode='determinate'：进度条有一个确定的最大值和当前值，用来显示 n/m
        self.progress.pack(fill=tk.X, pady=(8, 0))

        # ★名言：等待聚类时像游戏加载界面那样在进度条下方轮播一句名言。
        #   淡蓝色底 + 楷体居中，做成一张克制的名言卡片，贴合干净主题。
        #   平时不 pack（隐藏）；_next_quote 显示它，_cluster_finished 再收回去。
        self._quote_box = tk.Frame(status_panel, bg=C_ACCENT_SOFT)   # 淡蓝底板
        self._quote_label = tk.Label(self._quote_box, text="", bg=C_ACCENT_SOFT,
                                     fg=C_ACCENT_DK, font=FONT_QUOTE,
                                     anchor=tk.CENTER, justify=tk.CENTER,
                                     wraplength=920)
        # wraplength：名言太长时自动换行，不至于把窗口撑爆
        self._quote_label.pack(fill=tk.X, padx=16, pady=8)

        # 主区域：可滚动画布。Canvas 是 Tk 里能“放东西还能滚动”的画布
        container = ttk.Frame(self.root)                     # 一个占满剩余空间的容器
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        # fill=BOTH + expand=True = 让容器随窗口拉伸时自动变大，铺满剩余空间
        self.canvas = tk.Canvas(container, bg=C_BG, highlightthickness=0)
        # bg=C_BG：用我们定义的主题浅色背景，和窗口融为一体
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        # Scrollbar=滚动条。command=...yview 表示：拖动滚动条时通知画布纵向滚动
        self.canvas.configure(yscrollcommand=vsb.set)        # 反过来：画布滚动时更新滚动条位置
        vsb.pack(side=tk.RIGHT, fill=tk.Y)                   # 滚动条放最右边，纵向铺满
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)   # 画布占剩余

        # 真正承载内容的是一个普通 Frame，把它“嵌”进画布
        self.grid_frame = ttk.Frame(self.canvas)
        # 记录 canvas 里嵌 grid_frame 的窗口编号，方便下面让它跟着画布一起变宽
        self._grid_window = self.canvas.create_window((0, 0), window=self.grid_frame, anchor='nw')
        # 画布一变宽，grid_frame 也跟着撑满整个宽度，卡片才不会只挤在左边
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        # bind("<Configure>", 回调)：每当 Frame 尺寸改变就触发，自动更新画布可滚动范围。
        self.grid_frame.bind("<Configure>",
                             lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # lambda 是“匿名小函数”：lambda 参数: 表达式。
        #   这行等价于定义 def _f(e): self.canvas.configure(...) 然后把 _f 传进去。
        #   bbox("all") = 算出所有子控件包住的矩形范围，scrollregion=它 即“该滚多远”。

        # ★空状态：还没有聚类结果时，画布中央显示一句引导，避免大片空白。
        #   用 place(relx/rely) 定位，窗口尺寸变了也会自动保持居中。
        self._empty = tk.Frame(self.canvas, bg=C_BG)
        tk.Label(self._empty, text="还没有聚类结果", bg=C_BG, fg=C_TEXT_SUB,
                 font=('Microsoft YaHei', 17, 'bold')).pack(pady=(0, 10))
        tk.Label(self._empty, text="选择照片文件夹，点击「开始聚类」\n"
                                   "程序会自动识别照片里的人脸，并按人物智能分组",
                 bg=C_BG, fg=C_TEXT_DIS, justify=tk.CENTER, font=FONT).pack()
        self._empty.place(relx=0.5, rely=0.42, anchor='center')

        # 鼠标滚轮滚动（Windows 下滚轮的 delta 以 120 为单位，除以120得到“几格”）
        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        # bind_all = 给“所有控件”绑事件；int(-1*delta/120)：向上滚是正数，翻转成向上翻页

    def _on_canvas_resize(self, event):
        """画布大小变化时统一处理：让卡片区跟着撑满宽度。

        用 after_idle 把多次连续触发合并成一次，避免拖动窗口时疯狂重算。
        """
        self.root.after_idle(self._apply_canvas_layout)

    def _apply_canvas_layout(self):
        """真正执行布局：卡片区撑满画布宽度（卡片才不会只挤在左半边）。"""
        try:
            w = self.canvas.winfo_width()
            if self._grid_window is not None and w > 1:
                self.canvas.itemconfigure(self._grid_window, width=w)
        except Exception:
            pass

    # ---------------- ★顶栏标题横幅 ----------------
    def _build_header(self):
        """顶部横幅：应用名称 + 一句话副标题，给界面一个‘门面’。"""
        # 用普通 tk.Frame（可以设 bg），白底 + 一条主色调细线顶边，干净克制。
        header = tk.Frame(self.root, bg=C_PANEL)
        header.pack(fill=tk.X, pady=(0, 10))
        tk.Frame(header, bg=C_ACCENT, height=3).pack(fill=tk.X)   # 顶部 3px 主色细线

        title_row = tk.Frame(header, bg=C_PANEL)
        title_row.pack(fill=tk.X, padx=22, pady=(12, 6))
        tk.Label(title_row, text="人脸聚类相册", bg=C_PANEL,
                 fg=C_TEXT_TITLE, font=FONT_TITLE).pack(side=tk.LEFT)
        tk.Label(title_row, text="自动识别照片里的人，按人物智能分组", bg=C_PANEL,
                 fg=C_TEXT_SUB, font=FONT_SUB).pack(side=tk.LEFT, padx=(12, 0), pady=(8, 0))

    # ---------------- 交互 ----------------
    def _choose_dir(self):
        """弹出系统目录选择框，把所选文件夹写入输入框"""
        # filedialog.askdirectory：弹出系统自带的“选择文件夹”窗口，返回用户选的路径
        d = filedialog.askdirectory(title="选择照片文件夹")
        if d:                       # 用户点了“取消”会返回空字符串''，''是假值，if不成立
            self.input_dir.set(d)   # 把选到的路径写进输入框（界面随之显示）

    def _show_help(self):
        """弹出帮助窗口：先介绍软件功能，再解释参数含义。"""
        messagebox.showinfo(
            "功能与参数说明",
            "关于本软件\n"
            "    人脸聚类相册会扫描你选择的照片文件夹，自动识别照片里的人脸，\n"
            "    把“同一人”的照片归到一组，就像手机相册的人物分组。\n"
            "    照片在本机处理、不会上传，隐私安全。\n\n"
            "首次使用\n"
            "    需要联网下载人脸识别模型（约 300MB，只下载一次）；\n"
            "    若当前无网络会给出提示，联网后重试即可。\n\n"
            "主要功能\n"
            "    · 人脸识别：自动检测并提取每张脸的特征；\n"
            "    · 自动分组：同一人物自动聚成一组，不用一张张翻；\n"
            "    · 查看详情：点击人物卡片，即可查看该人物的全部照片；\n"
            "    · 命名人物：给人物起名字，导出时自动建同名文件夹；\n"
            "    · 一键导出：把分组好的照片按人物复制/移动到新文件夹归档。\n\n"
            "──────────\n"
            "参数说明\n"
            "eps 阈值（当前默认 0.43）\n"
            "    判断两张脸有多‘像’的敏感度。值越小越严格，越难把照片分到一组；\n"
            "    值越大越宽松，容易把不同的人误并到一起。照片拍得模糊/角度多变时，\n"
            "    可以稍微调大到 0.6 左右；照片都很清晰正规时，保持 0.43 就好。\n\n"
            "最少张数（默认 2）\n"
            "    至少要凑够几张相同的人脸，才把它算作‘一个人物’。\n"
            "    设为 2 表示只出现过 1 次的独照会被归入“未分类”，避免杂七杂八的\n"
            "    路人脸也占一个格子。\n\n"
            "GPU 加速\n"
            "    本机有 NVIDIA 显卡且运行库齐全时会自动启用，速度明显更快；\n"
            "    若驱动在但运行库没装，勾选后程序会自动联网下载加速组件（约 1.2GB，\n"
            "    只下一次）；没有 NVIDIA 显卡（A 卡/核显）或下载失败时，程序自动\n"
            "    改用 CPU 运行，功能完全一样。\n\n"
            "小科普：聚类其实分三步走\n"
            "    ① 先用 eps 粗分一次；② 每堆算一张‘平均脸’当老大，每张照片\n"
            "    重新认老大（不像的就踢出去）；③ 两个老大长得太像（比如同一个人\n"
            "    不同年龄段）就把两堆合并。反复几遍，结果会更准。")

    def _start_cluster(self):
        """读取并校验用户输入，清空旧结果后启动后台聚类线程"""
        d = self.input_dir.get().strip()    # .get()取输入框内容；.strip()去掉首尾空白
        if not d or not os.path.isdir(d):   # 空 or 不是有效目录
            messagebox.showwarning("提示", "请先选择有效的照片文件夹")  # 弹黄色警告框
            return
        if self._busy:                       # 上一轮还在跑？拒绝再次开跑
            messagebox.showinfo("提示", "上一轮聚类还在进行中，请等待它完成")
            return
        try:   # 参数格式校验（eps 允许小数，最少张数必须是整数）。int/float是“类型转换函数”
            eps = float(self.eps.get())     # 用户可能乱填，转换失败会抛异常 -> 被except接住
            mc = int(self.min_cluster.get())
        except Exception:
            messagebox.showwarning("提示", "eps 或最少张数格式不正确")
            return

        # 清空上次的界面内容（重新开始）
        for w in self.grid_frame.winfo_children():   # 拿到 grid_frame 所有子控件
            w.destroy()                              # 逐个销毁（≈ delete 掉这些控件）
        self._thumb_refs.clear()      # 引用也清空（旧缩略图随之被 GC 回收）
        self.groups = []              # 结果清空
        self.progress['value'] = 0    # 进度条归零
        self._hide_empty()            # 收起“还没有聚类结果”的占位提示
        self._busy = True             # 标记“正在跑”，同时禁用按钮防重复点击
        self._start_btn.config(state='disabled')
        self.status_text.set("正在加载模型...")   # 状态栏提示

        # ★名言：等 0.6 秒再开始轮播（避开“正在加载模型”抢镜）。之后每 4 秒换一句，
        #   聚类结束（_cluster_finished）会自动停掉并隐藏。
        self.root.after(600, self._show_quote)

        # ★ 启动后台线程。target=要在线程里运行的函数，args=(…) 是传给它的参数(元组)。
        # daemon=True：主程序退出时这个线程自动结束(不必手动 join)，防止卡住关不掉。
        threading.Thread(target=self._cluster_worker, args=(d, eps, mc), daemon=True).start()
        # .start() 真正开线程。开完之后 _start_cluster 立刻返回，界面继续响应；
        # 耗时的活都在新线程里做——这就是“界面不卡死”的关键。

    def _cluster_finished(self):
        """一轮聚类结束（成功或出错）时调用：复位忙碌标记、恢复“开始聚类”按钮"""
        # 这个方法只在 UI 主线程里被 _poll_queue 调用，所以没有多线程写变量的风险。
        self._busy = False
        self._start_btn.config(state='normal')   # 按钮恢复可点，允许跑下一轮(新的阈值)

        # ★名言：取消还没响的定时器，并“淡出”收起名言卡片，回到干净状态
        if self._quote_after_id is not None:
            self.root.after_cancel(self._quote_after_id)
            self._quote_after_id = None
        self._quote_seq += 1                     # 作废掉可能还在半路的切句动画
        if self._quote_box.winfo_manager():      # 卡片还显示着？先淡出再藏
            self._fade_box(False, on_done=lambda: (
                self._quote_label.config(text=""), self._quote_box.pack_forget()))
        else:                                    # 卡片本来就没显示，直接复位
            self._quote_label.config(text="")
            self._quote_box.pack_forget()

    # ------------------------------------------------------------------
    # ★名言动画：淡入/淡出卡片、轮播换句（都做成渐变，不硬切）
    # ------------------------------------------------------------------
    def _mix_color(self, c1, c2, t):
        """两个 #rrggbb 颜色按比例 t(0~1) 插值，得到中间色。用来做渐变动画。"""
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        return f'#{round(r1 + (r2 - r1) * t):02x}' \
               f'{round(g1 + (g2 - g1) * t):02x}' \
               f'{round(b1 + (b2 - b1) * t):02x}'

    def _fade_box(self, forward, on_done):
        """把名言卡片从“看不见”渐变到“看得见”(forward=True) 或反向淡出。

        同时渐变：卡片底板颜色(C_PANEL↔C_ACCENT_SOFT)和文字颜色(C_PANEL↔C_ACCENT_DK)。
        文字藏色用 C_PANEL：淡到极致时文字就像“溶解”进白面板，不突兀。
        动画期间若 _quote_seq 变了（新一轮操作），就立刻停手，避免冲突。
        """
        seq = self._quote_seq            # 记录本段动画的“世代”
        steps, delay = 10, 20            # 10 帧 x 20ms = 200ms 一段渐变
        bg0, bg1 = (C_PANEL, C_ACCENT_SOFT) if forward else (C_ACCENT_SOFT, C_PANEL)
        fg0, fg1 = (C_PANEL, C_ACCENT_DK) if forward else (C_ACCENT_DK, C_PANEL)

        def step(i):
            if seq != self._quote_seq:   # 中途来了新动画，旧帧直接作废
                return
            t = i / steps
            c = self._mix_color(bg0, bg1, t)   # 底板色
            self._quote_box.config(bg=c)
            self._quote_label.config(bg=c, fg=self._mix_color(fg0, fg1, t))
            if i < steps:
                self.root.after(delay, lambda: step(i + 1))
            else:
                on_done()                # 渐变动画走完，交给回调继续

        step(0)

    def _show_quote(self):
        """聚类开始：把第一句名言淡入显示（卡片从进度条下方“长”出来）。"""
        if not self._busy:               # 窗口被关了/已结束，就不折腾了
            return
        self._quote_seq += 1
        q = QUOTES[self._quote_idx % len(QUOTES)]
        self._quote_idx += 1
        self._quote_label.config(text=f"❝ {q} ❞")
        self._quote_box.pack(fill=tk.X, pady=(8, 0))   # 插到进度条正下方
        self._fade_box(True, on_done=self._schedule_next_quote)

    def _schedule_next_quote(self):
        """当前这句名言显示满 4 秒后，轮到下一句。"""
        if not self._busy:
            return
        self._quote_after_id = self.root.after(4000, self._next_quote)

    def _next_quote(self):
        """轮播换句：先淡出旧句，换上新句文字，再淡入。全程不硬切。"""
        self._quote_after_id = None
        if not self._busy:               # 已结束/被取消？不再继续轮播
            return
        q = QUOTES[self._quote_idx % len(QUOTES)]   # % 取余实现“轮着转”，转完从头再来
        self._quote_idx += 1
        self._fade_box(False, on_done=lambda: self._fade_in_new(q))

    def _fade_in_new(self, q):
        """新名言淡入（承接 _next_quote 的淡出）。"""
        if not self._busy:
            return
        self._quote_seq += 1
        self._quote_label.config(text=f"❝ {q} ❞")
        self._fade_box(True, on_done=self._schedule_next_quote)

    # ---------------- 联网下载（精简 exe 的关键） ----------------
    def _stream_download(self, url, dest, label):
        """把 url 流式下载到 dest 文件，百分比进度通过消息队列回报。

        下载过程抛出的异常交给调用方处理（网络断了、文件不存在等都会抛）。
        """
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get('Content-Length') or 0)   # 总字节数（可能没有）
            done = 0
            last_pct = -1
            with open(dest, 'wb') as f:
                while True:
                    chunk = resp.read(256 * 1024)      # 每次读 256KB，边下边写
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:                          # 有总长度才算得出百分比
                        pct = min(100, int(done * 100 / total))
                        if pct != last_pct:            # 只在百分比变化时才上报，少刷队列
                            last_pct = pct
                            self.msg_queue.put(('progress', (done, total, f"{label} {pct}%")))

    def _ensure_model_ready(self):
        """确保人脸识别模型已下载到本地。已就绪→True；需要时联网下载，成功→True，失败→False。

        模型不打包进 exe：第一次使用时从这里联网下载（insightface 官方源），
        下载到程序同目录 / 用户目录下的 .insightface，之后一直复用。
        """
        model_dir = os.path.join(MODEL_ROOT, 'models', 'buffalo_l')
        try:
            # 已有完整的模型（目录里至少一个 .onnx）就直接用
            if os.path.isdir(model_dir) and any(f.endswith('.onnx') for f in os.listdir(model_dir)):
                return True
            os.makedirs(model_dir, exist_ok=True)
            zip_path = os.path.join(os.path.dirname(model_dir), 'buffalo_l.zip')
            self.msg_queue.put(('status', "首次使用，正在联网下载人脸识别模型（约 300MB）..."))
            self._stream_download(MODEL_DL_URL, zip_path, "正在下载人脸识别模型")
            # 解压：zip 可能自带 buffalo_l/ 顶层文件夹，也可能没有，两种都兼容
            with zipfile.ZipFile(zip_path) as zf:
                if any(n.startswith('buffalo_l/') for n in zf.namelist()):
                    zf.extractall(os.path.join(MODEL_ROOT, 'models'))
                else:
                    zf.extractall(model_dir)
            try:
                os.remove(zip_path)      # 解压完删掉压缩包，省空间
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _gpu_wheel_url(self):
        """确定 onnxruntime-gpu（CUDA provider 所在）wheel 的下载地址。

        优先用配置的 GPU_RUNTIME_URL；没配置就自动去 PyPI 找和 exe 里捆绑的
        onnxruntime 同版本的 Windows 安装包，保证 provider 版本完全匹配。
        """
        if GPU_RUNTIME_URL:
            return GPU_RUNTIME_URL
        ver = onnxruntime.__version__
        py_tag = 'cp%d%d' % sys.version_info[:2]          # 例如 cp310
        api = 'https://pypi.org/pypi/onnxruntime-gpu/%s/json' % ver
        with urllib.request.urlopen(api, timeout=30) as resp:
            data = json.load(resp)
        for f in data['urls']:
            fn = f['filename']
            if fn.endswith('win_amd64.whl') and ('-%s-' % py_tag) in fn:
                return f['url']
        raise RuntimeError("未找到 onnxruntime-gpu %s 的 Windows 安装包" % ver)

    def _pypi_wheel_url(self, pkg, major=None):
        """找某个 PyPI 包在 Windows 上的 wheel 下载地址。major 指定主版本号（如 cudnn 只要 9.x）。"""
        api = 'https://pypi.org/pypi/%s/json' % pkg
        with urllib.request.urlopen(api, timeout=30) as resp:
            data = json.load(resp)

        def ver_key(v):                      # 把 "9.10.2.21" 拆成数字元组比大小，避免 9.9>9.10 这种错
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
        """从 wheel(实为 zip) 里挑出名字以 need 开头的 DLL，解压到 dest 目录。"""
        with zipfile.ZipFile(wheel_path) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base.lower().endswith('.dll') and base.lower().startswith(need):
                    with zf.open(name) as src, open(os.path.join(dest, base), 'wb') as dst:
                        shutil.copyfileobj(src, dst)

    def _try_enable_gpu_on_demand(self):
        """勾选了 GPU 但本机缺 CUDA 运行库时：联网下载 GPU 组件并启用。

        从 PyPI 下载 5 个官方包（onnxruntime-gpu + NVIDIA 的 cublas/cudnn/cudart/
        cufft 运行库，合计约 1.2GB，只下一次），抽出 DLL 放进 onnxruntime 的 capi
        目录——onnxruntime 就是从那里找 CUDA provider 的。
        返回 True=启用成功 / False=失败（没网络、下载失败、目录不可写等），
        失败时调用方自动退回 CPU，绝不崩。
        """
        try:
            self.msg_queue.put(('status', "正在联网下载 GPU 加速组件（约 1.2GB，只下一次）..."))
            # onnxruntime 只从自己的 capi 目录加载 CUDA provider，DLL 必须放进去
            capi = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi')
            if not os.path.isdir(capi) or not os.access(capi, os.W_OK):
                return False                       # 包目录不可写，放弃 GPU 走 CPU
            tmp = os.path.join(capi, '_gpu_tmp')
            os.makedirs(tmp, exist_ok=True)

            if GPU_RUNTIME_URL:
                # 用户自托管：一个 zip 里装好全部 DLL，解开挪进 capi 即可
                zip_path = os.path.join(tmp, 'gpu_runtime.zip')
                self._stream_download(GPU_RUNTIME_URL, zip_path, "正在下载 GPU 加速组件")
                self._extract_dlls(zip_path, tmp, ('cublas', 'cudnn', 'cudart', 'cufft',
                                                   'onnxruntime_providers_cuda',
                                                   'onnxruntime_providers_shared'))
            else:
                # 官方源：onnxruntime-gpu（CUDA provider）+ NVIDIA 运行库
                wheel = os.path.join(tmp, 'ort_gpu.whl')
                self._stream_download(self._gpu_wheel_url(), wheel, "正在下载 GPU 加速组件")
                self._extract_dlls(wheel, tmp, ('onnxruntime_providers_cuda',
                                                'onnxruntime_providers_shared'))
                os.remove(wheel)
                for pkg, major in (('nvidia-cublas-cu12', 12), ('nvidia-cuda-runtime-cu12', 12),
                                   ('nvidia-cudnn-cu12', 9), ('nvidia-cufft-cu12', None)):
                    w = os.path.join(tmp, pkg + '.whl')
                    self._stream_download(self._pypi_wheel_url(pkg, major), w,
                                          "正在下载 GPU 加速组件")
                    self._extract_dlls(w, tmp, ('cublas', 'cudnn', 'cudart', 'cufft'))
                    os.remove(w)

            # 全部 DLL 挪进 onnxruntime 的 capi 目录
            for f in os.listdir(tmp):
                if f.lower().endswith('.dll'):
                    shutil.move(os.path.join(tmp, f), os.path.join(capi, f))
            shutil.rmtree(tmp, ignore_errors=True)

            self._gpu_runtime = _gpu_runtime_ready()
            self._cuda_available = self._nvidia_driver and self._gpu_runtime
            return self._cuda_available
        except Exception:
            return False

    # ---------------- 后台聚类 ----------------
    def _cluster_worker(self, d, eps, min_cluster):
        """在工作线程里跑完整聚类流程，结果通过消息队列发回 UI"""
        # 这个方法运行在“后台线程”里。注意：Tk 界面只能由主线程改！
        # 所以这里绝不去碰界面控件，只把结果打包放进 self.msg_queue，让主线程取走。
        try:
            images = collect_images(d)          # 1) 收集所有图片路径
            if not images:                      # 空列表 = False，直接判空
                self.msg_queue.put(('error', "该文件夹下没有找到任何图片"))
                # 往队列塞一个“消息”。消息就是一个 tuple(元组)：('error', 文字)
                # 约定第一个元素是类型，后面是内容。有点像 C++ 里传 struct{kind,text}。
                return                          # return 提前结束整个函数
            # f"..." 是 f-string 格式化字符串：{表达式} 会被求值替换成文字。
            # 等价 C++ 的字符串拼接但更省事：f"共 {len(images)} 张"
            self.msg_queue.put(('status', f"共 {len(images)} 张图片，正在加载人脸模型..."))

            # 懒加载模型：只在第一次聚类时初始化并复用。self.app 之前是 None
            if self.app is None:
                # 1) 先确保模型就位：不在本地就联网下载（首次使用）。失败=没网/下载失败
                if not self._ensure_model_ready():
                    self.msg_queue.put(('error',
                        "人脸识别模型没有就绪。首次使用需要联网下载模型（约 300MB），\n"
                        "当前可能没有网络或下载失败，请检查网络后重新开始聚类。"))
                    return
                # 2) 决定是否用 GPU：勾了 GPU 且本机有 CUDA 就直接用；
                #    勾了但本机没装 CUDA，就尝试“联网下载 GPU 加速组件”；
                #    都搞不定就退回 CPU（绝不因显卡问题让程序崩掉）。
                want_gpu = False
                if self.use_gpu.get():
                    if self._cuda_available:
                        want_gpu = True
                    elif self._try_enable_gpu_on_demand():
                        want_gpu = True
                    else:
                        self.msg_queue.put(('status',
                                            "未启用 GPU：本机无 NVIDIA 显卡或无法联网下载加速组件，已使用 CPU"))
                providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if want_gpu
                             else ['CPUExecutionProvider'])
                self.msg_queue.put(('status',
                                    "正在加载人脸模型（" + ("GPU 加速" if want_gpu else "CPU")
                                    + "）..."))
                # FaceAnalysis(name='buffalo_l',...) 参数名=值 是“关键字参数”，不用记顺序。
                # providers 是“执行后端”列表：能上 GPU(CUDA)就 GPU，不行退回 CPU。
                try:
                    self.app = FaceAnalysis(name='buffalo_l', root=MODEL_ROOT, providers=providers)
                    self.app.prepare(ctx_id=0 if want_gpu else -1, det_size=(640, 640))
                    # ctx_id=0 用GPU / -1 用CPU。det_size 检测分辨率，越大越准越慢
                except Exception:
                    # ★ 兜底：就算探测说“有”，实际加载失败（驱动不对/缺运行时）也退回 CPU
                    self.app = FaceAnalysis(name='buffalo_l', root=MODEL_ROOT,
                                            providers=['CPUExecutionProvider'])
                    self.app.prepare(ctx_id=-1, det_size=(640, 640))
                    self.msg_queue.put(('status', "显卡不可用，已自动改用 CPU 运行"))

            # ------------------------------------------------------------------
            # 第二阶段：逐张图片做人脸检测 + 特征提取
            # ------------------------------------------------------------------
            embeddings = []   # 收集到的所有人脸 512 维特征（每张脸一个 512 元素的向量）
            qualities = []    # 与 embeddings 一一对应：每张脸的“清楚分”（0~1），模糊=低分
            metas = []        # 与 embeddings 一一对应：记录这张脸来自哪张图、人脸框在哪
            total = len(images)
            use_half = False  # 中等/巨大量级时，特征改存 float16（内存减半，存半精度算全精度）
            for i, p in enumerate(images):
                # enumerate(list)：遍历时同时给你 (下标, 元素)。类似 C++ for i 但更顺手。
                # p 是当前图片的完整路径
                img = read_image(p)   # 兼容中文路径（cv2.imread 读不了中文名文件）
                if img is None:
                    continue    # 损坏或无法读取的图片直接跳过（continue=跳过本次循环）
                faces = self.app.get(img)   # ★ 人脸检测！返回这张图里所有检测到的人脸
                # insightface 模型会自动做检测+对齐+提取特征，faces 是个 list
                if not use_half and len(embeddings) + len(faces) > FACE_TIER_MID:
                    # ★ 量级升级：人脸总数已跨过“中等”档，把已收的特征一次性转成
                    #   float16（半精度），之后的也都按半精度存——内存省一半，
                    #   聚类时再转回 float32 算，精度几乎无损。
                    use_half = True
                    embeddings = [e.astype(np.float16) for e in embeddings]
                for face in faces:          # 一张合照可能有多张脸，逐一处理
                    embeddings.append(face.embedding.astype(np.float16 if use_half
                                                            else np.float32))
                    # face.embedding 是人脸特征向量(默认numpy float32)。astype 转成float32。
                    # .append 往列表尾部加元素(≈ vector::push_back)
                    metas.append({'image': p, 'bbox': face.bbox.tolist()})
                    # face.bbox 是人脸框(可能是numpy数组)，.tolist() 转成普通Python列表
                    # 这里塞的是一个 dict：{'image': 图片路径, 'bbox': [x1,y1,x2,y2]}
                    # 顺手给这张脸估个“清楚分”，模糊的后面聚类时会放宽标准（见 _face_quality_score）
                    qualities.append(_face_quality_score(img, face.bbox,
                                                         float(face.det_score)))
                if (i + 1) % 3 == 0 or i + 1 == total:   # 每3张(或最后一张)回报一次进度
                    # % 取余；or 是“逻辑或”。减少往队列塞消息的次数，降低开销
                    self.msg_queue.put(('progress', (i + 1, total,
                                                      f"已处理 {i + 1}/{total} 张，累计人脸 {len(embeddings)} 个")))
                    # 消息里还套了一个元组 (当前, 总数, 文字)

            if not embeddings:
                self.msg_queue.put(('error', "所有图片中都没有检测到人脸"))
                return

            # ★ 巨大量级“确认闸门”：人脸数太多，先问用户同不同意，同意才继续。
            #   这里在后台线程，不能直接弹窗，所以发消息给主线程弹窗，
            #   再用事件(Event)等主线程把用户回答传回来。
            if len(embeddings) > FACE_TIER_HUGE:
                est_mb = len(embeddings) * 2 // (1024 * 1024)   # 每张脸≈2KB 的估算
                self._confirm_event.clear()
                self._confirm_ok = False
                self.msg_queue.put(('confirm', (len(embeddings), est_mb)))
                self._confirm_event.wait()          # 阻塞等待主线程回话
                if not self._confirm_ok:
                    self.msg_queue.put(('cancelled', "人脸数量过多，已取消本轮聚类"))
                    return

            # ------------------------------------------------------------------
            # 第三阶段：归一化 + 智能聚类（先粗分，再认老大、认亲合并）
            # ------------------------------------------------------------------
            # 把“列表 of 向量”变成一个二维 numpy 矩阵，形状 (人脸数, 512)
            emb = np.array(embeddings)
            if emb.dtype == np.float16:
                emb = emb.astype(np.float32)   # 存半精度、算全精度：精度几乎无损
            # L2 归一化：每个向量除以自己的长度，长度变为1。
            # 好处：两个都归一化后，点积就代表余弦相似度，判断“像不像”很简单。
            # np.linalg.norm(emb, axis=1) 对每行求长度；keepdims=True 保住维度好做除法；
            # +1e-9 防止除以0(加个小epsilon，数值上保险)。
            emb = _l2_normalize(emb)
            # ★ 上面一行就完成了“所有行各自归一化”，numpy 是向量化计算，
            #    相当于 C++ 里一个循环 + SIMD，但写起来是一句。这就是 numpy 的威力。

            self.msg_queue.put(('status', "正在聚类，请耐心等待..."))
            # 用升级后的聚类：不再一把尺子量到底，而是
            #   粗分 → 每堆算“平均脸”当老大 → 每张脸重新认老大 → 两个老大太像就并堆，
            # 反复几遍，错分漏分的脸都会被纠正回来（详见 smart_cluster 上方的说明）。
            # 第三只脚：把每张脸的“清楚分”也传进去，模糊的脸自动放宽标准，不怕被冤枉。
            labels = smart_cluster(emb, eps, min_cluster, qualities=qualities)
            # labels 是个一维数组，长度=人脸数。labels[i]=第i张脸属于哪个组。
            # 特别地，标签 = -1 表示“谁都不像”：归不进任何组，进“未分类”。

            # 分组：按标签把人脸归类。dict 的键是组号，值是这组的人脸信息列表
            persons = {}                    # 空字典
            unclassified = []               # 装 -1 的散脸
            for idx, lab in enumerate(labels):   # 遍历所有标签
                if lab == -1:
                    unclassified.append(metas[idx])
                else:
                    # dict.setdefault(键, 默认值)：如果键不存在，先给它一个空列表，再返回它。
                    # 于是下面的 .append 无论键新不新都能用——等价于 C++ 里
                    #   auto &v = m[lab]; if(!m.count(lab)) ... 的简化写法。
                    persons.setdefault(int(lab), []).append(metas[idx])
                    # int(lab) 因为 numpy 的整数要先转普通int才能当dict键（实际上numpy整数也行，
                    # 这里显式转一下更稳）。

            # 排序：按“每组照片数”从多到少。sorted(可迭代对象, key=取值函数)
            # key=lambda x: -len(x)：对每个元素 x 取 -len(x) 作为排序依据，负数=降序
            sorted_persons = sorted(persons.values(), key=lambda x: -len(x))
            # persons.values() 是字典里所有“值”(这里是各人物组)组成的视图
            # 列表推导式：把每组套一个dict外壳。结果形如
            #   [{'type':'person','name':'','items':[原图1,原图2,...]}, ...]
            # 'name' 是人物名字，默认空字符串，用户点“命名”后才会填上（导出时当文件夹名）
            groups = [{'type': 'person', 'name': '', 'items': g} for g in sorted_persons]
            if unclassified:                 # 有噪声才追加“未分类”组（空组没意义）
                groups.append({'type': 'unclassified', 'items': unclassified})

            self.msg_queue.put(('done', groups))   # 结果发回 UI 线程渲染
            # 注意：到这里线程活干完了，函数结束即线程结束。界面更新交给了主线程的 _poll_queue。

        except Exception as e:               # 捕获任何异常，as e 把异常对象给变量e
            import traceback                 # 调试工具：能打印完整的错误调用栈
            self.msg_queue.put(('error', f"聚类出错：{e}\n\n{traceback.format_exc()}"))
            # f-string 里直接塞异常信息；\n\n 空两行；format_exc() 生成完整回溯文本给用户看

    # ---------------- 队列轮询 ----------------
    def _poll_queue(self):
        """UI 主线程定时轮询消息队列，把后台线程的消息同步到界面"""
        # 这个方法每 120ms 被 Tk 调用一次(见 __init__ 的 after)。做一件事：取光队列里新消息。
        try:
            while True:                        # 一直取，直到队列空被下面的异常打断
                msg = self.msg_queue.get_nowait()   # 非阻塞取消息；队列空会抛 queue.Empty
                kind = msg[0]                  # 消息的第0个元素：类型
                # Python 的元组/列表都能用下标取，msg[0] 就是 'status'/'progress'/...
                if kind == 'status':           # 只是更新一句状态文字
                    self.status_text.set(msg[1])
                elif kind == 'progress':       # 带进度：更新进度条的最大值/当前值
                    cur, total, text = msg[1]  # 解包元组拿三个数
                    self.progress['maximum'] = total   # 设置进度条最大值
                    self.progress['value'] = cur       # 设置当前进度
                    self.status_text.set(text)         # 顺便更新文字（如“已处理3/10”）
                elif kind == 'confirm':          # 巨大量级：弹窗问用户要不要继续
                    n, est_mb = msg[1]
                    # 主线程弹确认框，把用户的回答写回 _confirm_ok 并唤醒后台线程
                    self._confirm_ok = messagebox.askyesno(
                        "大量人脸确认",
                        f"检测到约 {n} 张人脸，继续聚类预计占用内存约 {est_mb} MB、"
                        f"耗时也会明显变长。\n\n确定继续吗？")
                    self._confirm_event.set()
                elif kind == 'cancelled':        # 用户拒绝继续，优雅地结束本轮
                    self.status_text.set(msg[1])
                    self._cluster_finished()
                elif kind == 'error':
                    self.status_text.set("出错")
                    self._cluster_finished()   # 出错也代表本轮结束，恢复可再跑
                    messagebox.showerror("错误", msg[1])   # 红色错误弹窗，msg[1]是详情
                elif kind == 'done':           # 聚类完成！msg[1] 就是最终 groups
                    self.groups = msg[1]
                    # 下面两行统计：列表推导式先筛出'person'组再数个数
                    # len([...]) = 多少个'person'组；sum(...) 累加每组 items 的数量
                    n_person = len([g for g in self.groups if g['type'] == 'person'])
                    n_unc = sum(len(g['items']) for g in self.groups if g['type'] == 'unclassified')
                    #   if 能跟在推导式后面做“过滤”
                    self.progress['value'] = self.progress['maximum'] or 1   # 进度条拉满(兜底1)
                    # (A or B)：A为真取A，A为假(如0)取B。这里防止maximum是0导致进度条不刷新
                    self.status_text.set(
                        f"完成 —— 识别出 {n_person} 个人物"
                        + (f"，{n_unc} 张单张脸归入未分类" if n_unc else ""))
                    # + 可以拼接字符串；if..else.. 表达式能夹在括号里动态拼后半句
                    self._cluster_finished()   # 本轮结束，恢复按钮允许再次聚类
                    self._render_groups()   # 用最终结果重绘人物卡片
        except queue.Empty:
            pass    # 队列空则忽略，继续用 after 定时轮询
        # 再次预约自己 120ms 后再跑一次（相当于无限定时器，保证 UI 一直“看得见”新消息）
        self.root.after(120, self._poll_queue)

    # ---------------- ★美化：给控件加“悬停高亮”的小助手 ----------------
    def _bind_card_hover(self, widget, color):
        """让某个控件在鼠标移入/移出时改变边框颜色，做出“可点击”的提示。

        参数：
          widget —— 要绑定的控件（人物卡片 Frame 或里面的 Label）
          color  —— 鼠标悬停时要变成的边框色
        Tk 的事件绑定写法：
          widget.bind("<Enter>", 回调)  鼠标移进控件瞬间触发
          widget.bind("<Leave>", 回调)  鼠标移出控件瞬间触发
        回调里用 lambda 传一个额外的固定颜色参数（用默认参数提前“拍照”，防晚绑定）。
        """
        widget.bind("<Enter>", lambda e, c=color: widget.config(highlightbackground=c))
        # highlightbackground 是 tk.Frame/Label 的“边框颜色”，改成主色调=出现高亮描边
        widget.bind("<Leave>", lambda e: widget.config(highlightbackground=C_BORDER))
        # 鼠标一走，边框颜色恢复成默认的淡色

    def _hide_empty(self):
        """收起画布中央的“还没有聚类结果”占位提示（有内容了就不需要它）。"""
        if self._empty is not None:
            self._empty.place_forget()

    # ---------------- 命名人物 ----------------
    def _rename_group(self, group):
        """给某个人物起名字：弹输入框，把名字存进这组的 'name' 字段，然后刷新卡片"""
        # simpledialog.askstring = 弹一个“输入框”小窗，返回用户敲的文字；点取消返回 None
        new_name = simpledialog.askstring(
            "命名人物",
            "给这位人物起个名字（导出相册时子文件夹就用这个名字）：\n"
            "留空或点取消 = 不改名。",
            parent=self.root,
            initialvalue=group.get('name', ''))   # 默认填上次起过的名字
        if new_name is not None and new_name.strip():
            group['name'] = new_name.strip()      # 存进这组（字典可变，直接改）
            self._render_groups()                 # 重画卡片，让新名字显示出来

    # ---------------- 导出相册到硬盘 ----------------
    def _export_album(self):
        """把分类好的照片按人物写进硬盘：每人一个文件夹，文件夹用人物名字命名。

        规则：
          - 未分类的照片不导出（它们没名字，也没法定归属）；
          - 写入方式由单选决定：'copy'=复制一份（原图不动）；'move'=移动原图（原文件夹里就没它了）；
          - 没起名字的人物，文件夹用“人物N”兜底，保证每个都有地方放。
        """
        if self._busy:
            messagebox.showinfo("提示", "聚类还在跑，等它完成再导出。")
            return
        # 先取出有名字或至少有归属的人物组；未分类跳过
        persons = [g for g in self.groups if g['type'] == 'person']
        if not persons:
            messagebox.showwarning("提示", "还没有可导出的人物，先聚类试试。")
            return
        # 选一个输出文件夹（导出时会在这里面按人物建子文件夹）
        out_root = filedialog.askdirectory(title="选择相册输出文件夹（将在此目录下按人物建子文件夹）")
        if not out_root:
            return

        # 一个文件夹里不能有重名文件：重名就自动加 _2、_3……
        mode = self.export_mode.get()     # 'copy' 或 'move'
        copied = 0               # 统计：成功写了几张
        skipped = 0              # 统计：跳过几张（文件不存在等原因）

        # 逐个处理每个人物
        for i, group in enumerate(persons):
            # 文件夹名优先用人名；没人名用“人物N”兜底
            folder = group.get('name') or f"人物{i + 1}"
            # 文件夹名不能含 Windows 禁止的字符 < > : " / \ | ? *，把它们清掉
            folder = re.sub(r'[<>:"/\\|?*]', '', folder).strip() or f"人物{i + 1}"
            folder_path = os.path.join(out_root, folder)
            try:
                os.makedirs(folder_path, exist_ok=True)   # 建子文件夹（已存在就忽略）
            except Exception:
                messagebox.showerror("错误", f"无法创建文件夹：{folder_path}")
                return

            # 把这个人的每张照片复制/移动过去
            for item in group['items']:
                src = item['image']
                if not os.path.isfile(src):          # 原图找不到了就跳过
                    skipped += 1
                    continue
                base = os.path.basename(src)         # 原名，如 photo.jpg
                dst = os.path.join(folder_path, base)
                # 若目标已有同名文件（不同子目录撞名），加个 _2 再试
                n = 2
                while os.path.exists(dst):
                    stem, ext = os.path.splitext(base)
                    dst = os.path.join(folder_path, f"{stem}_{n}{ext}")
                    n += 1
                try:
                    if mode == 'move':
                        shutil.move(src, dst)        # 移动原图（原位置就没有了）
                    else:
                        shutil.copy2(src, dst)       # 复制一份（原图保留）
                    copied += 1
                except Exception:
                    skipped += 1                    # 单个失败不中断，继续写下一张

        # 收尾：弹窗告诉用户结果
        word = "移动" if mode == 'move' else "复制"
        messagebox.showinfo(
            "导出完成",
            f"已{word} {copied} 张照片到：\n{out_root}\n"
            + (f"（{skipped} 张因原图缺失或写入失败被跳过）" if skipped else "")
            + "\n\n未分类的照片没有导出。")

    # ---------------- 渲染人物网格 ----------------
    def _render_groups(self):
        """把聚类结果渲染成人物卡片网格（每张卡片用该人物第一张脸作缩略图）"""
        for w in self.grid_frame.winfo_children():   # 清空画布里旧控件
            w.destroy()
        self._thumb_refs.clear()          # 并释放旧缩略图引用

        cols = GRID_COLS                  # 每行 4 张卡片
        # 让 4 列等宽、平均拉伸，把整行铺满（否则卡片会缩在左边）
        for c in range(cols):
            self.grid_frame.columnconfigure(c, weight=1, uniform='card_col')
        self._hide_empty()        # 有结果了，收起空状态占位提示
        for i, group in enumerate(self.groups):   # 遍历每一组人物
            r, c = divmod(i, cols)                # divmod(a,b) 一次给(a//b, a%b)，即(行,列)
            # 白底卡片：细边框 + 悬停变主色，干净不花哨
            card = tk.Frame(self.grid_frame, bg='white', padx=8, pady=8,
                            highlightbackground=C_BORDER, highlightthickness=1)
            card.grid(row=r, column=c, padx=12, pady=12, sticky='nsew')   # 放进网格(r,c)
            # sticky='nsew'：让卡片向东南西北四方“伸展”以填满格子
            self._bind_card_hover(card, C_CARD_HOV)   # 卡片悬停高亮

            # 卡片内容分三块：上面是“人脸缩略图”，中间是“人物名字”，下面是“照片数”。
            # 用 tk.Frame 把三行信息竖着叠放(pack 默认从上往下)，结构更清楚。
            head = tk.Frame(card, bg='white')
            head.pack(fill=tk.X)
            first = group['items'][0]      # 取这组的第一张脸信息 dict
            thumb = make_face_thumb(first['image'], first['bbox'], size=(176, 176))
            # 用第1张脸做人脸缩略图（前面写的读图+裁剪函数）
            if thumb:                      # 不是None（生成成功）才显示
                lbl = tk.Label(head, image=thumb, bg='white', cursor='hand2')
                # Label 直接拿图片当内容；cursor='hand2' 鼠标悬停变“小手”
                lbl.image = thumb  # type: ignore[attr-defined]   ★ 防 GC 关键！Tk 的图片对象若不被变量引用会被垃圾回收
                #                                   导致显示空白。把图片挂到控件属性上保命。
                lbl.pack(pady=(8, 2))      # 显示
                # bind("<Button-1>", 回调)：绑定“鼠标左键按下”事件。
                # 回调用了 lambda e, g=group, idx=i: ...：
                #   ① 必须接收事件参数 e（Tk 回调第一个参数固定是事件对象）
                #   ② g=group, idx=i 是“默认参数提前取值”——
                #      这是 Python 经典坑的解法：若写 lambda e: self._show_person(group, idx)，
                #      group/idx 是循环变量，循环结束后全变成最后一次的值(闭包晚绑定)。
                #      用默认参数把“当前这一轮的值”提前“拍照固定”下来，每个按钮才对。
                lbl.bind("<Button-1>", lambda e, g=group, idx=i: self._show_person(g, idx))
                self._bind_card_hover(lbl, C_CARD_HOV)   # 图片区也能触发高亮
                self._thumb_refs.append(thumb)   # 再存一份引用，双保险

            # 卡片下方的标题文字：人物名字加粗，“共 M 张”灰色小字。
            # 人物名字：用户起过名就用名字，没起名就先叫“人物 N”（未分类固定叫“未分类”）
            if group['type'] == 'unclassified':
                title = "未分类"
            else:
                title = group.get('name') or f"人物 {i + 1}"
            ttk.Label(card, text=title, font=FONT_BOLD,
                      background='white').pack(pady=(10, 0))
            ttk.Label(card, text=f"共 {len(group['items'])} 张",
                      font=FONT_SMALL, background='white',
                      foreground=C_TEXT_SUB).pack(pady=(0, 4))

            # 人物组才有“命名”按钮：点一下弹输入框起个名字，方便导出时当文件夹名。
            # 未分类不允许命名（也不导出），所以不给按钮。
            if group['type'] == 'person':
                ttk.Button(card, text="命名", width=8,
                           command=lambda g=group: self._rename_group(g)).pack(pady=(2, 0))
            # pady=(6,0)：上下留白 (上6, 下0)。很多Tk参数接受(左右/上下)这种元组。

    # ---------------- 人物详情窗口 ----------------
    def _show_person(self, group, idx):
        """新建一个窗口，网格展示某个分组里的所有照片，点击可打开原图"""
        win = tk.Toplevel(self.root)   # Toplevel = 新建一个独立的顶层子窗口
        if group['type'] == 'unclassified':
            win.title(f"未分类 · 共 {len(group['items'])} 张照片")
        else:
            # 详情窗口标题也带上人物名字（有名字用名字，没名字用“人物N”）
            gname = group.get('name') or f"人物 {idx + 1}"
            win.title(f"{gname} · 共 {len(group['items'])} 张照片")
        win.geometry("1140x780")       # 子窗口大小
        win.minsize(760, 540)          # 最小尺寸

        # 详情窗口顶部一行提示，弱化处理
        ttk.Label(win, text="点击任意照片用系统默认程序打开原图",
                  font=FONT_SMALL, foreground=C_TEXT_SUB, padding=(12, 8)).pack(fill=tk.X)

        # 详情窗口里同样用一个可滚动 Canvas 铺照片网格（与主界面结构完全一致，可对照看）
        container = ttk.Frame(win)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        canvas = tk.Canvas(container, bg=C_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        gf = ttk.Frame(canvas)                 # gf = grid frame，内容承载器
        gf_window = canvas.create_window((0, 0), window=gf, anchor='nw')
        gf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        # 画布变宽时，让内容 frame 跟着撑满整个宽度（否则照片只会挤在左边一列）
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(gf_window, width=canvas.winfo_width())
                    if canvas.winfo_width() > 1 else None)
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        # 注意这里的 lambda 没有存到变量，纯当作“匿名回调”用。e 是Tk传入的事件对象。

        refs = []   # 本窗口缩略图引用（防 GC），局部变量，等窗口关闭后被回收
        cols = GRID_COLS
        # 让 cols 列等宽、平均拉伸，把整行铺满（否则照片会缩在左边）
        for c in range(cols):
            gf.columnconfigure(c, weight=1, uniform='detail_col')
        for i, item in enumerate(group['items']):   # 遍历这个人物组的每一张照片
            r, c = divmod(i, cols)
            frame = tk.Frame(gf, bg='white', padx=4, pady=4,
                             highlightbackground=C_BORDER, highlightthickness=1)
            frame.grid(row=r, column=c, padx=8, pady=8, sticky='nsew')
            self._bind_card_hover(frame, C_CARD_HOV)   # 照片卡悬停同样高亮

            thumb = make_face_thumb(item['image'], item['bbox'], size=(200, 200))
            if thumb:
                lbl = tk.Label(frame, image=thumb, bg='white', cursor='hand2')
                lbl.image = thumb  # type: ignore[attr-defined]
                lbl.pack()
                # 点照片 -> 打开原图。这里把“图片路径”也用默认参数固定住，防止晚绑定错乱
                lbl.bind("<Button-1>", lambda e, p=item['image']: self._open_full(p))
                refs.append(thumb)

            # 文件名太长就截断，末尾加“...”
            name = os.path.basename(item['image'])   # 只取路径最后一段=文件名
            if len(name) > 22:
                name = name[:20] + "..."             # 切片取前20个字符
            ttk.Label(frame, text=name, font=FONT_SMALL, background='white').pack(pady=(4, 0))

        setattr(win, '_thumb_refs', refs)  # 挂在窗口对象上保持引用，窗口关闭后随窗口一起释放
        # 给对象临时“加”一个新属性是合法的(Python 对象很自由)——这里借它保引用。

    @staticmethod
    # @staticmethod 是“装饰器”：给下面的方法打个标记。被标的方法不接收 self，
    # 相当于 C++ 的“静态成员函数”，不依赖对象状态，直接用类名调也可。
    def _open_full(path):
        """用系统默认程序打开原图；失败则弹窗显示路径"""
        try:
            os.startfile(path)   # Windows 专属：用默认程序打开该文件（如图片查看器）
        except Exception:
            messagebox.showinfo("照片路径", path)   # 打不开就弹窗告诉用户路径


# ============================================================================
# main 入口
# ============================================================================
def main():
    root = tk.Tk()       # ★ 创建主窗口对象（每个Tk程序只能有一个，它就是“根窗口”）
    try:
        # Windows 高 DPI 适配：不调用的话高分屏下界面会发虚/模糊
        from ctypes import windll   # 调用 Windows 原生 C API 的库；windll.shcore 是系统函数
        windll.shcore.SetProcessDpiAwareness(1)   # 告诉系统“本程序自己感知DPI”
    except Exception:
        pass              # pass = 什么都不做（语法上占位，因为 except 里不能完全空着）
    FaceAlbumGUI(root)    # 实例化主界面类（构造时自动把界面都搭好）
    root.mainloop()       # ★ 进入事件循环：程序在这里“卡住”，一直处理点击/重绘等事件，
    #                         直到用户关窗口才返回，main 结束、程序退出。类似WinMain的消息循环。


# ============================================================================
# 程序真正开始执行的地方
# ============================================================================
# 每个 .py 文件被“直接运行”时，解释器会先设置特殊变量 __name__ = '__main__'。
# 若该文件是被别人 import 当模块用，则 __name__ = 模块名(如'face_album_gui')，不会等于
# '__main__'，于是下面这行不会执行——这样可以安全地被复用而不自动弹界面。
if __name__ == '__main__':
    main()
