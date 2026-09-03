# 本地人脸聚类相册 (Face Album)

一个纯本地的桌面 GUI 应用：自动识别照片文件夹里的人脸，用 AI 提取特征后按人物自动分组，
并以“人物卡片”的方式浏览相册。所有照片和数据都留在本地，不上传任何图片。

## 功能特性

- 📁 一键选择照片文件夹，自动递归扫描常见图片格式（jpg / png / bmp / webp 等）
- 🧠 基于 insightface (buffalo_l) 人脸检测 + 512 维特征提取
- 🔀 智能聚类：先粗分，再“认老大 + 认亲合并”反复精修，把同一人物归为一组
- 🎚️ 自适应标准：给每张脸估“清楚分”，模糊的照片自动放宽分组标准，清晰照片保持严格
- 🃏 人物卡片网格浏览，点击卡片查看该人物的全部照片
- 🖼️ 点击照片用系统默认程序打开原图
- ⚡ 可选 GPU 加速（有 NVIDIA 显卡时自动启用，否则回退 CPU）
- 🎨 简洁美观的现代 UI（自定义配色 + 悬停高亮 + 进度条）

## 环境要求

- Python 3.9+（建议 3.10）
- 依赖见 [requirements.txt](requirements.txt)

## 安装

```bash
# 1. 创建虚拟环境（可选）
conda create -n face-album python=3.10 -y
conda activate face-album

# 2. 安装依赖
pip install -r requirements.txt
```

> 首次运行会自动下载 insightface 的人脸模型（buffalo_l，约 600MB），下载到
> `~/.insightface/models/` 目录（或本仓库的 `.insightface/` 目录）。请耐心等待，
> 之后会复用缓存，不再重复下载。

## 使用

```bash
python face_album_gui.py
```

Windows 下也可以直接双击 [run_gui.bat](run_gui.bat)（需要已配置好 conda 环境 `face-album`）。

操作步骤：
1. 点击「浏览...」选择照片文件夹
2. 调整参数（一般用默认值即可）
3. 点击「开始聚类 ▶」
4. 聚类完成后点击人物卡片查看该人物的全部照片

## 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| eps 阈值 | 0.43 | 判断两张脸多“像”的敏感度。越小越严格，越大越宽松。照片模糊时可调到 0.6 |
| 最少张数 | 2 | 至少凑够几张同人脸才算一个“人物”。设 1 会让独照也单独成组 |
| GPU 加速 | 开 | 优先用 NVIDIA 显卡跑模型，没有显卡时自动回退 CPU |

## 目录结构

```
face_album/
├── face_album_gui.py   # 主程序（GUI + 聚类逻辑）
├── run_gui.bat         # Windows 一键启动脚本
├── bg.png              # 背景图资源
├── requirements.txt    # Python 依赖
└── .insightface/       # 人脸模型缓存（自动下载，不入库）
```

## 技术说明

- 人脸检测/特征：insightface + buffalo_l 模型（ArcFace 512 维特征）
- 聚类：DBSCAN 粗分 + 迭代精修（每堆算“平均脸”当老大重新认领，两个老大太像就合并），
  归一化后的特征按余弦相似度归组
- 模糊自适应：用 Laplacian 边缘锐利度 + 检测置信度给每张脸估“清楚分”，
  模糊的脸认老大时阈值自动放宽，避免被误判成未分类
- 界面：Python 标准库 Tkinter，后台线程 + 消息队列保证界面不卡死
- 中文路径：自定义 `read_image()` 绕过 `cv2.imread` 对中文路径的兼容性问题

## 许可

仅供学习交流使用。人脸模型（insightface buffalo_l）版权归其原作者所有。
