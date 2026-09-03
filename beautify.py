# 读取原文件
with open('face_album_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 颜色改进 - 在第一个颜色定义处开始替换
old_colors = '''C_BG          = '#eef1f7'   # 窗口/画布背景：极浅的蓝灰，比刺眼的纯白柔和
C_BG_DEEP     = '#e6eaf5'   # 画布背景的"深一档"：让白色卡片浮在更明显的底色上
C_PANEL       = '#ffffff'   # 面板背景：纯白，用于顶栏、工具条、人物卡片
C_ACCENT      = '#4f6ef2'   # 主色调（靛蓝紫）：按钮、标题、进度条、悬停高亮都用它
C_ACCENT_LT   = '#6c85f6'   # 主色调的"浅一档"：鼠标悬停在按钮上时用
C_ACCENT_DK   = '#3b58d0'   # 主色调的"深一档"：按钮按下那一刻用
C_ACCENT_SOFT = '#eef1fb'   # 主色调的"极浅打底"：按钮悬停背景、小徽章底色
C_BORDER      = '#d8dfed'   # 卡片/输入框的"淡边框"颜色
C_BORDER_DK   = '#c6cfe6'   # 稍深一点的边框/分隔线颜色
C_TEXT        = '#27303f'   # 主文字颜色：深蓝灰，比纯黑更耐看
C_TEXT_TITLE  = '#1c2536'   # 标题/大号文字：比正文更深，更有分量
C_TEXT_SUB    = '#7b8496'   # 次要文字颜色：中灰，用于说明、提示
C_TEXT_DIS    = '#aab3c4'   # 置灰文字：按钮禁用/占位提示
C_OK          = '#2f9e5f'   # 状态灯·绿色：就绪/完成
C_RUN         = '#4f6ef2'   # 状态灯·蓝色：正在处理
C_ERR         = '#e5484d'   # 状态灯·红色：出错
C_WARN        = '#e8900c'   # 状态灯·琥珀：警告
C_CARD_HOV    = '#4f6ef2'   # 鼠标悬停在人物卡片上时，边框变成主色调（提示"可以点"）'''

new_colors = '''C_BG          = '#faf7f2'   # 极浅米色背景
C_BG_DEEP     = '#f5f1ed'   # 次要背景米色
C_PANEL       = '#fffbf8'   # 极浅奶油色面板
C_ACCENT      = '#d4956e'   # 主色调温暖杏色
C_ACCENT_LT   = '#e0a87c'   # 浅温暖杏色
C_ACCENT_DK   = '#c47e54'   # 深温暖棕色
C_ACCENT_SOFT = '#f5ede7'   # 极浅米杏打底
C_BORDER      = '#e8ddd5'   # 柔和米色边框
C_BORDER_DK   = '#d9cfc6'   # 棕米色边框
C_TEXT        = '#5c4a40'   # 温暖深棕文字
C_TEXT_TITLE  = '#4a3830'   # 深棕标题
C_TEXT_SUB    = '#9b8b7e'   # 暖灰棕次文字
C_TEXT_DIS    = '#bfb0a6'   # 浅棕灰置灰
C_OK          = '#6ba383'   # 暖绿状态
C_RUN         = '#d4956e'   # 橙状态响应
C_ERR         = '#d97a66'   # 暖红出错
C_WARN        = '#e8a76e'   # 暖琥珀警告
C_CARD_HOV    = '#d4956e'   # 杏色悬停'''

content = content.replace(old_colors, new_colors)

# 字体改进
content = content.replace(
    "FONT_TITLE = ('Microsoft YaHei', 20, 'bold')",
    "FONT_TITLE = ('Microsoft YaHei', 22, 'bold')"
)

# 保存
with open('face_album_gui.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Colors and fonts updated successfully")
