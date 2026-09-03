import re

with open('face_album_gui.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到并替换顶栏部分
for i, line in enumerate(lines):
    if '# 工具条：一个白色"面板"容器' in line:
        # 改进工具条内边距和背景
        if i+1 < len(lines) and 'top = ttk.Frame' in lines[i+1]:
            lines[i+1] = lines[i+1].replace("padding=(14, 10)", "padding=(16, 12)")
            lines[i+1] = lines[i+1].replace("padx=16, pady=(0, 8)", "padx=18, pady=(0, 10)")

# 改进卡片边框和阴影样式
for i, line in enumerate(lines):
    if 'card = tk.Frame(self.grid_frame' in line:
        # 改进卡片样式：更圆润的border
        lines[i] = line.replace('highlightthickness=1', 'highlightthickness=2')

# 改进进度条样式
for i, line in enumerate(lines):
    if 'self.progress = ttk.Progressbar' in line:
        if i+1 < len(lines):
            lines[i+1] = lines[i+1].replace("pady=(6, 0)", "pady=(8, 0)")

# 改进状态面板
for i, line in enumerate(lines):
    if 'status_panel = ttk.Frame(self.root' in line:
        lines[i] = line.replace("padding=(14, 8)", "padding=(16, 10)")
        if i+1 < len(lines):
            lines[i+1] = lines[i+1].replace("padx=16, pady=(0, 8)", "padx=18, pady=(0, 10)")

with open('face_album_gui.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Final beautification applied")
