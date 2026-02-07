# Joint Picker (Tkinter)

一个独立的 Tkinter 小工具：在人体示意图上点选/拖拽关节点，保存为 `main_joint.json` 这种格式。

## 默认读取

- 图片：`4dhoi_preprocess/webapp_debug/asset/data/human_kp.png`
- 关节：`4dhoi_preprocess/webapp_debug/asset/data/main_joint.json`

## 运行

在 `4dhoi_preprocess/webapp_debug` 目录下运行：

```bash
python tools/joint_picker_tk/joint_picker_tk.py
```

也可以指定文件：

```bash
python tools/joint_picker_tk/joint_picker_tk.py \
  --image asset/data/human_kp.png \
  --joints asset/data/main_joint.json
```

## 用法

- 左侧列表选中一个关节
- 在右侧图上单击：把该关节坐标设为鼠标位置
- 单击点位附近：进入拖拽（按住左键拖动）
- `保存JSON`：覆盖写回当前加载的 JSON 文件
- `另存为...`：导出到新文件

## 注意

- 该工具用 Tk 自带的 `PhotoImage` 直接加载 PNG（需要 Tk 8.6+，多数 Linux/conda 环境默认 OK）。
- 如果弹出“PNG 不支持”的报错，你可以升级 Tk，或把图片换成 GIF/PPM。需要的话我也可以改成依赖 Pillow 的版本。
