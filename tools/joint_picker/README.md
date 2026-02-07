# 4DHOI Joint Picker

一个轻量的关节点点选/微调小工具，用于编辑类似 `main_joint.json` 这种 2D 像素坐标 JSON：

```json
{
  "leftHand": [1031, 61],
  "hip": [546, 340]
}
```

## 运行

推荐用本地静态服务器（否则直接双击打开 `index.html` 时，浏览器可能因为 CORS 无法 `fetch` 本地 png/json）。

在 `4dhoi_preprocess/webapp_debug` 下执行：

```bash
python -m http.server 8008
```

然后打开：

- `http://127.0.0.1:8008/tools/joint_picker/`

## 使用

- 默认会尝试加载：
  - 图片：`asset/data/human_kp.png`
  - 关节：`asset/data/main_joint.json`
- 选择关节后，在图上单击即可设置坐标
- 拖拽点位可微调
- “加载按钮名”会把 `asset/data/button_name.json` 里的关节名导入列表（没坐标的会显示为 `-`，需要你逐个点）
- “下载 JSON”导出结果

## 常见问题

- 如果看不到图片/加载失败：确认你是通过 `http.server` 打开的页面，不是 `file://`。
