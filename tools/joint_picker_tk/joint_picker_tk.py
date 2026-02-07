#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox


Point = Tuple[float, float]
Joints2D = Dict[str, List[float]]  # {name: [x, y]}


def _resource_rel(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, *parts))


DEFAULT_IMAGE = _resource_rel("..", "..", "asset", "data", "human_kp.png")
DEFAULT_JOINTS = _resource_rel("..", "..", "asset", "data", "main_joint.json")


def load_joints(path: str) -> Joints2D:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    joints: Joints2D = {}
    if not isinstance(obj, dict):
        return joints
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        if not (isinstance(v, list) and len(v) >= 2):
            continue
        try:
            x = float(v[0])
            y = float(v[1])
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            joints[k] = [x, y]
    return joints


def save_joints(path: str, joints: Joints2D) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(joints, f, ensure_ascii=False, indent=2)


def dist2(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


@dataclass
class DragState:
    name: str
    prev: Point


class JointPickerApp:
    def __init__(self, root: tk.Tk, image_path: str, joints_path: str):
        self.root = root
        self.root.title("4DHOI Joint Picker (Tkinter)")

        self.image_path: Optional[str] = None
        self.joints_path: Optional[str] = None

        self.joints: Joints2D = {}
        self.selected_name: Optional[str] = None

        self.point_radius = 6
        self.drag: Optional[DragState] = None

        # try to load image via Tk (PNG supported by Tk 8.6+)
        self._img_tk: Optional[tk.PhotoImage] = None
        self._img_w = 1
        self._img_h = 1

        self._build_ui()

        if os.path.exists(image_path):
            self.load_image(image_path)
        if os.path.exists(joints_path):
            self.load_joints_file(joints_path)

    def _build_ui(self) -> None:
        self.root.geometry("1200x780")

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X)

        btn_load_img = tk.Button(top, text="加载图片", command=self.on_load_image)
        btn_load_img.pack(side=tk.LEFT, padx=6, pady=6)

        btn_load_json = tk.Button(top, text="加载关节JSON", command=self.on_load_joints)
        btn_load_json.pack(side=tk.LEFT, padx=6, pady=6)

        btn_save = tk.Button(top, text="保存JSON", command=self.on_save)
        btn_save.pack(side=tk.LEFT, padx=6, pady=6)

        btn_save_as = tk.Button(top, text="另存为...", command=self.on_save_as)
        btn_save_as.pack(side=tk.LEFT, padx=6, pady=6)

        radius_frame = tk.Frame(top)
        radius_frame.pack(side=tk.LEFT, padx=18)
        tk.Label(radius_frame, text="半径").pack(side=tk.LEFT)
        self.radius_var = tk.IntVar(value=self.point_radius)
        radius = tk.Scale(
            radius_frame,
            from_=2,
            to=20,
            orient=tk.HORIZONTAL,
            length=160,
            variable=self.radius_var,
            command=self.on_radius_change,
        )
        radius.pack(side=tk.LEFT)

        main = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # left panel
        left = tk.Frame(main)
        main.add(left, width=360)

        tk.Label(left, text="关节列表（选中后：点击图像设置坐标，拖拽点位微调）").pack(
            side=tk.TOP, anchor="w", padx=8, pady=(10, 4)
        )

        search_frame = tk.Frame(left)
        search_frame.pack(side=tk.TOP, fill=tk.X, padx=8)
        tk.Label(search_frame, text="搜索").pack(side=tk.LEFT)
        self.search_var = tk.StringVar(value="")
        ent = tk.Entry(search_frame, textvariable=self.search_var)
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ent.bind("<KeyRelease>", lambda _e: self.refresh_list())

        self.listbox = tk.Listbox(left)
        self.listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_joint)

        coord_frame = tk.LabelFrame(left, text="当前关节坐标")
        coord_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 10))

        self.coord_var = tk.StringVar(value="-")
        tk.Label(coord_frame, textvariable=self.coord_var).pack(side=tk.LEFT, padx=8, pady=8)

        # right panel (canvas)
        right = tk.Frame(main)
        main.add(right)

        self.canvas = tk.Canvas(right, bg="#111", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        yscroll = tk.Scrollbar(right, orient=tk.VERTICAL, command=self.canvas.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        xscroll = tk.Scrollbar(self.root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # status bar
        self.status_var = tk.StringVar(value="就绪")
        status = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def on_radius_change(self, _val: str) -> None:
        self.point_radius = int(self.radius_var.get())
        self.redraw_points()

    def on_load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp"), ("All", "*")],
        )
        if not path:
            return
        self.load_image(path)

    def on_load_joints(self) -> None:
        path = filedialog.askopenfilename(
            title="选择关节 JSON",
            filetypes=[("JSON", "*.json"), ("All", "*")],
        )
        if not path:
            return
        self.load_joints_file(path)

    def on_save(self) -> None:
        if self.joints_path:
            try:
                save_joints(self.joints_path, self.joints)
                self.set_status(f"已保存：{self.joints_path}")
            except Exception as e:
                messagebox.showerror("保存失败", str(e))
        else:
            self.on_save_as()

    def on_save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存关节 JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="main_joint_custom.json",
        )
        if not path:
            return
        try:
            save_joints(path, self.joints)
            self.joints_path = path
            self.set_status(f"已保存：{path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def load_image(self, path: str) -> None:
        try:
            img = tk.PhotoImage(file=path)
        except Exception as e:
            messagebox.showerror(
                "图片加载失败",
                f"无法用 Tk 直接加载图片：{path}\n\n{e}\n\n"
                "如果你的 Tk 版本不支持 PNG，可以考虑安装 Pillow 并把图片转成 GIF/PPM，"
                "或升级到 Tk 8.6+。",
            )
            return

        self._img_tk = img
        self.image_path = path
        self._img_w = int(img.width())
        self._img_h = int(img.height())

        self.canvas.delete("all")
        self.canvas_img_id = self.canvas.create_image(0, 0, image=img, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, self._img_w, self._img_h))

        self.set_status(f"图片已加载：{path} ({self._img_w}x{self._img_h})")
        self.redraw_points()

    def load_joints_file(self, path: str) -> None:
        try:
            joints = load_joints(path)
        except Exception as e:
            messagebox.showerror("JSON 读取失败", str(e))
            return

        self.joints = joints
        self.joints_path = path
        self.refresh_list(select_first=True)
        self.redraw_points()
        self.set_status(f"关节已加载：{path} ({len(self.joints)} 个)")

    def refresh_list(self, select_first: bool = False) -> None:
        q = self.search_var.get().strip().lower()
        names = sorted(self.joints.keys())
        if q:
            names = [n for n in names if q in n.lower()]

        self._list_names = names
        self.listbox.delete(0, tk.END)
        for n in names:
            p = self.joints.get(n)
            if p:
                label = f"{n}    ({int(round(p[0]))}, {int(round(p[1]))})"
            else:
                label = n
            self.listbox.insert(tk.END, label)

        if select_first and names:
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(0)
            self.listbox.event_generate("<<ListboxSelect>>")

    def on_select_joint(self, _evt=None) -> None:
        idxs = self.listbox.curselection()
        if not idxs:
            return
        idx = int(idxs[0])
        if idx < 0 or idx >= len(getattr(self, "_list_names", [])):
            return
        name = self._list_names[idx]
        self.selected_name = name
        p = self.joints.get(name)
        if p:
            self.coord_var.set(f"{name}: x={int(round(p[0]))}, y={int(round(p[1]))}")
        else:
            self.coord_var.set(f"{name}: -")
        self.redraw_points()

    def _canvas_xy(self, event) -> Point:
        return (float(self.canvas.canvasx(event.x)), float(self.canvas.canvasy(event.y)))

    def _find_near_point(self, xy: Point, radius_px: float) -> Optional[str]:
        r2 = radius_px * radius_px
        best_name = None
        best_d2 = float("inf")
        for name, p in self.joints.items():
            d2 = dist2((p[0], p[1]), xy)
            if d2 <= r2 and d2 < best_d2:
                best_d2 = d2
                best_name = name
        return best_name

    def on_canvas_motion(self, event) -> None:
        x, y = self._canvas_xy(event)
        self.set_status(f"cursor: x={int(round(x))}, y={int(round(y))}")

    def on_canvas_click(self, event) -> None:
        if not self._img_tk:
            return

        xy = self._canvas_xy(event)
        # prefer dragging if click near existing point
        near = self._find_near_point(xy, radius_px=max(10, self.point_radius + 6))
        if near is not None:
            self.selected_name = near
            self._select_name_in_list(near)
            self.drag = DragState(name=near, prev=(self.joints[near][0], self.joints[near][1]))
            self.redraw_points()
            return

        if not self.selected_name:
            self.set_status("请先在左侧选择一个关节")
            return

        x, y = xy
        x = float(max(0, min(self._img_w - 1, x)))
        y = float(max(0, min(self._img_h - 1, y)))
        self.joints[self.selected_name] = [x, y]
        self.coord_var.set(f"{self.selected_name}: x={int(round(x))}, y={int(round(y))}")
        self.refresh_list(select_first=False)
        self.redraw_points()

    def on_canvas_drag(self, event) -> None:
        if not self.drag:
            return
        xy = self._canvas_xy(event)
        x = float(max(0, min(self._img_w - 1, xy[0])))
        y = float(max(0, min(self._img_h - 1, xy[1])))
        self.joints[self.drag.name] = [x, y]
        if self.selected_name == self.drag.name:
            self.coord_var.set(f"{self.drag.name}: x={int(round(x))}, y={int(round(y))}")
        self.redraw_points()

    def on_canvas_release(self, _event) -> None:
        if not self.drag:
            return
        self.refresh_list(select_first=False)
        self.set_status(f"已移动：{self.drag.name}")
        self.drag = None

    def _select_name_in_list(self, name: str) -> None:
        try:
            idx = self._list_names.index(name)
        except ValueError:
            return
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)
        self.on_select_joint()

    def redraw_points(self) -> None:
        if not self._img_tk:
            return

        # remove old point items
        self.canvas.delete("joint_point")
        self.canvas.delete("joint_label")

        r = self.point_radius
        for name, p in self.joints.items():
            x, y = p[0], p[1]
            selected = name == self.selected_name

            fill = "#60a5fa" if selected else "#e6e8f2"
            outline = "#1f2937" if not selected else "#93c5fd"
            width = 2 if selected else 1

            self.canvas.create_oval(
                x - r,
                y - r,
                x + r,
                y + r,
                fill=fill,
                outline=outline,
                width=width,
                tags=("joint_point",),
            )
            self.canvas.create_text(
                x + r + 4,
                y - r - 2,
                text=name,
                fill=fill,
                anchor="nw",
                font=("TkFixedFont", 9),
                tags=("joint_label",),
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--joints", default=DEFAULT_JOINTS)
    args = ap.parse_args()

    root = tk.Tk()
    JointPickerApp(root, image_path=args.image, joints_path=args.joints)
    root.mainloop()


if __name__ == "__main__":
    main()
