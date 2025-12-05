#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Геометрия ---

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Segment:
    p1: Point
    p2: Point

@dataclass
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

# --- Алгоритмы отсечения ---

def cohen_sutherland_clip(seg: Segment, rect: Rect) -> Optional[Segment]:
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def outcode(p: Point) -> int:
        code = INSIDE
        if p.x < rect.xmin: code |= LEFT
        elif p.x > rect.xmax: code |= RIGHT
        if p.y < rect.ymin: code |= BOTTOM
        elif p.y > rect.ymax: code |= TOP
        return code

    p1, p2 = Point(seg.p1.x, seg.p1.y), Point(seg.p2.x, seg.p2.y)
    c1, c2 = outcode(p1), outcode(p2)

    while True:
        if c1 == 0 and c2 == 0:
            return Segment(p1, p2)
        if c1 & c2:
            return None
        out = c1 or c2
        if out & TOP:
            x = p1.x + (p2.x - p1.x) * (rect.ymax - p1.y) / (p2.y - p1.y)
            y = rect.ymax
        elif out & BOTTOM:
            x = p1.x + (p2.x - p1.x) * (rect.ymin - p1.y) / (p2.y - p1.y)
            y = rect.ymin
        elif out & RIGHT:
            y = p1.y + (p2.y - p1.y) * (rect.xmax - p1.x) / (p2.x - p1.x)
            x = rect.xmax
        elif out & LEFT:
            y = p1.y + (p2.y - p1.y) * (rect.xmin - p1.x) / (p2.x - p1.x)
            x = rect.xmin
        if out == c1:
            p1, c1 = Point(x, y), outcode(Point(x, y))
        else:
            p2, c2 = Point(x, y), outcode(Point(x, y))

def sutherland_hodgman_clip_polygon(poly: List[Point], clipper: List[Point]) -> List[Point]:
    def inside(p: Point, a: Point, b: Point) -> bool:
        return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) >= 0

    def intersection(p1: Point, p2: Point, a: Point, b: Point) -> Point:
        A1 = p2.y - p1.y
        B1 = p1.x - p2.x
        C1 = A1 * p1.x + B1 * p1.y
        A2 = b.y - a.y
        B2 = a.x - b.x
        C2 = A2 * a.x + B2 * a.y
        det = A1 * B2 - A2 * B1
        if det == 0:
            return p2  # параллельные, но для выпуклого допускается
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return Point(x, y)

    output = poly
    for i in range(len(clipper)):
        input_list = output
        output = []
        A, B = clipper[i], clipper[(i + 1) % len(clipper)]
        for j in range(len(input_list)):
            P = input_list[j]
            Q = input_list[(j + 1) % len(input_list)]
            if inside(Q, A, B):
                if not inside(P, A, B):
                    output.append(intersection(P, Q, A, B))
                output.append(Q)
            elif inside(P, A, B):
                output.append(intersection(P, Q, A, B))
    return output

# --- Чтение файлов ---

def read_segments_rect(path: str) -> Tuple[List[Segment], Rect]:
    with open(path, 'r') as f:
        parts = f.read().strip().split()
    it = iter(parts)
    n = int(next(it))
    segs = []
    for _ in range(n):
        x1, y1, x2, y2 = map(float, (next(it), next(it), next(it), next(it)))
        segs.append(Segment(Point(x1, y1), Point(x2, y2)))
    xmin, ymin, xmax, ymax = map(float, (next(it), next(it), next(it), next(it)))
    rect = Rect(xmin, ymin, xmax, ymax)
    return segs, rect

def read_polygon(path: str) -> List[Point]:
    with open(path, 'r') as f:
        parts = f.read().strip().split()
    it = iter(parts)
    n = int(next(it))
    pts = []
    for _ in range(n):
        x, y = map(float, (next(it), next(it)))
        pts.append(Point(x, y))
    return pts

# --- Визуализация ---

def plot_segments(segments: List[Segment], rect: Rect, clipped: List[Segment], canvas: FigureCanvasTkAgg):
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axhline(0, color='lightgray', linewidth=0.5)
    ax.axvline(0, color='lightgray', linewidth=0.5)

    ax.plot([rect.xmin, rect.xmax, rect.xmax, rect.xmin, rect.xmin],
            [rect.ymin, rect.ymin, rect.ymax, rect.ymax, rect.ymin],
            color='blue', linewidth=2, label='Окно')

    for s in segments:
        ax.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y],
                color='gray', linestyle='--',
                label='Исходный' if 'Исходный' not in ax.get_legend_handles_labels()[1] else "")

    for s in clipped:
        ax.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y],
                color='red', linewidth=2,
                label='Отсечённый' if 'Отсечённый' not in ax.get_legend_handles_labels()[1] else "")

    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_title("Коэн–Сазерленд (отрезки)")
    canvas.draw()

def plot_polygon(subject: List[Point], clipper: List[Point], clipped: List[Point], canvas: FigureCanvasTkAgg):
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axhline(0, color='lightgray', linewidth=0.5)
    ax.axvline(0, color='lightgray', linewidth=0.5)

    def close_poly(p):
        return p + [p[0]] if p else p

    if clipper:
        c = close_poly(clipper)
        ax.plot([p.x for p in c], [p.y for p in c], color='blue', linewidth=2, label='Окно (клипер)')
    if subject:
        s = close_poly(subject)
        ax.plot([p.x for p in s], [p.y for p in s], color='gray', linestyle='--', label='Исходный многоугольник')
    if clipped:
        cl = close_poly(clipped)
        ax.plot([p.x for p in cl], [p.y for p in cl], color='green', linewidth=2, label='Отсечённый')
        ax.fill([p.x for p in cl], [p.y for p in cl], color='green', alpha=0.2)

    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_title("Сазерленд–Ходжман (многоугольник)")
    canvas.draw()

# --- UI ---

class ClipApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Отсечение (вариант 13)")
        self.geometry("1000x750")

        self.segments: List[Segment] = []
        self.rect: Optional[Rect] = None

        self.poly_subject: List[Point] = []
        self.poly_clipper: List[Point] = []

        self._build_ui()

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка отрезков
        frame_segments = ttk.Frame(notebook)
        notebook.add(frame_segments, text="Отрезки")

        top_seg = ttk.Frame(frame_segments)
        top_seg.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        ttk.Button(top_seg, text="Открыть файл отрезков", command=self.load_segments_file).pack(side=tk.LEFT, padx=5)
        self.segments_path = tk.StringVar(value="")
        ttk.Label(top_seg, textvariable=self.segments_path, width=50, anchor="w").pack(side=tk.LEFT, padx=5)
        ttk.Button(top_seg, text="Отсечь", command=self.run_segments_clip).pack(side=tk.LEFT, padx=5)

        fig1 = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas_segments = FigureCanvasTkAgg(fig1, master=frame_segments)
        self.canvas_segments.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Вкладка многоугольника
        frame_poly = ttk.Frame(notebook)
        notebook.add(frame_poly, text="Многоугольник")

        top_poly = ttk.Frame(frame_poly)
        top_poly.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        ttk.Button(top_poly, text="Исходный многоугольник", command=self.load_subject_poly).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_poly, text="Клипер (окно)", command=self.load_clipper_poly).pack(side=tk.LEFT, padx=5)
        self.subject_path = tk.StringVar(value="")
        self.clipper_path = tk.StringVar(value="")
        ttk.Label(top_poly, textvariable=self.subject_path, width=30, anchor="w").pack(side=tk.LEFT, padx=5)
        ttk.Label(top_poly, textvariable=self.clipper_path, width=30, anchor="w").pack(side=tk.LEFT, padx=5)
        ttk.Button(top_poly, text="Отсечь", command=self.run_polygon_clip).pack(side=tk.LEFT, padx=5)

        fig2 = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas_poly = FigureCanvasTkAgg(fig2, master=frame_poly)
        self.canvas_poly.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --- Обработчики отрезков ---

    def load_segments_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            segments, rect = read_segments_rect(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")
            return
        self.segments = segments
        self.rect = rect
        self.segments_path.set(path)
        messagebox.showinfo("Файл загружен",
                            f"Отрезков: {len(segments)}\nОкно: ({rect.xmin}, {rect.ymin}) – ({rect.xmax}, {rect.ymax})")

    def run_segments_clip(self):
        if not self.segments or self.rect is None:
            messagebox.showwarning("Нет данных", "Сначала выберите файл с отрезками.")
            return
        clipped = []
        for s in self.segments:
            c = cohen_sutherland_clip(s, self.rect)
            if c is not None:
                clipped.append(c)
        plot_segments(self.segments, self.rect, clipped, self.canvas_segments)

    # --- Обработчики многоугольника ---

    def load_subject_poly(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.poly_subject = read_polygon(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать многоугольник:\n{e}")
            return
        self.subject_path.set(path)
        messagebox.showinfo("Загружено", f"Исходный многоугольник: {len(self.poly_subject)} вершин")

    def load_clipper_poly(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.poly_clipper = read_polygon(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать клипер:\n{e}")
            return
        self.clipper_path.set(path)
        messagebox.showinfo("Загружено", f"Клипер: {len(self.poly_clipper)} вершин")

    def run_polygon_clip(self):
        if not self.poly_subject or not self.poly_clipper:
            messagebox.showwarning("Нет данных", "Загрузите исходный многоугольник и клипер.")
            return
        clipped = sutherland_hodgman_clip_polygon(self.poly_subject, self.poly_clipper)
        plot_polygon(self.poly_subject, self.poly_clipper, clipped, self.canvas_poly)

def main():
    app = ClipApp()
    app.mainloop()

if __name__ == "__main__":
    main()
