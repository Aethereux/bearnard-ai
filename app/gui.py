import tkinter as tk
from tkinter import PhotoImage

# ---- Rounded Rectangle Function ---- #
def round_rectangle(canvas, x1, y1, x2, y2, r=35, **kwargs):
    points = [
        x1+r, y1,
        x2-r, y1,
        x2, y1,
        x2, y1+r,
        x2, y2-r,
        x2, y2,
        x2-r, y2,
        x1+r, y2,
        x1, y2,
        x1, y2-r,
        x1, y1+r,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


root = tk.Tk()
root.title("BearNard Assistant UI")
root.geometry("1050x600")
root.configure(bg="#e0a526")

panel = tk.Frame(root, bg="white", bd=0)
panel.place(relx=0.5, rely=0.5, anchor="center", width=1000, height=550)


root.mainloop()
