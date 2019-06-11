import tkinter as tk
import os

class Canvas:

    IMG_WIDTH = 640
    IMG_HEIGHT = 480

    def __init__(self, width, height):
        self.root = tk.Tk()
        self.root.title('Sztuczna bezinteligencja')

        self.width = width
        self.height = height

        self.canvas = tk.Canvas(self.root,width=self.width, height=self.height)
        self.canvas.pack()

        self.imgs = []

    def loadImages(self, img_list):
        for img in img_list:
            self.imgs.append(tk.PhotoImage(file=os.getcwd() + '\\img\\' + img))

    def paintImages(self):
        self.canvas.create_image(self.getCenters()[0][0], self.getCenters()[0][1], image=self.imgs[0])
        self.canvas.create_image(self.getCenters()[1][0], self.getCenters()[1][1], image=self.imgs[1])

    def getCenters(self):
        centerL = (Canvas.IMG_WIDTH / 2, Canvas.IMG_HEIGHT / 2)
        centerR = (self.width - Canvas.IMG_WIDTH / 2, Canvas.IMG_HEIGHT / 2)
        return (centerL, centerR)

    def paintPoint(self, x, y, width, color):
        self.canvas.create_oval(x, y, x + width, y + width, fill=color, outline='')

    def paintLine(self, x1, y1, x2, y2, width, color):
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def loop(self):
        self.canvas.mainloop()


