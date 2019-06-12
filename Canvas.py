import tkinter as tk
import os

class Canvas:


    def __init__(self, width, height, imgwidth, imgheight):
        self.root = tk.Tk()
        self.root.title('Sztuczna bezinteligencja')

        self.width = width
        self.height = height

        self.canvas = tk.Canvas(self.root,width=self.width, height=self.height)
        self.canvas.pack()

        self.imgs = []

        self.IMG_WIDTH = imgwidth
        self.IMG_HEIGHT = imgheight

    def loadImages(self, img_list):
        for img in img_list:
            self.imgs.append(tk.PhotoImage(file=os.getcwd() + '\\img\\' + img))

    def paintImages(self):
        self.canvas.create_image(self.getCenters()[0][0], self.getCenters()[0][1], image=self.imgs[0])
        self.canvas.create_image(self.getCenters()[1][0], self.getCenters()[1][1], image=self.imgs[1])

    def getCenters(self):
        centerL = (self.width / 4, self.height / 2)
        centerR = (self.width * 3 / 4, self.height / 2)
        return (centerL, centerR)

    def paintPoint(self, x, y, width, color):
        self.canvas.create_oval(x, y, x + width, y + width, fill=color, outline='')

    def paintLine(self, x1, y1, x2, y2, width, color):
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def paintPairs(self, pairs, coords1, coords2, lcolor, rcolor, linecolor):
        for pair in pairs:
            indexLeft = pair[1]
            indexRight = pair[0]

            x1 = coords1[indexLeft][0]
            y1 = coords1[indexLeft][1]
            x2 = self.width - self.IMG_WIDTH + coords2[indexRight][0]
            y2 = coords2[indexRight][1]

            if (linecolor is not None):
                self.paintLine(x1, y1, x2, y2, 1, linecolor)
            if (lcolor is not None):
                self.paintPoint(x1, y1, 5, lcolor)
            if (rcolor is not None):
                self.paintPoint(x2, y2, 5, rcolor)

    def loop(self):
        self.canvas.mainloop()


