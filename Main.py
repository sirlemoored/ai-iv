import subprocess
import os
import time
import pickle
import math
import sys
import tkinter as tk
import numpy as np
import scipy.spatial.distance as dist
import Canvas as cv

class ImagePair:

    # img1, img2 --> STRING

    def __init__(self, img_src_1, img_src_2):
        self.img1 = img_src_1
        self.img2 = img_src_2
        self.coords1 = None
        self.props1 = None
        self.coords2 = None
        self.props2 = None

    def loadFiles(self):
        process = subprocess.Popen(
            "powershell.exe ./extract_features/extract_features_32bit.exe -haraff -sift -i " + "\'" + os.getcwd() + "\\img\\" + self.img1 + "\'" + " ")
        process.wait()
        print("Image successfully processed: ", self.img1)
        process = subprocess.Popen(
            "powershell.exe ./extract_features/extract_features_32bit.exe -haraff -sift -i " + "\'" + os.getcwd() + "\\img\\" + self.img2 + "\'" + " ")
        process.wait()
        print("Image successfully processed: ", self.img2)

    def loadMatrices(self):
        file = open("img/" + self.img1 + ".haraff.sift", "r")
        dimX = int(file.readline())
        dimY = int(file.readline())
        self.coords1 = np.zeros((dimY, 2))
        self.props1 = np.zeros((dimY, dimX))

        i = 0
        for line in file:
            initial_list = [float(x) for x in line.split()]
            self.coords1[i] = [initial_list[0], initial_list[1]]
            self.props1[i] = initial_list[5:]
            i += 1
        file.close()

        file = open("img/" + self.img2 + ".haraff.sift", "r")
        dimX = int(file.readline())
        dimY = int(file.readline())
        self.coords2 = np.zeros((dimY, 2))
        self.props2 = np.zeros((dimY, dimX))

        i = 0
        for line in file:
            initial_list = [float(x) for x in line.split()]
            self.coords2[i] = [initial_list[0], initial_list[1]]
            self.props2[i] = initial_list[5:]
            i += 1
        file.close()

    def computeDistanceMatrix(self):
        matrix = dist.cdist(self.props1, self.props2, 'minkowski', p=1)
        pickle.dump(matrix, open(self.img1[:-4] + self.img2[:-4] + '_distances', mode='wb'))
        return matrix

    def loadDistanceMatrix(self):
        matrix = pickle.load(open(self.img1[:-4] + self.img2[:-4] + '_distances', mode='rb'))
        return matrix

if __name__ == '__main__':

    imgs = ImagePair("sky2.ppm", "sky1.ppm")
    #imgs.loadFiles()
    imgs.loadMatrices()
    distanceMatrix = imgs.loadDistanceMatrix()
    closestNeighborsA = np.argmin(distanceMatrix, axis=0)
    closestNeighborsB = np.argmin(distanceMatrix, axis=1)

    pairs = []

    for x in range (0, closestNeighborsA.shape[0]):
        a = x
        neighborA = closestNeighborsB[x]
        if (closestNeighborsA[neighborA] == a):
            pairs.append((x, closestNeighborsB[x]))

    print(len(pairs))
    canvas = cv.Canvas(1300, 500)
    canvas.loadImages([imgs.img1, imgs.img2])
    canvas.paintImages()
    for pair in pairs:

        indexLeft = pair[0]
        indexRight = pair[1]

        x1 = imgs.coords1[indexLeft][0]
        y1 = imgs.coords1[indexLeft][1]
        x2 = canvas.width - cv.Canvas.IMG_WIDTH + imgs.coords2[indexRight][0]
        y2 = imgs.coords2[indexRight][1]

        canvas.paintPoint(x1, y1, 5, 'red')
        canvas.paintPoint(x2, y2, 5, 'red')
        canvas.paintLine(x1, y1, x2, y2, 1, 'white')
    canvas.loop()
