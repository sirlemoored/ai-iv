import subprocess
import os
from PIL import Image
import random
import pickle
import math
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
        self.width = 0
        self.height = 0

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
        pickle.dump(matrix, open("dist/" + self.img1[:-4] + self.img2[:-4] + '_distances', mode='wb'))
        return matrix

    def loadDistanceMatrix(self):
        matrix = pickle.load(open("dist/" + self.img1[:-4] + self.img2[:-4] + '_distances', mode='rb'))
        return matrix

    def loadSize(self):
        im = Image.open(os.getcwd() + "\\img\\" + self.img1)
        self.width = im.size[0]
        self.height = im.size[1]

    def getPointingPairs(self, distanceMatrix):
        closestNeighborsA = np.argmin(distanceMatrix, axis=0)
        closestNeighborsB = np.argmin(distanceMatrix, axis=1)

        pairs = []

        for x in range(0, closestNeighborsA.shape[0]):
            a = x
            a_pointing_at = closestNeighborsA[a]
            if (closestNeighborsB[a_pointing_at] == a):
                pairs.append((a, a_pointing_at))

        return pairs

    def getPairsCohesive(self, pairs, percent_of_total, percent_correct):
        if (percent_of_total < 0 or percent_of_total > 1 or percent_correct < 0 or percent_correct > 1):
            return None

        num_closest = int(len(pairs) * percent_of_total)
        num_correct = int(num_closest * percent_correct)
        pairs_cpy = pairs.copy()

        res_pairs = []

        for pair in pairs:
            closest_points_a = [i[0] for i in sorted(pairs_cpy, reverse=False, key=lambda p: (self.coords2[pair[0]][0] - self.coords2[p[0]][0]) ** 2 + (self.coords2[pair[0]][1] - self.coords2[p[0]][1]) ** 2)  [:num_closest]]
            closest_points_b = [i[0] for i in sorted(pairs_cpy, reverse=False, key=lambda p: (self.coords1[pair[1]][0] - self.coords1[p[1]][0]) ** 2 + (self.coords1[pair[1]][1] - self.coords1[p[1]][1]) ** 2)  [:num_closest]]
            if(len(np.intersect1d(closest_points_a, closest_points_b)) >= num_correct):
                res_pairs.append(pair)
        return res_pairs

    def findClosestPoints(self, pair, pairs_cpy, n):
        closest_points = sorted(pairs_cpy, reverse=False, key=lambda p: (self.coords2[pair[0]][0] -  self.coords2[p[0]][0]) ** 2 + (self.coords2[pair[0]][1] - self.coords2[p[0]][1]) ** 2)[:n]
        return closest_points

    def getPairsAffinic(self, pairs, error_radius, loops, apply_heuristic):
        best_err = math.inf
        best_A = None
        pairs_cpy = pairs.copy()
        for i in range (0, loops):
            try:
                if (apply_heuristic == True):
                    random_point = random.sample(pairs, 1)[0]
                    random_points = self.findClosestPoints(random_point, pairs_cpy, 3)
                else:
                    random_points = random.sample(pairs, 3)
                xy_s = [self.coords2[i[0]] for i in random_points]
                uv_s = [self.coords1[i[1]] for i in random_points]
                A = np.linalg.inv(np.array([
                    np.concatenate((xy_s[0], [1, 0, 0, 0])),
                    np.concatenate((xy_s[1], [1, 0, 0, 0])),
                    np.concatenate((xy_s[2], [1, 0, 0, 0])),
                    np.concatenate(([0, 0, 0], xy_s[0], [1])),
                    np.concatenate(([0, 0, 0], xy_s[1], [1])),
                    np.concatenate(([0, 0, 0], xy_s[2], [1]))
                ])) @ np.array([
                    [uv_s[0][0]],
                    [uv_s[1][0]],
                    [uv_s[2][0]],
                    [uv_s[0][1]],
                    [uv_s[1][1]],
                    [uv_s[2][1]]
                ])
                A = np.linalg.inv(np.reshape(np.concatenate((A, [[0], [0], [1]])), (3,3)))

                total_err = 0

                for pair in pairs:
                    uv1 = self.coords2[pair[0]]
                    xy1 = self.coords1[pair[1]]

                    new_uv = A @ np.array([[xy1[0]], [xy1[1]], [1]])
                    err = np.sqrt((uv1[0] - new_uv[0][0]) ** 2 + (uv1[1] - new_uv[1][0]) ** 2)
                    total_err += err

                if (total_err < best_err):
                    best_A = A
                    best_err = total_err
                    print(i, best_err)
            except:
                print(i)
        res = []
        for pair in pairs:
            xy1 = self.coords2[pair[0]]
            uv1 = self.coords1[pair[1]]
            new_uv = best_A @ np.array([[uv1[0]], [uv1[1]], [1]])
            err = np.sqrt((uv1[0] - new_uv[0][0]) ** 2 + (uv1[1] - new_uv[1][0]) ** 2)
            if (err < error_radius):
                res.append((uv1[0], uv1[1], new_uv[0][0], new_uv[1][0]))
        return res

    def getPairsPerspective(self, pairs, error_radius, loops, apply_heuristic):
        best_err = math.inf
        best_H = None
        pairs_cpy = pairs.copy()
        for i in range (0, loops):
            try:
                if (apply_heuristic == True):
                    random_point = random.sample(pairs, 1)[0]
                    random_points = self.findClosestPoints(random_point, pairs_cpy, 4)
                else:
                    random_points = random.sample(pairs, 4)
                xy_s = [self.coords2[i[0]] for i in random_points]
                uv_s = [self.coords1[i[1]] for i in random_points]

                H = np.linalg.inv(np.array([
                    np.concatenate((xy_s[0], [1, 0, 0, 0], [-uv_s[0][0] * xy_s[0][0], -uv_s[0][0] * xy_s[0][1]])),
                    np.concatenate((xy_s[1], [1, 0, 0, 0], [-uv_s[1][0] * xy_s[1][0], -uv_s[1][0] * xy_s[1][1]])),
                    np.concatenate((xy_s[2], [1, 0, 0, 0], [-uv_s[2][0] * xy_s[2][0], -uv_s[2][0] * xy_s[2][1]])),
                    np.concatenate((xy_s[3], [1, 0, 0, 0], [-uv_s[3][0] * xy_s[3][0], -uv_s[3][0] * xy_s[3][1]])),
                    np.concatenate(([0, 0, 0], xy_s[0], [1], [-uv_s[0][1] * xy_s[0][0], -uv_s[0][1] * xy_s[0][1]])),
                    np.concatenate(([0, 0, 0], xy_s[1], [1], [-uv_s[1][1] * xy_s[1][0], -uv_s[1][1] * xy_s[1][1]])),
                    np.concatenate(([0, 0, 0], xy_s[2], [1], [-uv_s[2][1] * xy_s[2][0], -uv_s[2][1] * xy_s[2][1]])),
                    np.concatenate(([0, 0, 0], xy_s[3], [1], [-uv_s[3][1] * xy_s[3][0], -uv_s[3][1] * xy_s[3][1]]))
                ])) @ np.array([
                    [uv_s[0][0]],
                    [uv_s[1][0]],
                    [uv_s[2][0]],
                    [uv_s[3][0]],
                    [uv_s[0][1]],
                    [uv_s[1][1]],
                    [uv_s[2][1]],
                    [uv_s[3][1]]
                ])

                H = np.concatenate((np.squeeze(H), [1]))

                H = np.linalg.inv(H.reshape((3,3)))

                total_err = 0

                for pair in pairs:
                    uv1 = self.coords2[pair[0]]
                    xy1 = self.coords1[pair[1]]

                    new_uv = H @ np.array([[xy1[0]], [xy1[1]], [1]])
                    err = np.sqrt((uv1[0] - new_uv[0][0]) ** 2 + (uv1[1] - new_uv[1][0]) ** 2)
                    total_err += err

                if (total_err < best_err):
                    best_H = H
                    best_err = total_err
            except:
                print(i)
        res = []
        for pair in pairs:
            xy1 = self.coords2[pair[0]]
            uv1 = self.coords1[pair[1]]
            new_uv = best_H @ np.array([[uv1[0]], [uv1[1]], [1]])
            err = np.sqrt((uv1[0] - new_uv[0][0]) ** 2 + (uv1[1] - new_uv[1][0]) ** 2)
            if (err < error_radius):
                res.append((uv1[0], uv1[1], new_uv[0][0], new_uv[1][0]))
        return res


if __name__ == '__main__':

    imgs = ImagePair("sky1.ppm", "sky2.ppm")
    #imgs.loadFiles()
    imgs.loadSize()
    imgs.loadMatrices()
    distanceMatrix = imgs.loadDistanceMatrix()
    pairs = imgs.getPointingPairs(distanceMatrix)
    newPairs = imgs.getPairsAffinic(pairs, error_radius=10000, loops=1000, apply_heuristic=False)
    #newPairs = imgs.getPairsCohesive(pairs, percent_of_total=0.2, percent_correct=0.9)
    #newPairsPersp = imgs.getPairsPerspective(pairs, error_radius=10, loops=500, apply_heuristic=True)

    canvas = cv.Canvas(2 * imgs.width + 20, 1.1 * imgs.height, imgs.width, imgs.height)
    canvas.loadImages((imgs.img1, imgs.img2))
    canvas.paintImages()
    #canvas.paintPairs(newPairs, imgs.coords1, imgs.coords2, 'red', 'red', 'lime')
    canvas.paintPairsAffPersp(newPairs, 'green', 'green', 'blue')

    canvas.loop()
