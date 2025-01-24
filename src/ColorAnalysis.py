import cv2
import numpy as np
import Utilities
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans



class ColorAnalysis:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, img, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = img

    def show3DHistogram(self):
        #img = Utilities.resize_image(img, 100)

        # resize image# convert from BGR to RGB
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get rgb values from image to 1D array
        r, g, b = cv2.split(self.IMAGE)
        r = r.flatten()
        g = g.flatten()
        b = b.flatten()

        # plotting
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(r, g, b)
        plt.show()

    def dominantColors(self):
        img = self.IMAGE
        #img = Utilities.resize_image(img, 100)
        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_

        # returning after converting to integer from float
        return self.COLORS.astype(int)

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def plotClusters(self):
        # plotting
        fig = plt.figure()
        ax = Axes3D(fig)
        for label, pix in zip(self.LABELS, self.IMAGE):
            ax.scatter(pix[0], pix[1], pix[2], color=self.rgb_to_hex(self.COLORS[label]))
        plt.show()

def create_visual_output(colors, width, height_max):
    n_clusters = len(colors)
    visual_output = 255*np.ones((height_max, width, 3), np.uint8)
    height = int(width / n_clusters)
    box_width = int(width / n_clusters)
    if height < height_max:
        visual_output = 255*np.ones((height, width, 3), np.uint8)

    for i in range(n_clusters):
        visual_output[:, box_width*i:box_width*(i+1)]= colors[i]
    return visual_output