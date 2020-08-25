import numpy as np
import cv2
import matplotlib.pyplot as plt

def imshow(image, ax=None, points=None, marker='r.'):

    show = False
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        show = True

    ax.imshow(image)
    if points is not None:
        if isinstance(points, list):
            for p in range(len(points)):
                ax.plot(points[p][:, 0], points[p][:, 1], marker[p])
        else:
            ax.plot(points[:, 0], points[:, 1], marker)

    ax.axis('off')

    if show:
        plt.show()
