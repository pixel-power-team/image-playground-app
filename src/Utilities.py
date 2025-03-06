import cv2
from matplotlib import pyplot as plt


def showHistogram(img):
    color = ('r', 'g', 'b')
    if len(img.shape) == 2:
        color = ('b')

    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

def calculate_histogram(img):
    color = ('r', 'g', 'b')
    histogram = [0, 0, 0]
    if len(img.shape) == 2:
        color = ('b')
        histogram = [0]


    for i, col in enumerate(color):
        histogram[i] = cv2.calcHist([img], [i], None, [256], [0, 256])

    return histogram

def plotHistogramVector(histogram):
    plt.plot(histogram, color='b')
    plt.xlim([0, len(histogram)])
    plt.show()


def resize_image(img, height):
    aspect_ratio = img.shape[0] / img.shape[1]
    dim = (height, int(height * aspect_ratio))
    return cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

def resize_image(img, width, height, interpolation=cv2.INTER_NEAREST):
    dim = (width, height)
    return cv2.resize(img, dim, interpolation)

def grabWebcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, im = cap.read()
        cv2.imshow('video test', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
        # if key == ord(' '):
        # cv2.imwrite('vid_result.jpg',im)

def ensure_one_channel_grayscale_image(img):
    if len(img.shape) == 2:
        # Das Bild ist bereits in Graustufen
        return img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # Das Bild ist farbig (3 Kanäle: BGR oder RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Ungültiges Bildformat.")

def ensure_three_channel_grayscale_image(gray_scale_img):
    if len(gray_scale_img.shape) == 2:
        img = cv2.cvtColor(gray_scale_img, cv2.COLOR_GRAY2BGR)
        # Das Bild ist bereits in Graustufen
        return img
    elif len(gray_scale_img.shape) == 3 and gray_scale_img.shape[2] == 3:
        # Das Bild ist farbig (3 Kanäle: BGR oder RGB)
        img = cv2.cvtColor(gray_scale_img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Ungültiges Bildformat.")