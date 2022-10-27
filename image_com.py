import os
import random
import imutils as im
import numpy as np
from cv2 import cv2
from tqdm.auto import tqdm


def get_img_contours(img_path):
    img_names = os.listdir(img_path)
    output = {}

    for img_name in tqdm(img_names):
        image = cv2.imread(img_path + img_name)

        # Resize the image - change width to 500
        newwidth = 500

        # RGB to Gray scale conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise removal with iterative bilateral filter(removes noise while preserving edges)
        d, sigmaColor, sigmaSpace = 11, 17, 17
        filtered_img = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

        # Find Edges of the grayscale image
        lower, upper = 15, 25
        edged = cv2.Canny(filtered_img, lower, upper)

        # Find contours based on Edges
        cnts, hir = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        NumberPlateCnt = None

        # loop over our contours to find the best possible approximate contour of number plate
        count = 0
        for c in cnts:
            peri = cv2.arcLength(c, True)

            epsilon = 0.01 * peri
            approx = cv2.approxPolyDP(c, epsilon, True)

            if len(approx) == 4:  # Select the contour with 4 corners
                # print(approx)
                NumberPlateCnt = approx  # This is our approx Number Plate Contour
                break

        if NumberPlateCnt is not None:
            output[img_name] = NumberPlateCnt
            # print(img_name, NumberPlateCnt)
        else:
            # print("LP is not detected.")
            pass

    return output


if __name__ == '__main__':
    origin_img_path = 'data/plates/train/a/'
    imgs = get_img_contours(origin_img_path)


