import cv2
import matplotlib.pyplot as plt
import numpy as np


def rearrange(cnt):
    '''
    Function to rearrange the contour bounding boxes. in default the contour bounding boxes comes in the sorted order of
    their y co-ordinates . this function returns a list of rectangles [(x1,y1,w1,h1),(x2,y2,w2,h2)...] which are sorted in
    the order of x axis on each line. a line will have all recangles of y coordinates between y and y+h of first rectangle

    '''

    b_rect = []
    for c in cnt:
        rect = cv2.boundingRect(c)
        if rect[2] <= 18 or rect[3] <= 18:
            continue
        b_rect.append(rect)
    if b_rect == []:
        return []
    p = b_rect[0][1]+b_rect[0][3]
    #print('length of brect:',len(b_rect))
    s_rect = []
    i = 0
    length = len(b_rect)
    while i < length:
        p = b_rect[i][1]+b_rect[i][3]
        elem_on_line = []  # elements on a line
        outer = True
        while i < length and p > b_rect[i][1]:
            elem_on_line.append(b_rect[i])
            i += 1
            outer = False
        if outer:
            i += 1
        elem_on_line = sorted(elem_on_line)  # ,key=lambda x:x[0]
        # print(elem_on_line,i)
        s_rect.extend(elem_on_line)
    return s_rect


def charactersegmentation(words, border=50, ellipse=(3, 3), rect=(2, 2)):

    threshold = 128
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ellipse)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, rect)
    seperated = []
    contours = []

    for word in words:
        grad = cv2.morphologyEx(word, cv2.MORPH_GRADIENT, kernel)
        _, bw = cv2.threshold(grad, threshold, 255.0,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel2)
        cnt, _ = cv2.findContours(
            connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # reversing contour list to start processig from top
        cnt.reverse()
        contours.append(cnt)

        s_rect = rearrange(cnt)
        chars = []
        if s_rect == []:
            continue
        for rect in s_rect:

            # Filtering character
            x, y, w, h = rect
            char = word[y:y+h, x:x+w]
            char = cv2.copyMakeBorder(
                char, border, border, border, border, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            char = cv2.resize(char, (86, 86))
            chars.append(char)

        seperated.append(chars)

    return contours, seperated
