import cv2
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


def splMean(img, thresh):
    sum = 0
    nt = 0
    for row in img:
        for elem in row:
            if elem > thresh:
                sum += elem
            else:
                nt += 1
    if sum != 0:
        avg = sum/(img.size-nt)
    else:
        avg = 0
    # print(avg)
    return avg


def wordsegmentation(bin_image, dilation_rectangle=(16, 5), closing_rectangle=(6, 3), dilation_iterations=3):
    assert bin_image is not None

    """(row, col) = bin_image.shape
    resize_shape = (int(col * 0.7), int(row * 0.9))
    image = cv2.resize(bin_image, resize_shape) """
    image = bin_image.copy()
    img = image.copy()

    # Dilation
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, dilation_rectangle)
    img = cv2.bitwise_not(img)
    dilate = cv2.dilate(img, dilation_kernel, iterations=dilation_iterations)

    # Further noise removal
    threshold = 128
    _, bw = cv2.threshold(dilate, threshold, 255.0,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Closing
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, closing_rectangle)
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, closing_kernel)

    # Contours
    contours, _ = cv2.findContours(
        connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.reverse()

    sorted_rectangles = rearrange(contours)

    words = []
    for rect in sorted_rectangles:
        x, y, w, h = rect
        word = (image[y:y+h, x:x+w])
        _, thresh4 = cv2.threshold(word, 127, 255, cv2.THRESH_TOZERO)
        # inc is the increment for differ
        inc = 1*(255-splMean(thresh4, 90))
        word = np.array([[min(j+inc, 255) if j > 90 else j for j in thresh4[k]]
                         for k in range(len(thresh4))], dtype=np.uint8)
        words.append(word)

    return image, contours, words
