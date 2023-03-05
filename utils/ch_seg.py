from pickle import FALSE
import numpy as np
import cv2


def character_segmentation(img_loc, morph_close=FALSE, **kwargs):
    default_kwargs = {
        'ellipse': (3, 3),
        'rect': (3, 3),

    }
    kwargs = {**default_kwargs, **kwargs}
    image = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kwargs["ellipse"])
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, kwargs['rect'])
    threshold = 128

    chars = []

    #     img = cv2.bitwise_not(word)
    char_img = image.copy()
    grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    #     word_copy = cv2.bitwise_not(word)
    _, bw = cv2.threshold(grad, threshold, 255.0,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel2)
    contours, hierarchy = cv2.findContours(
        connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # reversing contour list to start processig from top
    contours.reverse()
    b_rect = []
    cnts = []
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] <= 10 or rect[3] <= 10:
            continue
        b_rect.append(rect)
        cnts.append(c)
    sorted_cnt = [[y, x] for y, x in sorted(
        zip(b_rect, cnts), key=lambda pair: pair[0])]
    for i, [rect, cnt] in enumerate(sorted_cnt):
        x, y, w, h = rect
        cv2.rectangle(char_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # removing overlap
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
        new_image = cv2.bitwise_and(cv2.bitwise_not(image), mask)
        if morph_close:
            kernel = np.ones((3, 3), np.uint8)
            new_image = cv2.morphologyEx(new_image, cv2.MORPH_CLOSE, kernel)
        new_image = cv2.bitwise_not(new_image)

        char = new_image[y:y+h, x:x+w]
        char = cv2.copyMakeBorder(
            char, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        char = cv2.resize(char, (224, 224))
        # cv2.imwrite(os.path.join("out", img_name, f"{i:03d}.jpg"), char)
        chars.append(char)
    return chars
