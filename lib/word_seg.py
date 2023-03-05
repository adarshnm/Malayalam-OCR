import cv2
import numpy as np
from  wordsegmentation import wordsegmentation

img = cv2.imread("data/binarized_test_data/inp6.jpg", 0)


s_rect, words = wordsegmentation(img)

print(s_rect)
