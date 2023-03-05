import cv2

def save_image(image,filename):
    cv2.imwrite(filename, image)
