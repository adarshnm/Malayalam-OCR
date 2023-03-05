import cv2
import numpy as np

from matplotlib import pyplot as plt
import math
import pandas
import pickle
import gc
from sklearn.model_selection import train_test_split


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


def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4, 6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def load_train_test_data():

    pickle_in = open("data/dataset/dataset_121.pickle", "rb")
    dataset = pickle.load(pickle_in)
    dataset = np.array(dataset)

    X = np.array([i[0]for i in dataset])
    y = np.array([i[1] for i in dataset])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=15)
    del dataset
    del X
    del y
    gc.collect()
    return (X_train, y_train), (X_test, y_test)


def load_train_test_val_data(test_size=0.2, val_size=0.15):
    pickle_in = open("data/dataset/dataset_121.pickle", "rb")
    dataset = pickle.load(pickle_in)
    dataset = np.array(dataset)

    X = np.array([i[0]for i in dataset])
    y = np.array([i[1] for i in dataset])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=10)
    del dataset
    del X
    del y
    gc.collect()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_val = X_val.astype('float32') / 255.
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    y_val = y_val.astype('float32')
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def load_48x48_train_test_val_data(test_size=0.2, val_size=0.15):
    pickle_in = open("data/dataset/dataset_48x48_121.pickle", "rb")
    dataset = pickle.load(pickle_in)
    dataset = np.array(dataset)

    X = np.array([i[0]for i in dataset])
    y = np.array([i[1] for i in dataset])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=10)
    del dataset
    del X
    del y
    gc.collect()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_val = X_val.astype('float32') / 255.
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    y_val = y_val.astype('float32')
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def load_28x28_train_test_val_data(test_size=0.2, val_size=0.15):
    pickle_in = open("data/dataset/dataset_28x28_121.pickle", "rb")
    dataset = pickle.load(pickle_in)
    dataset = np.array(dataset)

    X = np.array([i[0]for i in dataset])
    y = np.array([i[1] for i in dataset])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=10)
    del dataset
    del X
    del y
    gc.collect()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_val = X_val.astype('float32') / 255.
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    y_val = y_val.astype('float32')
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def load_32x32_train_test_data(test_size=0.2):

    pickle_in = open("data/dataset/dataset_32x32_121_min.pickle", "rb")
    dataset = pickle.load(pickle_in)
    dataset = np.array(dataset)

    X = np.array([i[0]for i in dataset])
    y = np.array([i[1] for i in dataset])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=15)
    del dataset
    del X
    del y
    gc.collect()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    return (X_train, y_train), (X_test, y_test)


def load_32x32_min_train_test_val_data(test_size=0.2, val_size=0.15):
    pickle_in = open("data/dataset/dataset_32x32_121_min.pickle", "rb")
    dataset = pickle.load(pickle_in)
    dataset = np.array(dataset)

    X = np.array([i[0]for i in dataset])
    y = np.array([i[1] for i in dataset])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=10)
    del dataset
    del X
    del y
    gc.collect()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_val = X_val.astype('float32') / 255.
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    y_val = y_val.astype('float32')
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def load_48x48_min_train_test_val_data(test_size=0.2, val_size=0.15):
    pickle_in = open("data/dataset/dataset_48x48_121_min.pickle", "rb")
    dataset = pickle.load(pickle_in)
    dataset = np.array(dataset)

    X = np.array([i[0]for i in dataset])
    y = np.array([i[1] for i in dataset])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=10)
    del dataset
    del X
    del y
    gc.collect()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_val = X_val.astype('float32') / 255.
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    y_val = y_val.astype('float32')
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def load_86x86_min_train_test_val_data(test_size=0.2, val_size=0.15,random_state = 10):
    pickle_in = open("data/dataset/dataset_86x86_121_min.pickle", "rb")
    dataset = pickle.load(pickle_in)
    dataset = np.array(dataset)

    X = np.array([i[0]for i in dataset])
    y = np.array([i[1] for i in dataset])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state)
    del dataset
    del X
    del y
    gc.collect()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_val = X_val.astype('float32') / 255.
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    y_val = y_val.astype('float32')
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)
