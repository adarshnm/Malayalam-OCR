# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qtGui.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

from lib import sauvola
from lib import WordSegmentation as ws
from lib import niblack
import sys
import cv2
import os


class Ui_MainWindow(object):
    data_folder = 'data/binarized_test_data/cropped/'
    bin_methods = ['Sauvola', 'Otsu', "Niblack"]

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 860)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.binButton = QtWidgets.QPushButton(self.centralwidget)
        self.binButton.setGeometry(QtCore.QRect(540, 190, 121, 51))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        cv2.createTrackbar("Max", "Colorbars", 0, 255, self.nothing)
        self.binButton.setFont(font)
        self.binButton.setObjectName("binButton")
        self.binButton.clicked.connect(self.segmentation)
        self.image = QtWidgets.QComboBox(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(90, 130, 351, 31))
        self.image.setObjectName("image")
        self.refresh_images()
        self.image.currentTextChanged.connect(self.image_selection_changed)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 90, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        self.method = QtWidgets.QComboBox(self.centralwidget)
        self.method.setGeometry(QtCore.QRect(790, 130, 300, 31))
        self.method.setObjectName("method")
        self.method.addItems(self.bin_methods)
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(820, 90, 151, 31))
        self.inputImage = QtWidgets.QLabel(self.centralwidget)
        self.inputImage.setGeometry(QtCore.QRect(20, 290, 750, 491))
        self.inputImage.setFrameShape(QtWidgets.QFrame.Box)
        self.inputImage.setText("")
        self.inputImage.setObjectName("inputImage")

        # setting image

        self.binImage = QtWidgets.QLabel(self.centralwidget)
        self.binImage.setGeometry(QtCore.QRect(800, 290, 750, 491))
        self.binImage.setFrameShape(QtWidgets.QFrame.Box)
        self.binImage.setText("")
        self.binImage.setObjectName("binImage")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1016, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.exitAction)
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())
        self.actionRefresh = QtWidgets.QAction(MainWindow)
        self.actionRefresh.setObjectName("actionRefresh")
        self.actionRefresh.triggered.connect(self.refresh_images)
        self.menuFile.addAction(self.actionRefresh)
        self.menubar.addAction(self.menuFile.menuAction())
        self.image_selection_changed()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def image_selection_changed(self):
        pmap = QtGui.QPixmap(self.data_folder + self.image.currentText())
        pmap = pmap.scaled(self.inputImage.width(),
                           self.inputImage.height(), QtCore.Qt.KeepAspectRatio)
        self.inputImage.setPixmap(pmap)
        self.inputImage.setAlignment(QtCore.Qt.AlignCenter)

    def refresh_images(self):
        data_files = os.listdir(self.data_folder)
        self.image.addItems(data_files)

    def segmentation(self):
        output_dir = 'out/words/'
        file_name = self.image.currentText()
        img = cv2.imread(self.data_folder + file_name, cv2.IMREAD_GRAYSCALE)
        if self.method.currentText() == "Sauvola":
            bin_img = sauvola.binarize(img, [20, 20], 128, 0.194)
        elif self.method.currentText() == "Niblack":
            bin_img = niblack.binarize(
                image=img, size=[50, 50], offset=9, padding='symmetric')
        else:
            __, bin_img = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        word_segment = ws.word_segmentation(bin_img)
        if not os.path.exists(output_dir + '%s' % file_name):
            os.mkdir(output_dir + '%s' % file_name)

        word_image = []
        for (j, w) in enumerate(word_segment):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            word_image.append(wordImg)
            cv2.imwrite(output_dir + '%s/l-%d.png' %
                        (file_name, j), wordImg)  # save word
            # draw bounding box in summary image
            cv2.rectangle(bin_img, (x, y), (x + w, y + h), 0, 1)
        cv2.imwrite(output_dir + '%s/line-summary.jpg' % file_name, bin_img)
        pmap = QtGui.QPixmap(output_dir + file_name + '/line-summary.jpg')
        pmap = pmap.scaled(self.inputImage.width(), self.inputImage.height(), QtCore.Qt.KeepAspectRatio,
                           transformMode=QtCore.Qt.SmoothTransformation)
        self.binImage.setPixmap(pmap)
        self.binImage.setAlignment(QtCore.Qt.AlignCenter)

    def exitAction(self):
        sys.exit(app.exec_())

    def showdialog(self, head, info):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(head)
        msg.setInformativeText(info)
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def nothing(x):
        pass

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "Image Segmentation"))
        self.binButton.setText(_translate("MainWindow", "Segment"))
        self.label.setText(_translate("MainWindow", "Choose Image"))
        self.label2.setText(_translate("MainWindow", "Method"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
