import tensorflow as tf
import numpy as np
from src.training.load_model import load_model
import sys
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6 import uic
from PyQt6.QtWidgets import (QMainWindow, QTextEdit,
        QFileDialog, QApplication)
from PyQt6.QtGui import QPixmap


from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import QWidget

import cv2 as cv



checkpoint_dir = "runs/training/"
# Load model 
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
# Create a new model instance
model = load_model(2)

# Load the previously saved weights
model.load_weights(latest)

def predict(path):

    print(path)
    # Evaluate
    img = tf.keras.utils.load_img(
        path, target_size=(244, 244)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    print(np.max(img_array[0]), np.min(img_array[0]))
    
    predictions = None
    with tf.device('/cpu:0'):
        predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # _,_,class_names = get_dataset()

    class_names = ['CHUAN', 'THIEU NHAN']
    labelResult = str(class_names[np.argmax(score)])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(labelResult, 100 * np.max(score))
    )
    return labelResult
    # break



class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("UI/doan.ui", self)
        self.btnRun.clicked.connect(self.DeleteOb)
        self.btnGetImage.clicked.connect(self.GetImg)
        self.imgPath = ""
        
    def DeleteOb(self):
        if self.imgPath == "":
            print("Need to chose image!!!")
            return
        self.lbResult.setText("PREDICTING")
        ret = predict(self.imgPath)
        if ret == "CHUAN":
            self.lbResult.setStyleSheet('background-color: GREEN')
        if ret == "THIEU NHAN":
            self.lbResult.setStyleSheet('background-color: RED')
        self.lbResult.setText(ret)
        
        
    def GetImg(self):        
        print('Get image')
        appli= App()
        self.lbImage.setScaledContents(True)
        self.imgPath = appli.ImgPath
        self.lbImage.setPixmap(QPixmap(appli.ImgPath))


class App(QWidget):
    
    def __init__(self):
        super().__init__()
        self.title = ''
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height) 
        self.ImgPath= self.openFileNameDialog()
    
    def openFileNameDialog(self):
        # options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,
        "QFileDialog.getOpenFileName()", "","All Files (*);;Image Files (*.jpg *.png)")
        return fileName
    

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()