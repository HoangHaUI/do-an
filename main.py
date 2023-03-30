import tensorflow as tf
import numpy as np
from src.training.load_model import load_model
from src.training.load_image import get_dataset

checkpoint_dir = "./runs/training/"
# Load model 
latest = tf.train.latest_checkpoint(checkpoint_dir)

# Create a new model instance
model = load_model(2)

# Load the previously saved weights
model.load_weights(latest)

print(model.summary())

# import sys
# from PyQt6 import QtCore, QtGui, QtWidgets
# from PyQt6 import uic
# from PyQt6.QtWidgets import (QMainWindow, QTextEdit,
#         QFileDialog, QApplication)
# from PyQt6.QtGui import QPixmap


# from PyQt6.QtCore import *
# from PyQt6.QtGui import *
# from PyQt6.QtWidgets import QWidget

# import cv2 as cv



# class MainWindow(QtWidgets.QMainWindow):
    

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         uic.loadUi("UI/doan.ui", self)
#         self.btnRun.clicked.connect(self.predict())
#         self.btnGetImage.clicked.connect(self.GetImg)
#         self.img_path = ""
        
#     def predict(self):
#         print("PUSH BUTTON OK :v")
        
#     def GetImg(self):        
#         print('Get image')
#         appli= App()
#         self.lbImage.setScaledContents(True)
#         self.img_path = appli.ImgPath
#         self.lbImage.setPixmap(QPixmap(appli.ImgPath))


# class App(QWidget):
    
#     def __init__(self):
#         super().__init__()
#         self.title = 'PyQt5 file dialogs - pythonspot.com'
#         self.left = 10
#         self.top = 10
#         self.width = 640
#         self.height = 480
#         self.initUI()
 
#     def initUI(self):
#         self.setWindowTitle(self.title)
#         self.setGeometry(self.left, self.top, self.width, self.height) 
#         self.ImgPath= self.openFileNameDialog()
    
#     def openFileNameDialog(self):
#         # options = QFileDialog.Options()
#         # options |= QFileDialog.DontUseNativeDialog
#         fileName, _ = QFileDialog.getOpenFileName(self,
#         "QFileDialog.getOpenFileName()", "","All Files (*);;Image Files (*.jpg *.png)")
#         return fileName
    

# app = QtWidgets.QApplication(sys.argv)
# window = MainWindow()
# window.show()
# app.exec()