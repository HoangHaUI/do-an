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




class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("UI/doan.ui", self)
        self.btnRun.clicked.connect(self.DeleteOb)
        self.btnGetImage.clicked.connect(self.GetImg)
        
    def DeleteOb(self):
        print("PUSH BUTTON OK :v")
        
    def GetImg(self):        
        print('Get image')
        appli= App()
        self.lbImage.setScaledContents(True)
        self.lbImage.setPixmap(QPixmap(appli.ImgPath))


class App(QWidget):
    
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
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