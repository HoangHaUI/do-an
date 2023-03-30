
from PyQt5 import QtCore, QtGui, QtWidgets
import operate
import sys
import cv2 as cv
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtGui import QIcon

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1137, 860)
        MainWindow.setWindowIcon(QIcon(QPixmap(
"icon/finder.jpg")))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.result = QtWidgets.QPushButton(
self.centralwidget)
        self.result.setGeometry(QtCore.QRect(20, 20, 25, 25))
        self.result.setText("")
        self.result.setObjectName("pushButton")
        self.result.setIcon(QIcon(QPixmap(
"icon/Button-Next-icon.png")))
        self.result.setIconSize(QtCore.QSize(25,25))

        self.groupBox = QtWidgets.QGroupBox(
self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(
0, 60, 941, 761))
        self.groupBox.setObjectName("groupBox")
        self.display_image = QtWidgets.QLabel(self.groupBox)
        self.display_image.setGeometry(QtCore.QRect(10, 20, 921, 721))
        self.display_image.setText("")
        self.display_image.setObjectName("label")
        self.display_image.setAlignment(QtCore.Qt.
AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)

        self.image = QtWidgets.QToolButton(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(60, 20, 28, 28))
        self.image.setText("")
        self.image.setObjectName("toolButton")
        self.image.setIcon(QIcon(QPixmap(
"icon/Button-Add-icon.png")))
        self.image.setIconSize(QtCore.QSize(25,25))
        self.groupBox_2 = QtWidgets.QGroupBox(
self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(
3, 5, 81, 51))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(
970, 50, 81, 16))
        self.label_2.setObjectName("label_2")
        self.object_num = QtWidgets.QLabel(
self.centralwidget)
        self.object_num.setGeometry(QtCore.QRect(
1070, 52, 47, 13))
        self.object_num.setText("")
        self.object_num.setObjectName("object_num")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(
970, 110, 81, 16))
        self.label_3.setObjectName("label_3")
        
        self.color = QtWidgets.QComboBox(self.centralwidget)
        self.color.setGeometry(QtCore.QRect(
1060, 105, 55, 22))
        self.color.setObjectName("color")
        self.color.addItem("Color")
        self.color.addItem("Red")
        self.color.addItem("Green")
        self.color.addItem("Blue")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(
970, 140, 81, 16))
        self.label_4.setObjectName("label_4")
        self.object_range = QtWidgets.QComboBox(
self.centralwidget)
        self.object_range.setGeometry(QtCore.QRect(
1060, 135, 55, 22))
        self.object_range.setObjectName("object_range")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(
970, 170, 51, 16))
        self.label_5.setObjectName("label_5")
        self.toa_do = QtWidgets.QLabel(self.centralwidget)
        self.toa_do.setGeometry(QtCore.QRect(
1040, 170, 91, 16))
        self.toa_do.setText("")
        self.toa_do.setObjectName("label_6")
        self.dele = QtWidgets.QPushButton(self.centralwidget)
        self.dele.setGeometry(QtCore.QRect(970, 220, 31, 31))
        self.dele.setText("")
        self.dele.setObjectName("dele")
        self.dele.setIcon(QIcon(QPixmap(
"icon/Button-Delete-icon.png")))
        self.dele.setIconSize(QtCore.QSize(31,31))

        self.rec = QtWidgets.QPushButton(self.centralwidget)
        self.rec.setGeometry(QtCore.QRect(1020, 220, 31, 31))
        self.rec.setText("")
        self.rec.setObjectName("rec")
        self.rec.setIcon(QIcon(QPixmap(
"icon/Button-Reload-icon.png")))
        self.rec.setIconSize(QtCore.QSize(31,31))

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(
970, 280, 51, 16))
        self.label_7.setObjectName("label_7")
        self.sp1 = QtWidgets.QSpinBox(self.centralwidget)
        self.sp1.setGeometry(QtCore.QRect(1040, 275, 41, 21))
        self.sp1.setMaximum(2000)
        self.sp1.setMinimum(-2000)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.sp1.setFont(font)
        self.sp1.setObjectName("sp1")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(
970, 315, 51, 16))
        self.label_8.setObjectName("label_8")
        self.sp2 = QtWidgets.QSpinBox(self.centralwidget)
        self.sp2.setGeometry(QtCore.QRect(1040, 310, 41, 21))
        self.sp2.setMaximum(2000)
        self.sp2.setMinimum(-2000)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.sp2.setFont(font)
        self.sp2.setObjectName("sp2")
        self.groupBox_2.raise_()
        self.groupBox.raise_()
        self.result.raise_()
        self.image.raise_()
        self.label_2.raise_()
        self.object_num.raise_()
        self.label_3.raise_()
        self.color.raise_()
        self.label_4.raise_()
        self.object_range.raise_()
        self.label_5.raise_()
        self.toa_do.raise_()
        self.dele.raise_()
        self.rec.raise_()
        self.label_7.raise_()
        self.sp1.raise_()
        self.label_8.raise_()
        self.sp2.raise_()

        self.groupBox_3 = QtWidgets.QGroupBox(
self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(
950, 80, 181, 111))
        self.groupBox_3.setObjectName("groupBox_3")
        self.groupBox_3.lower()
        self.groupBox_4 = QtWidgets.QGroupBox(
self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(
950, 200, 181, 141))
        self.groupBox_4.setObjectName("groupBox_4")
        self.groupBox_4.lower()
        self.groupBox_3.autoFillBackground()
        self.groupBox_4.autoFillBackground()

        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(
620, 50, 21, 16))
        self.label_9.setObjectName("label_9")
        self.xy_value = QtWidgets.QLabel(self.centralwidget)
        self.xy_value.setGeometry(QtCore.QRect(
650, 50, 71, 16))
        self.xy_value.setText("")
        self.xy_value.setObjectName("xy_value")
        self.RGB_value = QtWidgets.QLabel(self.centralwidget)
        self.RGB_value.setGeometry(QtCore.QRect(
790, 50, 101, 16))
        self.RGB_value.setText("")
        self.RGB_value.setObjectName("RGB_value")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(
750, 50, 31, 16))
        self.label_12.setObjectName("label_12")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(
0, 0, 1137, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuOpen.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())

        self.dele.clicked.connect(self.DeleteOb)
        self.result.clicked.connect(self.ShowResult)
        self.image.clicked.connect(self.GetImg)
        self.rec.clicked.connect(self.MoveOb)
        self.display_image.mousePressEvent = self.getPos
        # self.display_image.alignment(QtCore.Qt.AlignTop())
        

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", 
"MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", 
"image"))
        self.groupBox_2.setTitle(_translate("MainWindow", 
"Button"))
        self.label_2.setText(_translate("MainWindow", 
"Object Number :"))
        self.label_3.setText(_translate("MainWindow", 
"Choose Color :"))
        self.label_4.setText(_translate("MainWindow", 
"Choose Object :"))
        self.label_5.setText(_translate("MainWindow", 
"Location: "))
        self.label_7.setText(_translate("MainWindow", 
"Move X:"))
        self.label_8.setText(_translate("MainWindow", 
"Move Y:"))
        self.menuFile.setTitle(_translate("MainWindow", 
"File"))
        self.menuOpen.setTitle(_translate("MainWindow", 
"Open"))
        self.menuTools.setTitle(_translate("MainWindow", 
"Tools"))
        self.color.setItemText(0, _translate("MainWindow", 
"Color"))
        self.color.setItemText(1, _translate("MainWindow", 
"Red"))
        self.color.setItemText(2, _translate("MainWindow", 
"Green"))
        self.color.setItemText(3, _translate("MainWindow", 
"Blue"))
        self.groupBox_3.setTitle(_translate("MainWindow", 
"SELECTED OBJECT"))
        self.groupBox_4.setTitle(_translate("MainWindow", 
"MOVE OBJECT"))
        self.label_9.setText(_translate("MainWindow", 
"x,y:"))
        self.label_12.setText(_translate("MainWindow", 
"RGB:"))

    def getPos(self , event):
        x = event.pos().x()
        y = event.pos().y()
        r,g,b = operate.Operate.GetRGB(self,self.img,y,x)
        self.xy_value.setText(str(x)+' , '+str(y))
        self.RGB_value.setText(str(r)+' , '+str(g)+' , 
'+str(b))
        print (str(x),str(y))

    def test(self):
        print('test OK!')

    def ShowResult(self):
        # r1,g1,b1,r2,g2,b2 ,3,180,0,0,255,100,100
        if self.color.currentText()== 'Red':
            self.r1=180
            self.g1=0
            self.b1=0
            self.r2=255
            self.g2=100
            self.b2=  100 
        
        elif  self.color.currentText()=='Green':
            self.r1=0
            self.g1=120
            self.b1=20
            self.r2=100
            self.g2=220
            self.b2=  120
        else: 
            self.r1=0
            self.g1=20
            self.b1=150
            self.r2=120
            self.g2=120
            self.b2=  250


        img= self.img.copy()
        img,length=operate.Operate.GetLengthImgae(self,img,
self.r1,self.g1,self.b1,self.r2,self.g2,self.b2)
        print('img,length',length)
        self.object_num.setText(str(length))
        self.object_range.clear()

        for i in range(length):
            self.object_range.addItem(str(i+1))

        self.ShowImage(img)

    def DeleteOb(self,ShowResult):
        num= self.object_range.currentText()
        num=int(num)-1
        image= self.img.copy()
        image,x,y= operate.Operate.hello(self,image,num,
self.r1,self.g1,self.b1,self.r2,self.g2,self.b2)
        self.toa_do.setText(str(x)+": "+str(y))
        self.ShowImage(image)

    def GetImg(self):        
        print('Get image')
        appli= App()
        self.img=cv.imread(appli.ImgPath)  
        print(self.img)
        self.ShowImage(self.img)

    def MoveOb(self):

        num= self.object_range.currentText()
        num=int(num)-1
        img= self.img.copy()
        pixel_num1= self.sp2.value()
        pixel_num2= self.sp1.value()
        img_del=operate.Operate.MovOb(self,img,num,
pixel_num1,pixel_num2,self.r1,self.g1,self.b1,self.r2,self.
g2,self.b2)
        self.ShowImage(img_del)
        
    def ShowImage(self,img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        img = QImage(img.data, width, height, bytesPerLine, QImage.Format_BGR888)
        self.display_image.setPixmap(QtGui.QPixmap(img))

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, 
self.height)       
        self.ImgPath= self.openFileNameDialog()
   
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,
"QFileDialog.getOpenFileName()", "","All Files (*);;Image Files (*.jpg *.png)", options=options)
        return fileName