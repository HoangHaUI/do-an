import tensorflow as tf
import numpy as np
from src.training.load_model import load_model
from src.common import get_crop_image
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
from ultralytics import YOLO
from collections import deque


# Load Yolo model
yolo_model = YOLO("Yolo/yolov8s.pt")

checkpoint_dir = "runs/training/"
# Load model 
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
# Create a new model instance
model = load_model(4)

# Load the previously saved weights
model.load_weights(latest)

def predict(path):
    
    # Yolo predict
    rets = yolo_model(path,  save_txt=True, save_conf=True)
    boxes = []
    boxes = [np.asarray(ret.boxes.xyxy) for ret in rets]

    
    classes = []
    classes = [np.asarray(ret.boxes.cls) for ret in rets]
    
    nap_coords, nhan_coords, bottle_coord = get_crop_image(classes, boxes)
    
    class_names = ['nap', 'nhan', 'thieu_nap', 'thieu_nhan']
    # Evaluate
    img = tf.keras.utils.load_img(path)
    img = np.asarray(img)
    
    img_naps = []
    for arr in nap_coords:
        img_crop = np.expand_dims(cv.resize(np.array(img[arr[1]:arr[3],arr[0]:arr[2],:]), (244,244), interpolation=cv.INTER_CUBIC), axis = 0)
        if len(img_naps) == 0:
            img_naps = img_crop
            continue
        img_naps = np.append(img_naps,img_crop,axis=0)
    
    img_nhans = []
    for arr in nhan_coords:
        img_crop = np.expand_dims(cv.resize(np.array(img[arr[1]:arr[3],arr[0]:arr[2],:]), (244,244), interpolation=cv.INTER_CUBIC), axis = 0)
        if len(img_nhans) == 0:
            img_nhans = img_crop
            continue
        img_nhans = np.append(img_nhans,img_crop,axis=0)
    
    
    print(img_naps.shape)
    
    predictions = None
    predictions_nap = None
    predictions_nhan = None
    with tf.device('/cpu:0'):
        predictions_nap = model.predict(np.asarray(img_naps))
        predictions_nhan = model.predict(np.asarray(img_nhans))
        
    # LOAD IMAGE TO SHOW
    img = cv.imread(path)
    
    # Write coordination
    for i in range(len(bottle_coord)):
        points= bottle_coord[i]
        img = cv.rectangle(img, (points[0], points[1]),(points[2], points[3]), (0,255,255), 2)
       
    
    # Get final result:
    final_result_nap = "CHUAN"
    indexes_fail_nap = []
    for i, ret in enumerate(predictions_nap):
        if class_names[np.argmax(ret)] == "thieu_nap":
            final_result_nap = "thieu_nap"
            points= nap_coords[i]
            img = cv.rectangle(img, (points[0], points[1]),(points[2], points[3]), (0,0,255), 15)
    
    final_result_nhan = "CHUAN"
    indexes_fail_nhan = []
    for i, ret in enumerate(predictions_nhan):
        if class_names[np.argmax(ret)] == "thieu_nhan":
            final_result_nhan = "thieu_nhan"
            points= nhan_coords[i]
            img = cv.rectangle(img, (points[0], points[1]),(points[2], points[3]), (0,0,255), 15)
            
            
    for i in range(len(bottle_coord)):
        points= bottle_coord[i]   
        cv.putText(img, f"{points[0]}:{points[1]}", (points[0],  points[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    cv.imwrite("ret.png", img)
    if final_result_nap == "CHUAN" and final_result_nhan == "CHUAN":
        return "CHUAN"
    return "THIEU"



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
        if ret == "THIEU":
            self.lbResult.setStyleSheet('background-color: RED')
        self.lbResult.setText(ret)
        self.lbImage.setPixmap(QPixmap("./ret.png"))
        
        
        
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