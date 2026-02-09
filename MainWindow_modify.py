# -*- coding: utf-8 -*-

#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from lib.modified_item import ClickableImage

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1000)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.left_img = ClickableImage(self.centralwidget)
        #self.left_img = QtWidgets.QLabel(self.centralwidget)
        self.left_img.setGeometry(QtCore.QRect(20, 10, 640, 480))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.left_img.setFont(font)
        self.left_img.setObjectName("left_img")
        self.pushButton_open_cam = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_open_cam.setGeometry(QtCore.QRect(660, 500, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_open_cam.setFont(font)
        self.pushButton_open_cam.setObjectName("pushButton_open_cam")
        #self.right_img = QtWidgets.QLabel(self.centralwidget)
        self.right_img = ClickableImage(self.centralwidget)
        self.right_img.setGeometry(QtCore.QRect(670, 10, 640, 480))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.right_img.setFont(font)
        self.right_img.setObjectName("right_img")
        self.depth_img = QtWidgets.QLabel(self.centralwidget)
        self.depth_img.setGeometry(QtCore.QRect(20, 490, 640, 480))
        self.depth_img.setObjectName("depth_img")
        self.label_left = QtWidgets.QLabel(self.centralwidget)
        self.label_left.setGeometry(QtCore.QRect(20, 450, 200, 40))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_left.setFont(font)
        self.label_left.setTextFormat(QtCore.Qt.PlainText)
        self.label_left.setScaledContents(True)
        self.label_left.setIndent(-1)
        self.label_left.setObjectName("label_left")
        self.label_right = QtWidgets.QLabel(self.centralwidget)
        self.label_right.setGeometry(QtCore.QRect(1210, 450, 200, 40))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_right.setFont(font)
        self.label_right.setTextFormat(QtCore.Qt.PlainText)
        self.label_right.setScaledContents(True)
        self.label_right.setIndent(-1)
        self.label_right.setObjectName("label_right")
        self.label_depth = QtWidgets.QLabel(self.centralwidget)
        self.label_depth.setGeometry(QtCore.QRect(20, 920, 110, 40))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_depth.setFont(font)
        self.label_depth.setTextFormat(QtCore.Qt.PlainText)
        self.label_depth.setScaledContents(True)
        self.label_depth.setIndent(-1)
        self.label_depth.setObjectName("label_depth")
        self.pushButton_get_depth = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_get_depth.setGeometry(QtCore.QRect(820, 500, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_get_depth.setFont(font)
        self.pushButton_get_depth.setObjectName("pushButton_get_depth")
        self.pushButton_show_obj = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_show_obj.setGeometry(QtCore.QRect(980, 560, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_show_obj.setFont(font)
        self.pushButton_show_obj.setObjectName("pushButton_show_obj")
        self.pushButton_record_ref = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_record_ref.setGeometry(QtCore.QRect(980, 500, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_record_ref.setFont(font)
        self.pushButton_record_ref.setObjectName("pushButton_record_ref")
        self.pushButton_calc_error = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_calc_error.setGeometry(QtCore.QRect(1140, 500, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_calc_error.setFont(font)
        self.pushButton_calc_error.setObjectName("pushButton_calc_error")
        self.pushButton_calib = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_calib.setGeometry(QtCore.QRect(1140, 560, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_calib.setFont(font)
        self.pushButton_calib.setObjectName("pushButton_calib")
        self.label_class = QtWidgets.QLabel(self.centralwidget)
        self.label_class.setGeometry(QtCore.QRect(660, 660, 301, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_class.setFont(font)
        self.label_class.setObjectName("label_class")
        self.label_obj_cur_pose = QtWidgets.QLabel(self.centralwidget)
        self.label_obj_cur_pose.setGeometry(QtCore.QRect(660, 700, 681, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_obj_cur_pose.setFont(font)
        self.label_obj_cur_pose.setObjectName("label_obj_cur_pose")
        self.label_obj_ref_pose = QtWidgets.QLabel(self.centralwidget)
        self.label_obj_ref_pose.setGeometry(QtCore.QRect(660, 730, 681, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_obj_ref_pose.setFont(font)
        self.label_obj_ref_pose.setObjectName("label_obj_ref_pose")
        self.label_motor_distance = QtWidgets.QLabel(self.centralwidget)
        self.label_motor_distance.setGeometry(QtCore.QRect(660, 860, 681, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_motor_distance.setFont(font)
        self.label_motor_distance.setObjectName("label_motor_distance")
        self.label_measure_errors = QtWidgets.QLabel(self.centralwidget)
        self.label_measure_errors.setGeometry(QtCore.QRect(660, 910, 681, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_measure_errors.setFont(font)
        self.label_measure_errors.setObjectName("label_measure_errors")
        self.pushButton_enable_detection = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_enable_detection.setGeometry(QtCore.QRect(660, 560, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_enable_detection.setFont(font)
        self.pushButton_enable_detection.setObjectName("pushButton_enable_detection")
        self.pushButton_stop_detection = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_stop_detection.setGeometry(QtCore.QRect(820, 560, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_stop_detection.setFont(font)
        self.pushButton_stop_detection.setObjectName("pushButton_stop_detection")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(1370, 30, 501, 501))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton_forward = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_forward.setGeometry(QtCore.QRect(205, 55, 90, 65))
        self.pushButton_forward.setStyleSheet("QPushButton {\n"
"    border-image: url(icons/up1.png);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    border-image: url(icons/up2.png);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    border-image: url(icons/up3.png);\n"
"}")
        self.pushButton_forward.setText("")
        self.pushButton_forward.setObjectName("pushButton_forward")
        self.pushButton_down = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_down.setGeometry(QtCore.QRect(210, 370, 90, 65))
        self.pushButton_down.setStyleSheet("QPushButton {\n"
"    border-image: url(icons/down1.png);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    border-image: url(icons/down2.png);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    border-image: url(icons/down3.png);\n"
"}")
        self.pushButton_down.setText("")
        self.pushButton_down.setObjectName("pushButton_down")
        self.pushButton_backward = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_backward.setGeometry(QtCore.QRect(205, 145, 90, 65))
        self.pushButton_backward.setStyleSheet("QPushButton {\n"
"    border-image: url(icons/down1.png);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    border-image: url(icons/down2.png);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    border-image: url(icons/down3.png);\n"
"}")
        self.pushButton_backward.setText("")
        self.pushButton_backward.setObjectName("pushButton_backward")
        self.pushButton_left = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_left.setGeometry(QtCore.QRect(135, 305, 80, 70))
        self.pushButton_left.setStyleSheet("QPushButton {\n"
"    border-image: url(icons/left1.png);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    border-image: url(icons/left2.png);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    border-image: url(icons/left3.png);\n"
"}")
        self.pushButton_left.setText("")
        self.pushButton_left.setObjectName("pushButton_left")
        self.pushButton_right = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_right.setGeometry(QtCore.QRect(294, 305, 80, 70))
        self.pushButton_right.setStyleSheet("QPushButton {\n"
"    border-image: url(icons/right1.png);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    border-image: url(icons/right2.png);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    border-image: url(icons/right3.png);\n"
"}")
        self.pushButton_right.setText("")
        self.pushButton_right.setObjectName("pushButton_right")
        self.pushButton_up = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_up.setGeometry(QtCore.QRect(210, 240, 90, 65))
        self.pushButton_up.setStyleSheet("QPushButton {\n"
"    border-image: url(icons/up1.png);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    border-image: url(icons/up2.png);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    border-image: url(icons/up3.png);\n"
"}")
        self.pushButton_up.setText("")
        self.pushButton_up.setObjectName("pushButton_up")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(1370, 600, 501, 381))
        self.textBrowser.setObjectName("textBrowser")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1370, 566, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton_VCap_left = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_VCap_left.setGeometry(QtCore.QRect(660, 620, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_VCap_left.setFont(font)
        self.pushButton_VCap_left.setObjectName("pushButton_VCap_left")
        self.pushButton_VCap_right = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_VCap_right.setGeometry(QtCore.QRect(820, 620, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_VCap_right.setFont(font)
        self.pushButton_VCap_right.setObjectName("pushButton_VCap_right")
        self.pushButton_VCap_depth = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_VCap_depth.setGeometry(QtCore.QRect(980, 620, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_VCap_depth.setFont(font)
        self.pushButton_VCap_depth.setObjectName("pushButton_VCap_depth")
        self.label_motor_cur_pos = QtWidgets.QLabel(self.centralwidget)
        self.label_motor_cur_pos.setGeometry(QtCore.QRect(660, 800, 681, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_motor_cur_pos.setFont(font)
        self.label_motor_cur_pos.setObjectName("label_motor_cur_pos")
        self.label_obj_move_pos = QtWidgets.QLabel(self.centralwidget)
        self.label_obj_move_pos.setGeometry(QtCore.QRect(660, 760, 681, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_obj_move_pos.setFont(font)
        self.label_obj_move_pos.setObjectName("label_obj_move_pos")
        self.label_motor_ref_pos = QtWidgets.QLabel(self.centralwidget)
        self.label_motor_ref_pos.setGeometry(QtCore.QRect(660, 830, 681, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_motor_ref_pos.setFont(font)
        self.label_motor_ref_pos.setObjectName("label_motor_ref_pos")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.left_img.setText(_translate("MainWindow", "wait for left image!"))
        self.pushButton_open_cam.setText(_translate("MainWindow", "打开摄像头"))
        self.right_img.setText(_translate("MainWindow", "wait for right image!"))
        self.depth_img.setText(_translate("MainWindow", "TextLabel"))
        self.label_left.setText(_translate("MainWindow", "左视图"))
        self.label_right.setText(_translate("MainWindow", "右视图"))
        self.label_depth.setText(_translate("MainWindow", "深度图"))
        self.pushButton_get_depth.setText(_translate("MainWindow", "获取深度图"))
        self.pushButton_show_obj.setText(_translate("MainWindow", "显示结果图像"))
        self.pushButton_record_ref.setText(_translate("MainWindow", "记录当参考位置"))
        self.pushButton_calc_error.setText(_translate("MainWindow", "计算测量误差"))
        self.pushButton_calib.setText(_translate("MainWindow", "获取标定图像"))
        self.label_class.setText(_translate("MainWindow", "目标类别         ："))
        self.label_obj_cur_pose.setText(_translate("MainWindow", "当前目标位置："))
        self.label_obj_ref_pose.setText(_translate("MainWindow", "参考目标位置："))
        self.label_motor_distance.setText(_translate("MainWindow", "电机移动距离："))
        self.label_measure_errors.setText(_translate("MainWindow", "相对测量误差："))
        self.pushButton_enable_detection.setText(_translate("MainWindow", "开始目标检测"))
        self.pushButton_stop_detection.setText(_translate("MainWindow", "停止目标检测"))
        self.groupBox.setTitle(_translate("MainWindow", "控制区"))
        self.label.setText(_translate("MainWindow", "状态日志"))
        self.pushButton_VCap_left.setText(_translate("MainWindow", "开始录制-左"))
        self.pushButton_VCap_right.setText(_translate("MainWindow", "开始录制-右"))
        self.pushButton_VCap_depth.setText(_translate("MainWindow", "开始录制-深度"))
        self.label_motor_cur_pos.setText(_translate("MainWindow", "当前电机位置："))
        self.label_obj_move_pos.setText(_translate("MainWindow", "目标移动距离："))
        self.label_motor_ref_pos.setText(_translate("MainWindow", "参考电机位置："))
