# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'file_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_fileDialog(object):
    def setupUi(self, fileDialog):
        fileDialog.setObjectName("fileDialog")
        fileDialog.resize(826, 771)
        self.scanButton = QtWidgets.QPushButton(fileDialog)
        self.scanButton.setGeometry(QtCore.QRect(60, 300, 191, 61))
        self.scanButton.setStyleSheet("font: 16pt \"楷体\";")
        self.scanButton.setObjectName("scanButton")
        self.picLabel = QtWidgets.QLabel(fileDialog)
        self.picLabel.setGeometry(QtCore.QRect(290, 40, 500, 400))
        self.picLabel.setText("")
        self.picLabel.setWordWrap(True)
        self.picLabel.setObjectName("picLabel")
        self.loadButton = QtWidgets.QPushButton(fileDialog)
        self.loadButton.setGeometry(QtCore.QRect(60, 120, 191, 61))
        self.loadButton.setStyleSheet("font: 16pt \"楷体\";")
        self.loadButton.setObjectName("loadButton")
        self.out_text = QtWidgets.QTextBrowser(fileDialog)
        self.out_text.setGeometry(QtCore.QRect(50, 460, 731, 291))
        self.out_text.setStyleSheet("font: 14pt \"楷体\";")
        self.out_text.setObjectName("out_text")

        self.retranslateUi(fileDialog)
        QtCore.QMetaObject.connectSlotsByName(fileDialog)

    def retranslateUi(self, fileDialog):
        _translate = QtCore.QCoreApplication.translate
        fileDialog.setWindowTitle(_translate("fileDialog", "Dialog"))
        self.scanButton.setText(_translate("fileDialog", "扫描文档翻译"))
        self.loadButton.setText(_translate("fileDialog", "加载扫描文件"))
