# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'input_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_inputDialog(object):
    def setupUi(self, inputDialog):
        inputDialog.setObjectName("inputDialog")
        inputDialog.resize(906, 621)
        self.input_text = QtWidgets.QTextEdit(inputDialog)
        self.input_text.setGeometry(QtCore.QRect(250, 50, 461, 171))
        self.input_text.setStyleSheet("font: 16pt \"Consolas\";")
        self.input_text.setObjectName("input_text")
        self.output_text = QtWidgets.QTextBrowser(inputDialog)
        self.output_text.setGeometry(QtCore.QRect(250, 340, 471, 211))
        self.output_text.setStyleSheet("font: 16pt \"Consolas\";")
        self.output_text.setObjectName("output_text")
        self.label = QtWidgets.QLabel(inputDialog)
        self.label.setGeometry(QtCore.QRect(50, 50, 131, 81))
        self.label.setStyleSheet("font: 16pt \"楷体\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(inputDialog)
        self.label_2.setGeometry(QtCore.QRect(50, 350, 131, 81))
        self.label_2.setStyleSheet("font: 16pt \"楷体\";")
        self.label_2.setObjectName("label_2")
        self.input_label = QtWidgets.QLabel(inputDialog)
        self.input_label.setGeometry(QtCore.QRect(80, 120, 131, 81))
        self.input_label.setStyleSheet("font: 16pt \"楷体\";")
        self.input_label.setObjectName("input_label")
        self.output_label = QtWidgets.QLabel(inputDialog)
        self.output_label.setGeometry(QtCore.QRect(80, 430, 131, 81))
        self.output_label.setStyleSheet("font: 16pt \"楷体\";")
        self.output_label.setObjectName("output_label")
        self.audio_input = QtWidgets.QPushButton(inputDialog)
        self.audio_input.setGeometry(QtCore.QRect(740, 100, 141, 81))
        self.audio_input.setStyleSheet("font: 16pt \"楷体\";")
        self.audio_input.setObjectName("audio_input")
        self.translate_button = QtWidgets.QPushButton(inputDialog)
        self.translate_button.setGeometry(QtCore.QRect(330, 260, 301, 41))
        self.translate_button.setStyleSheet("font: 12pt \"楷体\";")
        self.translate_button.setObjectName("translate_button")
        self.read_result = QtWidgets.QPushButton(inputDialog)
        self.read_result.setGeometry(QtCore.QRect(740, 410, 141, 81))
        self.read_result.setStyleSheet("font: 16pt \"楷体\";")
        self.read_result.setObjectName("read_result")

        self.retranslateUi(inputDialog)
        QtCore.QMetaObject.connectSlotsByName(inputDialog)

    def retranslateUi(self, inputDialog):
        _translate = QtCore.QCoreApplication.translate
        inputDialog.setWindowTitle(_translate("inputDialog", "Dialog"))
        self.input_text.setHtml(_translate("inputDialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Consolas\'; font-size:16pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label.setText(_translate("inputDialog", "检测语言"))
        self.label_2.setText(_translate("inputDialog", "目标语言"))
        self.input_label.setText(_translate("inputDialog", "中文"))
        self.output_label.setText(_translate("inputDialog", "英文"))
        self.audio_input.setText(_translate("inputDialog", "语音输入"))
        self.translate_button.setText(_translate("inputDialog", "翻        译"))
        self.read_result.setText(_translate("inputDialog", "朗读结果"))
