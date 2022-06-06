import os
import threading
import time

from main_ui import Ui_MainWindow
from input_ui import Ui_inputDialog
from audio_ui import Ui_audioDialog
from picture_ui import Ui_picDialog
from file_ui import Ui_fileDialog
from search_ui import Ui_searchDialog

import sys

import PyQt5
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon, QPen
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtGui import QStandardItem

import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import translation_tool


class search_win(QtWidgets.QDialog, Ui_searchDialog):
    def __init__(self):
        super(search_win, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("图片目标翻译")

        self.fig = None
        self.gridlayout = None

        self.searchButton.clicked.connect(self.search_pic)

    def search_pic(self):
        try:
            pic_path, zh, en = translation_tool.search_pic(self.inout.text())
            if pic_path == '':
                QtWidgets.QMessageBox.warning(self, "提示", "很抱歉，未找到相关图片")
                return
            self.show_pic(pic_path, zh ,en)
        except Exception as e:
            print(e)
            QtWidgets.QMessageBox.critical(self, "错误", "搜索失败")
            return

    def show_pic(self, pic_path, zh, en):

        if pic_path == '':
            return
        pixmap = QtGui.QPixmap(pic_path)
        self.picLabel.setPixmap(pixmap)
        self.label.setText(f'中文 : {zh}    英文 : {en}')
        return


class file_win(QtWidgets.QDialog, Ui_fileDialog):
    def __init__(self):
        super(file_win, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('扫描文档翻译')

        self.file_path = ''

        self.loadButton.clicked.connect(self.load_file)
        self.scanButton.clicked.connect(self.scan_file)

    def load_file(self):
        file_path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, "选择文件", os.getcwd(), "JPG File(*.jpg);;PNG File (*.png);;JPEG File(*.jpeg)")
        if file_path == '':
            return
        self.file_path = file_path
        self.show_pic()


    def scan_file(self):
        if self.file_path == '':
            return
        try:
            res = translation_tool.scan_file(self.file_path)
            res = translation_tool.translate_cn_to_en(res)
            self.out_text.setText(res)
        except:
            QtWidgets.QMessageBox.critical(self, "错误", "扫描失败")
            return
        return

    def show_pic(self):
        if self.file_path == '':
            return
        image = mpimg.imread(self.file_path)
        self.fig = MyFigure(width=500, height=400)
        # axes = self.left_figure.fig.add_subplot(1, 1, 1)
        plt.imshow(image)
        plt.axis('off')
        self.gridlayout = PyQt5.QtWidgets.QGridLayout(self.picLabel)
        self.gridlayout.addWidget(self.fig, 0, 1)
        return


class pic_win(QtWidgets.QDialog, Ui_picDialog):
    def __init__(self):
        super(pic_win, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("智能识图翻译")

        self.pic_path = ''

        model = MyQStandardItemModelModel(2, 3)

        model.setVerticalHeaderLabels(['中文', '英文'])
        self.resultView.setModel(model)
        self.resultView.horizontalHeader().setStretchLastSection(True)
        self.resultView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.resultView.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.resultView.setEditTriggers(QtWidgets.QTableView.NoEditTriggers)

        self.loadButton.clicked.connect(self.load_pic)
        self.identifyButton.clicked.connect(self.identify_pic)

    def load_pic(self):
        file_path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, "选择文件", os.getcwd(), "JPG File(*.jpg);;PNG File (*.png);;JPEG File(*.jpeg)")
        if file_path == '':
            return
        self.pic_path = file_path
        self.show_pic()

    def show_pic(self):
        if self.pic_path == '':
            return
        image = mpimg.imread(self.pic_path)
        self.fig = MyFigure(width=500, height=400)
        # axes = self.left_figure.fig.add_subplot(1, 1, 1)
        plt.imshow(image)
        plt.axis('off')
        self.gridlayout = PyQt5.QtWidgets.QGridLayout(self.picLabel)
        self.gridlayout.addWidget(self.fig, 0, 1)

    def identify_pic(self):
        if self.pic_path == '':
            return
        try:
            res = translation_tool.identify_picture(self.pic_path)
            for i in range(3):
                model = self.resultView.model()
                model.setItem(0, i, QStandardItem(res[i]['zh']))
                model.setItem(1, i, QStandardItem(res[i]['en']))

        except:
            QtWidgets.QMessageBox.critical(self, "错误", "识别失败")
            return

class input_win(QtWidgets.QDialog, Ui_inputDialog):

    def __init__(self):
        super(input_win, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("输入翻译")

        self.translate_button.clicked.connect(self.translate)
        self.audio_input.clicked.connect(self.audio_make)
        self.read_result.clicked.connect(self.generate_read)

    def translate(self):
        print('start translate..')
        try:
            res = translation_tool.translate_cn_to_en(self.input_text.toPlainText())
            self.output_text.setText(res)
        except:
            QtWidgets.QMessageBox.critical(self, "错误", "翻译失败")

    def generate_read(self):
        translation_tool.generate_sound(self.output_text.toPlainText())

    def audio_make(self):
        win = audio_win()
        win.text_signal.connect(self.text_slot)
        win.exec()

    def text_slot(self, param):
        self.input_text.setText(param)


class audio_win(QtWidgets.QDialog, Ui_audioDialog):
    text_signal = pyqtSignal(str)

    def __init__(self):
        super(audio_win, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("语音输入")

        self.label.setText('尚未录制')

        self.startButton.clicked.connect(self.start)
        self.stopButton.clicked.connect(self.stop)

    def start(self):
        translation_tool.running = True
        audio_making = threading.Thread(target=translation_tool.make_audio)
        audio_making.start()
        time.sleep(1)
        self.label.setText('正在录制')
        self.out_text.setText('')

    def stop(self):
        translation_tool.running = False

        self.label.setText('正在解析')
        QtWidgets.QApplication.processEvents()
        time.sleep(3)
        if not os.path.exists('data/out'):
            QtWidgets.QMessageBox.critical(self, "错误", "录制失败")
            self.label.setText('尚未录制')
            return
        with open('data/out', 'r') as f:
            file = f.read()
        res = translation_tool.get_sound_text(file)
        self.out_text.setText(res)
        os.remove('data/out')
        self.label.setText('完成录制')

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.text_signal.emit(self.out_text.toPlainText())
        self.accept()


class ui_win(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(ui_win, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("易点智能翻译系统")

        self.input_translate.clicked.connect(self.input_trans)
        self.identify_translate.clicked.connect(self.pic_trans)
        self.file_translate.clicked.connect(self.file_trans)
        self.pic_translate.clicked.connect(self.search_trans)

    def search_trans(self):
        win = search_win()
        win.exec()

    def file_trans(self):
        win = file_win()
        win.exec()

    def pic_trans(self):
        win = pic_win()
        win.exec()

    def input_trans(self):
        win = input_win()
        win.exec()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        translation_tool.clear_data()


class MyFigure(FigureCanvas):
    def __init__(self, width, height):
        self.fig = plt.figure(figsize=(width, height))
        super(MyFigure, self).__init__(self.fig)


class MyQStandardItemModelModel(QStandardItemModel):
    """
    重写QStandardItemModel的data函数，使QTableView全部item居中
    """

    def data(self, index, role=None):
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        return QStandardItemModel.data(self, index, role)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ui_win()
    window.show()
    sys.exit(app.exec_())
