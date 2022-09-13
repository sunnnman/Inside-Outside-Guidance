import os
# envpath = '/home/zhupengqi/anaconda3/envs/seg/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap

# import t
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

# def loadfile(ui):
#     label = QLabel(ui.frame)
#     print("load--file")
#     fname, _ = QFileDialog.getOpenFileNames(ui.centralwidget,'选择图片', 'c:\\', 'image files(*.jpg *.gif *.png)')
#     label.setPixmap(QPixmap('./2007_001185.jpg'))


if __name__ == '__main__':

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    # ui = t.Ui_MainWindow()
    # ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
    os.remove('./test/')

