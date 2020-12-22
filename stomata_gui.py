import sys
import test
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog
from utils import detect

from PyQt5.QtCore  import QStringListModel

class UIAction(test.Ui_MainWindow):
    FIEPATH = ""
    MODELPATH = ""
    StomataNumber = 0
    INTERVAL = 0
    OUTPUT_VIDEO_PATH = "result.avi"
    CVS_PATH = "result.csv"

    LOG_LIST = []
    LOG_StringList = QStringListModel()

    def on_openFile_button(self):
        filePath, ok1 = QFileDialog.getOpenFileName(None, "请选择要添加的文件", "./", "Text Files (*.mp4);;All Files (*)")
        if filePath != "":
            self.FIEPATH = filePath
            self.OpenFilePathLabel.setText(self.FIEPATH)
            self.LOG_LIST.append("Open File:" + self.FIEPATH)
            self.LOG_StringList.setStringList(self.LOG_LIST)

    def on_modelFile_button(self):
        filePath, ok1 = QFileDialog.getOpenFileName(None, "请选择要添加的文件", "./", "Text Files (*.h5);;All Files (*)")
        if filePath != "":
            self.MODELPATH = filePath
            self.modelPathLabel.setText(self.MODELPATH)

            self.LOG_LIST.append("Open Model File:" + self.MODELPATH)
            self.LOG_StringList.setStringList(self.LOG_LIST)

    def on_run_button(self):
        self.StomataNumber = self.comboBoxStomataNumber.currentText()
        self.INTERVAL      = self.intervalLineEdit.text()

        self.LOG_LIST.append("StomataNumber:" + self.StomataNumber  + " Interval:" + self.INTERVAL + "Model Path:" +
                             self.MODELPATH + "Result video:" + self.OUTPUT_VIDEO_PATH + "CVS Path:" + self.CVS_PATH)
        self.LOG_StringList.setStringList(self.LOG_LIST)
        detect(self.FIEPATH, self.OUTPUT_VIDEO_PATH, int(self.INTERVAL), int(self.StomataNumber), self.CVS_PATH, self.MODELPATH, self.output_to_ui)

    def output_to_ui(self, log):
        self.LOG_LIST.append(log)
        self.LOG_StringList.setStringList(self.LOG_LIST)

    def on_clear_button(self):
        self.LOG_LIST.clear()
        self.LOG_StringList.setStringList(self.LOG_LIST)

    def init_ui_action(self):
        self.OpenFileButton.clicked.connect(self.on_openFile_button)
        self.OpenModelButton.clicked.connect(self.on_modelFile_button)
        for i in range(2, 11):
            self.comboBoxStomataNumber.addItem(str(i))

        self.videoNameLineEdit.setText(self.OUTPUT_VIDEO_PATH)
        self.cvslineEdit.setText(self.CVS_PATH)
        self.LOG_StringList.setStringList(self.LOG_LIST)
        self.reslultlistView.setModel(self.LOG_StringList)
        self.runButton.clicked.connect(self.on_run_button)
        self.clearLogButton.clicked.connect(self.on_clear_button)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = UIAction()
    ui.setupUi(MainWindow)
    ui.init_ui_action()
    MainWindow.show()
    sys.exit(app.exec_())