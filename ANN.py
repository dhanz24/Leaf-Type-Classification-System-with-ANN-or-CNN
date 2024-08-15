from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PIL import ImageQt
from matplotlib import pyplot as plt
import glob

imagePaths = "Dataset\\"
label_list = [
    "Daun Nangka",
    "DAUN BELIMBING WULUH",
    "Daun Sirih",
    "Nangka",
]
data = []
labels = []
for label in label_list:
    for imagePath in glob.glob(imagePaths + label + "\\*.jpg"):
        # print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)
        labels.append(label)
np.array(data).shape
# ubah type data dari list menjadi array
# ubah nilai dari tiap pixel menjadi range [0..1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print(labels)
# ubah nilai dari labels menjadi binary
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(labels)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
print("Ukuran data train =", x_train.shape)
print("Ukuran data test =", x_test.shape)
# buat ANN dengan arsitektur input layer (3072) - hidden layer (512) - hidden layer (1024) - output layer (3)
model = Sequential()
model.add(Dense(512, input_shape=(3072,), activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(len(label_list), activation="softmax"))
model.summary()
# tentukan hyperparameter
lr = 0.01
max_epochs = 100
opt_funct = SGD(learning_rate=lr)
# compile arsitektur yang telah dibuat
model.compile(
    loss="categorical_crossentropy", optimizer=opt_funct, metrics=["accuracy"]
)
H = model.fit(
    x_train, y_train, validation_data=(x_test, y_test), epochs=max_epochs, batch_size=32
)


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi("citra.ui", self)
        self.Image = None
        self.button_loadcitra.clicked.connect(self.open)
        self.pushButton.clicked.connect(self.histogram)
        self.button_prosescitra.clicked.connect(self.proses)

    def open(self):
        imagePath, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath)
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(pixmap)
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label.setScaledContents(True)

    def proses(self):
        # uji model menggunakan image lain
        query = self.Image
        output = query.copy()
        query = cv2.resize(query, (32, 32)).flatten()
        q = []
        q.append(query)
        q = np.array(q, dtype="float") / 255.0

        q_pred = model.predict(q)
        i = q_pred.argmax(axis=1)[0]
        label = lb.classes_[i]
        text = "{}: {:.2f}%".format(label, q_pred[0][i] * 100)
        cv2.putText(
            output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        self.Image = output
        self.displayImage(2)

    def histogram(self):
        N = np.arange(0, max_epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label="train_loss")
        plt.plot(N, H.history["val_loss"], label="val_loss")
        plt.plot(N, H.history["accuracy"], label="train_acc")
        plt.plot(N, H.history["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch #")
        plt.legend()
        plt.show()

        # menghitung nilai akurasi model terhadap data test
        predictions = model.predict(x_test, batch_size=32)
        target = (predictions > 0.5).astype(np.int)
        print(classification_report(y_test, target, target_names=label_list))

    def displayImage(self, window):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(
            self.Image,
            self.Image.shape[1],
            self.Image.shape[0],
            self.Image.strides[0],
            qformat,
        )

        img = img.rgbSwapped()

        if window == 1:
            self.label_Load.setPixmap(QPixmap.fromImage(img))
            self.label_Load.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
            )
            self.label_Load.setScaledContents(True)
        elif window == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)
        print("----Nilai Pixel Citra----\n", self.Image)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle("Project Akhir")
window.show()
sys.exit(app.exec_())
