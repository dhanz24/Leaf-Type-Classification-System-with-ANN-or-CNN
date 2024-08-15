from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    Flatten,
    Dense,
    Conv2D,
    MaxPool2D,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

imagePaths = "Dataset\\"
label_list = ["Nangka", "Daun Sirih"]
data = []
labels = []

for label in label_list:
    for imagePath in glob.glob(imagePaths + label + "\\*.jpg"):
        # print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32))
        data.append(image)
        labels.append(label)
np.array(data).shape

# ubah type data dari list menjadi array
# ubah nilai dari tiap pixel menjadi range [0..1]
for i in range(len(data)):
    for j in range(len(data[i])):
        for k in range(len(data[i][j])):
            data[i][j][k] = data[i][j][k] / 255.0

labels = np.array(labels)
print(labels)

# ubah nilai dari labels menjadi binary
# ubah nilai dari labels menjadi binary
lb = LabelEncoder()
labels = lb.fit_transform(labels)

test_indices = [1, 3, 5, 7, 9]  # contoh indeks data test
train_data = []
train_labels = []
test_data = []
test_labels = []

for i in range(len(data)):
    if i in test_indices:
        test_data.append(data[i])
        test_labels.append(labels[i])
    else:
        train_data.append(data[i])
        train_labels.append(labels[i])

x_train = np.array(train_data) / 255.0
x_test = np.array(test_data) / 255.0
y_train = np.array(train_labels)
y_test = np.array(test_labels)

print("Ukuran data train =", x_train.shape)
print("Ukuran data test =", x_test.shape)

model = Sequential()
model.add(InputLayer(input_shape=(32, 32, 3)))
model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2, padding="same"))
model.add(Conv2D(filters=50, kernel_size=2, strides=1, padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2, padding="same"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(label_list), activation="softmax"))
model.summary()

# tentukan hyperparameter
lr = 0.001
max_epochs = 100
opt_funct = Adam(learning_rate=lr)

# compile arsitektur yang telah dibuat
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt_funct, metrics=["accuracy"])

# Train model
H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=max_epochs, batch_size=32)

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
predictions = model.predict(x_test)
target = predictions.argmax(axis=1)
print(classification_report(y_test, target, target_names=label_list, zero_division=1))

# ...

# ...

# uji model menggunakan gambar lain
queryPath = "D:\Presetasi JST\Data Uji\002.jpg"  # Sesuaikan dengan path gambar yang ingin diuji
 = cv2.imread(queryPath)

# Resize the image
query = cv2.resize(fotoasli, (32, 32))
query = np.array([query], dtype="float") / 255.0

# Tambahkan kembali konversi warna dari RGB ke BGR untuk menampilkan dengan cv2.imshow
query_display = cv2.cvtColor(query.astype(np.uint8), cv2.COLOR_RGB2BGR)

# Predict using the model
q_pred = model.predict(query)
i = q_pred.argmax(axis=1)[0]
label = lb.classes_[i]
text = "{}: {:.2f}%".format(label, q_pred[0][i] * 100)

# Tambahkan label prediksi ke gambar
cv2.putText(query_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# Menampilkan ukuran jendela OpenCV
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", 600, 600)

# Menampilkan output gambar
cv2.imshow("Output", query_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
