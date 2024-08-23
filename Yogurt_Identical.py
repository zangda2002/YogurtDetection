import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
traindata_datagenerator = ImageDataGenerator(rescale=1./255,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True)
traindata_set = traindata_datagenerator.flow_from_directory(r'C:\Users\zangd\PycharmProjects\Nhom01_NhanDienSua\Images\data_train',
                                                       target_size=(63, 63),
                                                       batch_size=15,
                                                       class_mode='categorical')
testdata_datagenerator = ImageDataGenerator(rescale=1./255)
testdata_set = traindata_datagenerator.flow_from_directory(r'C:\Users\zangd\PycharmProjects\Nhom01_NhanDienSua\Images\data_test',
                                                           target_size=(63, 63),
                                                           batch_size=15,
                                                           class_mode='categorical')

classes = ['Fami', 'Milo', 'TH', 'Vina']
print("...Hoàn thành xử lý dữ liệu")
cnn = tf.keras.models.Sequential()
print("Đang thiết lập mạng CNN...")
cnn.add(tf.keras.layers.Conv2D(filters=40, kernel_size=3, activation='relu', input_shape=[63, 63, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=40, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=32, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Đang huấn luyện...")



history = cnn.fit(traindata_set, epochs=10, validation_data=testdata_set)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

cnn.save("model_yogurt.h5")