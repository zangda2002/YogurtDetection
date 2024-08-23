import numpy as np
from keras_preprocessing import image
import cv2
import os
from tensorflow.keras.models import load_model
vid = cv2.VideoCapture(0)
print("Kết nối camera thành công")
i = 0
classes = ['Fami', 'TH', 'Vina', 'Milo']
new_model = load_model('model_yogurt.h5')
while(True):
    r, frame = vid.read()
    cv2.imshow('frame', frame)
    cv2.imwrite(r'C:\Users\zangd\PycharmProjects\Nhom01_NhanDienSua\conclusion' + str(i) + ".jpg", frame)
    test_image = image.load_img(r'C:\Users\zangd\PycharmProjects\Nhom01_NhanDienSua\conclusion' + str(i) + ".jpg", target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = new_model.predict(test_image)
    result1 = result[0]
    for y in range(4):
        if result1[y] == 1.:
            break
    prediction = classes[y]
    print(prediction)
    os.remove(r'C:\Users\zangd\PycharmProjects\Nhom01_NhanDienSua\conclusion' + str(i) + ".jpg")
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()