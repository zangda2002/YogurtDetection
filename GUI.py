import numpy as np
from keras_preprocessing import image
import cv2
import os
from tensorflow.keras.models import load_model
from tkinter import *
import tkinter.messagebox

def Run():
    vid = cv2.VideoCapture(0)
    print("Kết nối camera thành công")
    i = 0
    classes = ['Fami', 'Milo', 'TH', 'Vina']
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

def quit():
    answer = tkinter.messagebox.askyesno("Thoát không?")
    if answer:
        root.destroy()

root = tkinter.Tk()
root.title("Nhận diện sữa hộp")
root.minsize(width=500, height=500)

icon = PhotoImage(file=r'C:\Users\zangd\PycharmProjects\Nhom01_NhanDienSua\ictuicon.png')
root.iconphoto(True, icon)

Label(root,text='PHẦN MỀM NHẬN DIỆN SỮA HỘP',fg='red',font=('arial',20),width=50).grid(row=0)
Label(root,text='NHÓM 1',fg='red',font=('arial',20),width=50).grid(row=1)
Label(root,text='GV HƯỚNG DẪN: TS.TRẦN QUANG QUÝ',fg='red',font=('arial',20),width=50).grid(row=2)
Label(root,text=' ', font=('arial',20)).grid(row=3)
Label(root,text='THÀNH VIÊN NHÓM:', font=('arial',20)).grid(row=4)
Label(root,text='-Nhóm trưởng: Nguyễn Lê Anh Vũ', font=('Times New Roman',18)).grid(row=5)
Label(root,text='-Đỗ Văn Đạt', font=('Times New Roman',18)).grid(row=6)
Label(root,text='-Lê Việt Dũng', font=('Times New Roman',18)).grid(row=7)
Label(root,text=' ', font=('arial',20)).grid(row=8)

button=Frame(root)
Button(button, text='Bắt đầu nhận dạng!', width='20', font=('Algerian',18), command=Run).pack(side=LEFT)
button.grid(row=9)
Button(button, text='Thoát!', width='20', font=('Algerian',18), command=quit).pack(side=LEFT)
button.grid(row=9)

root.mainloop()
