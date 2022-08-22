import numpy as np
from scipy.fftpack import dct
import cv2
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===============================

def distance(el1, el2):
    return np.linalg.norm(np.array(el1) - np.array(el2))

def delStaples(line):
    line = line[1:]
    line = line[:len(line)-1]
    return line

def toFixed(x):
    return float('{:.1f}'.format(x))


# МЕТОДЫ ================================================

def get_histogram(image, param = 30):
    hist, bins = np.histogram(image, bins=np.linspace(0, 1, param))
    return [hist, bins]

def get_dft(image, mat_side = 13):
    f = np.fft.fft2(image)
    f = f[0:mat_side, 0:mat_side]
    return np.abs(f)

def get_dct(image, mat_side = 13):
    c = dct(image, axis=1)
    c = dct(c, axis=0)
    c = c[0:mat_side, 0:mat_side]
    return c

def get_gradient(image, n = 2):
    n=n-1
    shape = image.shape[0]
    i, l = 0, 0
    r = n
    result = []

    while r <= shape:
        window = image[l:r, :]
        result.append(np.sum(window))
        i += 1
        l = i * n
        r = (i + 1) * n
    result = np.array(result)
    return result

def get_scale(image, scale = 0.35):
    # image = image.astype('float32') 
    h = image.shape[0]
    w = image.shape[1]
    new_size = (int(h * scale), int(w * scale))
    return cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)


# ПОЛУЧАЕМ ИЗОБРАЖЕНИЯ ЛИЦ ===============================

def get_faces():
    data_images = fetch_olivetti_faces()
    data_target = data_images['target']
    data_images = data_images['images']
    # width = data_images[0].shape[1]
    # height = data_images[0].shape[0]
    # print('Размер квадрата = ' + str(width) + '*' + str(height))
    return [data_images, data_target]

def get_cloaked_faces():
    cloaked_faces = list()
    cloaked_number = [i for i in range(40)]

    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/111_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/211_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/311_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/411_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/511_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/611_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/711_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/811_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/911_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1011_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1111_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1211_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1311_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1411_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1511_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1611_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1711_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1811_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1911_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2011_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2111_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2211_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2311_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2411_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2511_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2611_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2711_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2811_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2911_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3011_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3111_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3211_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3311_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3411_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3511_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3611_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3711_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3811_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3911_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/4011_cloaked.pgm")
    cv_image1 = np.array(cf1)
    cloaked_faces.append(cv_image1)
    # pillow_image = Image.fromarray(cv_image, "RGB")
    # pillow_image.save("/home/nata/Downloads/111_cloaked.pgm")
    return [cloaked_faces, cloaked_number]

def get_mask_faces():
    mask_faces = list()
    mask_number = [i for i in range(40)]

    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/1-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/2-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/3-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/4-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/5-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/6-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/7-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/8-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/9-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/10-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/11-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/12-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/13-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/14-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/15-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/16-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/17-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/18-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/19-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/20-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/21-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/22-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/23-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/24-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/25-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/26-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/27-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/28-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/29-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/30-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/31-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/32-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/33-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/34-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/35-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/36-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/37-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/38-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/39-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    cf1 = Image.open("/home/nata/1.SPBU_homework/8_Прикладные задачи построения современных вычислительных систем/Classifier1/40-with-mask.pgm")
    cv_image1 = np.array(cf1)
    mask_faces.append(cv_image1)
    return [mask_faces, mask_number]


# ДЕЛИМ ДАННЫЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ ===========

def split_train_test_cv(data, target, images_per_person_in_train):
    images_per_person = 10
    images_all = len(data)
    images_per_person_in_test = images_per_person - images_per_person_in_train
    #if images_per_person_in_train > 8:
    #    images_per_person_in_train = 8
    #if images_per_person_in_test > 10 - images_per_person_in_train:
    #    images_per_person_in_test = 10 - images_per_person_in_train
    
    x_train, x_test, y_train, y_test = [], [], [], []
    # Цикл от 0 до конца массива с изображениями с шагом 10 (потому что на 1 человека идет 10 изображений)  
    for i in range(0, images_all, images_per_person):
        # indices - список с номерами от i до i+10
        indices_train = list(range(i, i + images_per_person_in_train))
        # print('indices_train = ' + str(indices_train))
        x_train.extend(data[index] for index in indices_train)
        y_train.extend(target[index] for index in indices_train)
        indices_test = list(range(i + images_per_person_in_train, i + 10))
        # print('indices_test = ' + str(indices_test))
        x_test.extend(data[index] for index in indices_test)
        y_test.extend(target[index] for index in indices_test)
    
    return x_train, x_test, y_train, y_test


# КЛАССИФИКАЦИЯ ==========================================

def all_classifier (sign, train, test, method, method_name, parameter):
    featured_data = get_feature_i(sign, train[0], method, method_name, parameter)
    featured_elements = get_feature(sign, test, method, method_name, parameter)

    result = []
    for element in featured_elements:
        min_el = [1000, -1]
        for i in range(len(featured_data)):
            dist = distance(element, featured_data[i])
            # if method_name == 'get_histogram': 
            #     print ("Дистанция: " + str(dist))
            if dist < min_el[0]:
                min_el = [dist, i]
        if min_el[1] < 0:
            result.append(0)
        else:
            result.append(min_el[1])
    return result

def classifier (sign, trainXY, testXY1img, method, method_name, parameter):
    answers = all_classifier(sign, trainXY, testXY1img[0], method, method_name, parameter)
    return answers

def get_feature_i (sign, imgs, method, method_name, parameter):
    result = []
    if sign == 'normal_faces':
        for i in range(len(imgs)):
            if method_name == 'get_histogram':
                result.append(method(imgs[i]/255, parameter)[0])
            elif method_name == 'get_dft':
                result.append(method(imgs[i]/255, parameter))
            elif method_name == 'get_dct':
                result.append(method(imgs[i]/255, parameter))
            elif method_name == 'get_gradient':
                result.append(method(imgs[i]/255, parameter))
            # wrong!
            elif method_name == 'get_scale':
                result.append(method(imgs[i], parameter))
    if sign == 'cloaked' or sign == 'mask':
        for i in range(len(imgs)):
            if method_name == 'get_histogram':
                result.append(method(imgs[i], parameter)[0])
            elif method_name == 'get_dft':
                result.append(method(imgs[i], parameter))
            elif method_name == 'get_dct':
                result.append(method(imgs[i], parameter))
            elif method_name == 'get_gradient':
                result.append(method(imgs[i], parameter))
            # wrong!
            elif method_name == 'get_scale':
                result.append(method(imgs[i]*255, parameter))
    if sign == 'for_best_param':
        for i in range(len(imgs)):
            if method_name == 'get_histogram':
                result.append(method(imgs[i], parameter)[0])
            else:
                result.append(method(imgs[i], parameter))
    return result

def get_feature (sign, img, method, method_name, parameter):
    result = []
    if sign == 'for_best_param':
        if method_name == 'get_histogram':
            result.append(method(img, parameter)[0])
        else:
            result.append(method(img, parameter))
    else:
        if method_name == 'get_histogram':
            result.append(method(img/255, parameter)[0])
        elif method_name == 'get_dft':
            result.append(method(img/255, parameter))
        elif method_name == 'get_dct':
            result.append(method(img/255, parameter))
        elif method_name == 'get_gradient':
            result.append(method(img/255, parameter))
        elif method_name == 'get_scale':
            result.append(method(img, parameter))
    return result

def accuracy_test_num(sign, method, method_name, folds, parameter):
    data = get_faces()
    X = data[0]
    Y = data[1]
    
    # Делим выборку на train (массив) и test(одно выбранное изображение не из train, которое будем классифицировать)
    x_train, x_test, y_train, y_test = split_train_test_cv(X,Y,folds)
    train = [x_train,y_train]

    test_indexes = list(range(400-40*(folds)))
    print('test_indexes = ' + str(test_indexes))

    # Инициализация
    num_test_accuracy = [0 for i in range(len(test_indexes))]

    for test_num in range(len(test_indexes)):
        print('test_indexes[test_num] = ' + str(test_indexes[test_num]))
        x_test_i = x_test[test_indexes[test_num]]
        y_test_i = y_test[test_indexes[test_num]]
        test = [x_test_i,y_test_i]

        index_res = classifier(sign, train, test, method, method_name, parameter)
        # face_number = index_res[0]//10
        face_number = y_train[index_res[0]]

        print('y_test_i = ' + str(y_test_i))
        print('face_number = ' + str(face_number))        
        print('------------------')

        if face_number == y_test_i:
            if test_num == 0:
                num_test_accuracy[test_num] = 1
            else:
                num_test_accuracy[test_num] = num_test_accuracy[test_num - 1] + 1

    for test_num in range(len(test_indexes)):
        num_test_accuracy[test_num] = num_test_accuracy[test_num]/(test_num+1)
        num_test_accuracy[test_num] = num_test_accuracy[test_num]*100
        print('num_test_accuracy[' + str(test_num) + '] = ' + str(num_test_accuracy[test_num]))
    return num_test_accuracy


# ПАРАЛЛЕЛЬНАЯ СИСТЕМА=====================================

def parallel_classifier (sign, folds, image_number, parameters):
    # загрузка набора данных 
    data = get_faces()
    X = data[0]
    Y = data[1]
    
    # Делим выборку на train (массив) и test(одно выбранное изображение не из train, которое будем классифицировать)
    x_train, x_test, y_train, y_test = split_train_test_cv(X,Y,folds)
    train = [x_train,y_train]

    if sign == 'normal_faces' or sign == 'for_best_param':
        x_test_i = x_test[image_number]
        y_test_i = y_test[image_number]
    if sign == 'cloaked':
        data2 = get_cloaked_faces()
        x_test_i = data2[0][image_number]
        y_test_i = data2[1][image_number]
    if sign == 'mask':
        data2 = get_mask_faces()
        x_test_i = data2[0][image_number]
        y_test_i = data2[1][image_number]
    
    test = [x_test_i,y_test_i]

    test_results = get_voices(sign, train, test, folds, parameters)


    pic = []
    face_number = []
    for i in range(5):
        pic.append(test_results[i]%10) #для вывода картинки на экран
        face_number.append(test_results[i]//10)

    # Инициализируем массив голосов (за каждое из 40 изображений пока что 0 голосов)
    votes = []
    for i in range(40):
        votes.append(0)

    # Учитываем голоса методов
    for i in range(5):
        votes[face_number[i]] = votes[face_number[i]]+1

    # Находим изображение, за которое быо отдано большее количество голосов
    max_i = 0
    index = 0
    for i in range(40):
        if votes[i]> max_i:
            max_i = votes[i]
            index = i
    print ("result (people index) = " + str(index))
    return test_results, index

def get_voices(sign, X_train, x_test_i, folds, parameter):
    methods_names = ['get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale']
    methods = [get_histogram, get_dft, get_dct, get_gradient, get_scale]
    #parameter = [43, 13, 8, 3, 0.3]
    test_results = []
    # Для каждого из методов
    for i in range(5):
        m = methods[i]
        m_name = methods_names[i]
        test_results.append(classifier(sign, X_train, x_test_i, m, m_name, parameter[i])) #получили i - номер тестового изображения, что ближе всех
    #print (test_results)
    for i in range(5):
        v = test_results[i]
        a = int(''.join(str(j) for j in v))
        b = (a//folds)*(10 - folds) + a
        test_results[i] = b
    print ('voices = ' + str(test_results))
    return test_results


# ПАРАЛЛЕЛЬНАЯ СИСТЕМА. ЛУЧШИЕ ЗНАЧЕНИЯ====================

def parallel_fold_accuracy(sign, folds, parameters):
    data = get_faces()
    X = data[0]
    Y = data[1]
    
    accuracy = 0

    # Делим выборку на train (массив) и test(одно выбранное изображение не из train, которое будем классифицировать)
    X_train, X_test, y_train, y_test = split_train_test_cv(X,Y,folds)
    train = [X_train,y_train]
    for j in range(len(X_test)):
        x_test_i = X_test[j]
        y_test_i = y_test[j]
        test = [x_test_i,y_test_i]

        # Для выбранного test получаем массив голосов от каждого метода (изображения из train, наиболее близкие к test по каждому из признаков)
        test_results = get_voices(sign, train, test, folds, parameters)

        pic = []
        face_number = []
        for i in range(5):
            pic.append(test_results[i]%10) #для вывода картинки на экран
            face_number.append(test_results[i]//10)

        # Инициализируем массив голосов (за каждое из 40 изображений пока что 0 голосов)
        votes = []
        for i in range(40):
            votes.append(0)

        # Учитываем голоса методов
        for i in range(5):
            votes[face_number[i]] = votes[face_number[i]]+1

        # Находим изображение, за которое быо отдано большее количество голосов
        max_i = 0
        index = 0
        for i in range(40):
            if votes[i]> max_i:
                max_i = votes[i]
                index = i

        if index == y_test_i:
            accuracy = accuracy + 1
    accuracy = accuracy/len(X_test)
    accuracy = accuracy*100
    return accuracy

def parallel_accuracy_test_num(sign, folds, parameters):
    data = get_faces()
    X = data[0]
    Y = data[1]
    
    # Делим выборку на train (массив) и test(одно выбранное изображение не из train, которое будем классифицировать)
    X_train, X_test, y_train, y_test = split_train_test_cv(X,Y,folds)
    train = [X_train,y_train]

    test_indexes = list(range(400-40*(folds)))
    print('test_indexes = ' + str(test_indexes))

    # Инициализация
    num_test_accuracy = [0 for i in range(len(test_indexes))]

    for test_num in range(len(test_indexes)):
        x_test_i = X_test[test_indexes[test_num]]
        y_test_i = y_test[test_indexes[test_num]]
        test = [x_test_i,y_test_i]

        test_results = get_voices(sign, train, test, folds, parameters)

        pic = []
        face_number = []
        for i in range(5):
            pic.append(test_results[i]%10) #для вывода картинки на экран
            face_number.append(test_results[i]//10)

        # Инициализируем массив голосов (за каждое из 40 изображений пока что 0 голосов)
        votes = []
        for i in range(40):
            votes.append(0)

        # Учитываем голоса методов
        for i in range(5):
            votes[face_number[i]] = votes[face_number[i]]+1

        # Находим изображение, за которое быо отдано большее количество голосов
        max_i = 0
        index = 0
        for i in range(40):
            if votes[i]> max_i:
                max_i = votes[i]
                index = i

        if index == y_test_i:
            if test_num == 0:
                num_test_accuracy[test_num] = 1
            else:
                num_test_accuracy[test_num] = num_test_accuracy[test_num - 1] + 1
    for test_num in range(len(test_indexes)):
        num_test_accuracy[test_num] = num_test_accuracy[test_num]/(test_num+1)
        num_test_accuracy[test_num] = num_test_accuracy[test_num]*100
        print('num_test_accuracy[' + str(test_num) + '] = ' + str(num_test_accuracy[test_num]))
    return num_test_accuracy