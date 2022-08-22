import tkinter as tki
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import threading
import datetime
import time
import imutils
import cv2
import os
import sys
from ClassificationFunctions import *
from tkinter import messagebox as mb
from tkinter import Scrollbar
import io
import matplotlib.pyplot as plt


class PhotoBoothApp:
    def __init__(self):
        self.outputPath = os.path.dirname(os.path.abspath(__file__)) + "/data"
        self.thread = None
        self.stopEvent = None
        self.root = tki.Tk()
        self.scroll_x = tki.Scrollbar(self.root, orient=tki.HORIZONTAL)
        self.scroll_y = tki.Scrollbar(self.root, orient=tki.VERTICAL)
        self.canvas = tki.Canvas(self.root, xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        self.root.title("FaceClassifier1")
        self.root.protocol("WM_DELETE_WINDOW", self.onClose)
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.geometry("1150x650+50+50")
        self.root.resizable(False, False)
        self.canvas.grid(row=0, column=0)
        self.scroll_x.config(command=self.canvas.xview)
        self.scroll_y.config(command=self.canvas.yview)
        self.scroll_x.grid(row=1, column=0, sticky="we")
        self.scroll_y.grid(row=0, column=1, sticky="ns")
        self.frame_settings = tki.Frame(self.canvas, height=10080, width=2500)
        self.frame_settings.grid(column=0)
        
        self.canvas.config(width=1130, height=630)

        self.canvas.create_window((0, 0), window=self.frame_settings, anchor=tki.N + tki.W)
        
        self.root.bind("<Configure>", self.resize_s)
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())

        #------------

        lbl1 = tki.Label(self.frame_settings, text="Задайте количество эталонов", font=('Helvetica', 12, 'bold'))
        lbl1.grid(row=1, column=0, padx = 40, pady = 10, sticky='W')      

        self.e1 = tki.Entry(self.frame_settings, width=20, bg='floral white')
        self.e1.grid(row=2, column=0, padx = 40, sticky='W')

        btn1 = tki.Button(self.frame_settings, text="Готово", bg='floral white', width=5, command=self.get_folds)
        btn1.grid(row=2, column=0, pady = 8, sticky='E')

        btn2 = tki.Button(self.frame_settings, text="Разделить выборку", bg='floral white', command=self.get_train)
        btn2.grid(row=3, column=0, sticky='E')

        #------------

        lbl2 = tki.Label(self.frame_settings, text="Выберите признак", font=('Helvetica', 12, 'bold'))
        lbl2.grid(row=4, column=0, padx = 40, pady = 10, sticky='W')

        self.methods = ["Гистограмма", "DFT", "DCT", "Градиент", "Scale"]
        self.lbox = tki.Listbox(self.frame_settings, width=20, height=5, bg='floral white')
        self.lbox.grid(row=5, column=0, padx = 40, sticky='W', rowspan = 3)
        for method in self.methods:
            self.lbox.insert(END, method)

        btn3 = tki.Button(self.frame_settings, text="Готово", bg='floral white', width=5, command=self.get_method)
        btn3.grid(row=7, column=0, pady = 8, sticky='SE')

        #--------------

        lbl3 = tki.Label(self.frame_settings, text="Введите значение параметра признака", font=('Helvetica', 12, 'bold'))
        lbl3.grid(row=8, column=0, padx = 40, pady = 10, sticky='W')      

        self.e3 = tki.Entry(self.frame_settings, width=20, bg='floral white')
        self.e3.grid(row=9, column=0, padx = 40, sticky='W')

        btn4 = tki.Button(self.frame_settings, text="Готово", bg='floral white', width=5, command=self.get_parameter)
        btn4.grid(row=9, column=0, pady = 8, sticky='E')

        #---------------

        # btn5 = tki.Button(self.frame_settings, text="Получить значения признака",bg='floral white', command=self.get_train_results)
        # btn5.grid(row=10, column=0, pady = 8, sticky='NE')

        btn6 = tki.Button(self.frame_settings, text="Результаты для обучающей выборки",bg='floral white', command=self.get_train_classification)
        btn6.grid(row=10, column=0, sticky='NE')

        btn7 = tki.Button(self.frame_settings, text="Результаты для тестовой выборки",bg='floral white', command=self.get_test_classification)
        btn7.grid(row=11, column=0, pady = 8, sticky='NE')

        btn8 = tki.Button(self.frame_settings, text="Результаты для измененных изображений",bg='floral white', command=self.get_cloaked_results)
        btn8.grid(row=12, column=0, sticky='NE')

        btn9 = tki.Button(self.frame_settings, text="Результаты для изображений в масках",bg='floral white', command=self.get_mask_results)
        btn9.grid(row=13, column=0, pady = 8, sticky='NE')

        btn10 = tki.Button(self.frame_settings, text="Рассчитать оптимальные значения",bg='floral white', command=self.get_best_values)
        btn10.grid(row=14, column=0, sticky='NE')


        lbl23 = tki.Label(self.frame_settings, text="______________________________", font=('Helvetica', 16, 'bold'))
        lbl23.grid(row=15, column=0,sticky='E')

        lbl24 = tki.Label(self.frame_settings, text="     ", font=('Helvetica', 12, 'bold'))
        lbl24.grid(row=16, column=0, padx = 40, pady = 8, sticky='W')

        lbl25 = tki.Label(self.frame_settings, text="Лучшее число эталонов:", font=('Helvetica', 12, 'bold'))
        lbl25.grid(row=17, column=0, padx = 40, pady = 8, sticky='W')

        lbl26 = tki.Label(self.frame_settings, text="Лучшее значение параметра:", font=('Helvetica', 12, 'bold'))
        lbl26.grid(row=18, column=0, padx = 40, pady = 8, sticky='W')

        lbl27 = tki.Label(self.frame_settings, text="Лучшее значение точности:", font=('Helvetica', 12, 'bold'))
        lbl27.grid(row=20, column=0, padx = 40, pady = 8, sticky='W')

        lbl28 = tki.Label(self.frame_settings, text="     ", font=('Helvetica', 12))
        lbl28.grid(row=21, column=0, padx = 40, pady = 8, sticky='W')

        lbl29 = tki.Label(self.frame_settings, text="Графики", font=('Helvetica', 12, 'bold'))
        lbl29.grid(row=22, column=0, padx = 40, pady = 8, sticky='W')

        lbl30 = tki.Label(self.frame_settings, text="                                               ", font=('Helvetica', 12, 'bold'))
        lbl30.grid(row=1, column=1, pady = 8, sticky='W')

        #-------------

        lbl9 = tki.Label(self.frame_settings, text=" ")
        lbl9.grid(row=1, column=1, padx = 5, sticky='W')

        lbl10 = tki.Label(self.frame_settings, text="Обучающая выборка:", font=('Helvetica', 12, 'bold'))
        lbl10.grid(row=1, column=2, padx = 40, sticky='W')

        # ====================================================================================

        lbl11 = tki.Label(self.frame_settings, text="Классифицируемые изображения:", font=('Helvetica', 12, 'bold'))
        lbl11.grid(row=4, column=2, padx = 40,sticky='NW')

        lbl12 = tki.Label(self.frame_settings, text="Значения признака:", font=('Helvetica', 12, 'bold'))
        lbl12.grid(row=7, column=2, padx = 40,sticky='NW')

        lbl13 = tki.Label(self.frame_settings, text="Наиболее близкие значения признака из ОВ:", font=('Helvetica', 12, 'bold'))
        lbl13.grid(row=9, column=2, padx = 40,sticky='NW')

        lbl14 = tki.Label(self.frame_settings, text="Соответствующее изображение из ОВ:", font=('Helvetica', 12, 'bold'))
        lbl14.grid(row=11, column=2, padx = 40,sticky='SW')

        lbl15 = tki.Label(self.frame_settings, text="Человек под номером:", font=('Helvetica', 12, 'bold'))
        lbl15.grid(row=14, column=2, padx = 40,sticky='NW')

        lbl16 = tki.Label(self.frame_settings, text="Точность:", font=('Helvetica', 12, 'bold'))
        lbl16.grid(row=15, column=2, padx = 40,sticky='NW')

        # ====================================================================================


        self.data = get_faces()
        self.folds = 0
        self.x_train, self.x_test, self.y_train, self.y_test = [], [], [], []
        self.method = ''
        self.parameter = 0

        self.x_train_pic = []

        self.x_class_train_pic = []
        self.x_class_train_res = []
        self.x_train_res_train = []
        self.x_train_res_pic = []
        self.train_people_num = []
        # self.train_accuracy = []

        # self.best_folds = 0
        # self.best_param = 0
        # self.best_accuracy = 0
        self.chart2 = []
        self.chart3 = []

        num_of_imgs_return = 250
          
        # Тренировочная выборка
        j=0
        for i in range(360):
            self.x_train_pic.append(tki.Label(self.frame_settings))
            self.x_train_pic[i].grid(row = 1, column = j+3, rowspan = 2, padx=10)
            j=j+1

        # ==========================================================================================

        # Классифицируемые изображения
        j=0
        for i in range(num_of_imgs_return):
            self.x_class_train_pic.append(tki.Label(self.frame_settings))
            self.x_class_train_pic[i].grid(row = 4, column = j+3, sticky='N', rowspan = 3, padx=10)
            j=j+1

        # Значения признака
        j=0
        for i in range(num_of_imgs_return):
            self.x_class_train_res.append(tki.Label(self.frame_settings))
            self.x_class_train_res[i].grid(row = 7, column = j+3, sticky='N', rowspan = 3, padx=10)
            j=j+1

        # Наиболее близкие значения признака из tain
        j=0
        for i in range(num_of_imgs_return):
            self.x_train_res_train.append(tki.Label(self.frame_settings))
            self.x_train_res_train[i].grid(row = 9, column = j+3, sticky='N', rowspan = 3, padx=10)
            j=j+1


        # Соответствующие изображения из train
        j=0
        for i in range(num_of_imgs_return):
            self.x_train_res_pic.append(tki.Label(self.frame_settings))
            self.x_train_res_pic[i].grid(row = 11, column = j+3, rowspan = 3, padx=10)
            j=j+1
        
        # Номер человека
        j=0
        for i in range(num_of_imgs_return):
            self.train_people_num.append (tki.Label(self.frame_settings, text=""))
            self.train_people_num[i].grid(row=14, column=j+3)
            j=j+1
    
        # Точность
        self.train_accuracy = tki.Label(self.frame_settings, text="")
        self.train_accuracy.grid(row=15, column=3)


        # ==========================================================================================


        # Лучшее число эталонов (с максимальной средней точностью)
        self.best_folds = tki.Label(self.frame_settings, text="")
        self.best_folds.grid(row=17, column=0, sticky='E')

        # Лучшее значение параметра (соответствующее наивысшей точности при лучшем числе эталонов)
        self.best_param = tki.Label(self.frame_settings, text="")
        self.best_param.grid(row=19, column=0, sticky='E')

        # Лучшее значение точности (соответствует лучшему числу эталонов и значению параметра)
        self.best_accuracy = tki.Label(self.frame_settings, text="")
        self.best_accuracy.grid(row=20, column=0, sticky='E')

        # График срадней точности по числу эталонов
        self.chart1 = tki.Label(self.frame_settings, text="")
        self.chart1.grid(row=23, column=0, padx = 40, pady = 8, sticky='W', columnspan = 2, rowspan = 20)

        # График точности при лучшем числе эталонов по параметрам
        j=44
        for i in range(2):
            self.chart2.append(tki.Label(self.frame_settings, text=""))
            self.chart2[i].grid(row=j, column=0, padx = 40, pady = 8, sticky='W', columnspan = 2, rowspan = 30)
            j=j+30
        
        for i in range(8):
            self.chart3.append(tki.Label(self.frame_settings, text=""))
            self.chart3[i].grid(row=j, column=0, padx = 40, pady = 8, sticky='W', columnspan = 2, rowspan = 30)
            j=j+30



# ПОЛУЧАЕМ ЧИСЛО ЭТАЛОНОВ, ТРЕНИРОВОЧНУЮ ВЫБОРКУ, ВЫБОР МЕТОДА И ЕГО ПАРАМЕТР ========

    def get_folds(self):
        self.folds = 0
        f1 = float(self.e1.get())
        self.folds = int(f1)
        print (self.folds)

    def get_train(self):
        self.x_train, self.x_test, self.y_train, self.y_test = split_train_test_cv(self.data[0]*255, self.data[1], self.folds)
        for j in range(len(self.x_train)):
            image = Image.fromarray(self.x_train[j])
            image = ImageTk.PhotoImage(image)
            self.x_train_pic[j].configure(image=image)
            self.x_train_pic[j].image = image

    def get_method(self):    
        select = int(self.lbox.curselection()[0])
        self.method = self.methods[select]
        print(self.method)

    def get_parameter(self):
        p1 = float(self.e3.get())      
        if self.method == 'Scale':
            self.parameter = float(p1)
        else:
            self.parameter = int(p1)
        print(self.parameter)


# ПОЛУЧАЕМ РЕЗУЛЬТАТЫ ДЛЯ ИЗОБРАЖЕНИЙ ИЗ ОВ, ТВ, ИЗМЕНЕННЫХ И С МАСКАМИ ==============

    def get_train_classification (self):
        train = [self.x_train, self.y_train]
        train_indexes = list(range(40*(self.folds)))
        # indexes = rnd.sample(train_indexes, 5)
        indexes = train_indexes
        train_accuracy = 0
        j=0

        for i in range(len(indexes)):
            # Классифицируемые изображения 
            
            image = Image.fromarray(self.x_train[indexes[i]])
            image = ImageTk.PhotoImage(image)
            self.x_class_train_pic[i].configure(image=image)
            self.x_class_train_pic[i].image = image
                
            x_test_i = self.x_train[indexes[i]]
            y_test_i = self.y_train[indexes[i]]
            test = [x_test_i,y_test_i]

            if self.method == 'Гистограмма':
                # Получаем значение гистограммы 
                hist, bins = get_histogram(test[0]/255, self.parameter)
                hist = np.insert(hist, 0, 0.0)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(bins, hist)
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                # for j in range(len(self.x_train)):
                #     train[0][j] = self.x_train[j]/255
                index_res = classifier('normal_faces', train, test, get_histogram, 'get_histogram', self.parameter)
                hist, bins = get_histogram(self.x_train[index_res[0]]/255, self.parameter)
                hist = np.insert(hist, 0, 0.0)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(bins, hist)
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j = j+1

                # Изображения, соответствующие ближайшему значению признака
                
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'DFT':
                # Получаем значение DFT 
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dft(test[0], self.parameter).shape[0]),
                                        range(get_dft(test[0], self.parameter).shape[0]),
                                        np.flip(get_dft(test[0], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_dft, 'get_dft', self.parameter)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dft(self.x_train[index_res[0]], self.parameter).shape[0]),
                                        range(get_dft(self.x_train[index_res[0]], self.parameter).shape[0]),
                                        np.flip(get_dft(self.x_train[index_res[0]], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                # Изображения, соответствующие ближайшему значению признака
                
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'DCT':
                # Получаем значение DCT
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dct(test[0], self.parameter).shape[0]),
                                    range(get_dct(test[0], self.parameter).shape[0]),
                                    np.flip(get_dct(test[0], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_dct, 'get_dct', self.parameter)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dct(self.x_train[index_res[0]], self.parameter).shape[0]),
                                    range(get_dct(self.x_train[index_res[0]], self.parameter).shape[0]),
                                    np.flip(get_dct(self.x_train[index_res[0]], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                # Изображения, соответствующие ближайшему значению признака
                
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'Scale':
                # Получаем значение Scale
                res = get_scale(test[0]/255, self.parameter)
                
                image = Image.fromarray(cv2.resize(res*255, test[0].shape, interpolation = cv2.INTER_AREA))
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_scale, 'get_scale', self.parameter)
                
                res = get_scale(self.x_train[index_res[0]]/255, self.parameter)
                image = Image.fromarray(cv2.resize(res*255, self.x_train[index_res[0]].shape, interpolation = cv2.INTER_AREA))
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                # Изображения, соответствующие ближайшему значению признака
                
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'Градиент':
                # Получаем значение Градиента
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(range(0, len(get_gradient(test[0], self.parameter))), get_gradient(test[0], self.parameter))
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_gradient, 'get_gradient', self.parameter)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(range(0, len(get_gradient(self.x_train[index_res[0]], self.parameter))), get_gradient(self.x_train[index_res[0]], self.parameter))
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                # Изображения, соответствующие ближайшему значению признака
                
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.y_train[index_res[0]] == y_test_i:
                    train_accuracy = train_accuracy + 1
        train_accuracy = train_accuracy/len(indexes)
        train_accuracy = train_accuracy*100

        # print ('train_accuracy:' + str(train_accuracy))
        self.train_accuracy.configure(text = str(toFixed(train_accuracy)) + '%')            
            
    def get_test_classification (self):
            train = [self.x_train, self.y_train]
            test_indexes = list(range(400-40*(self.folds)))
            # indexes = rnd.sample(test_indexes, 5)
            indexes = test_indexes
            test_accuracy = 0
            j=0

            for i in range(len(indexes)):
                
                image = Image.fromarray(self.x_test[indexes[i]])
                image = ImageTk.PhotoImage(image)
                self.x_class_train_pic[i].configure(image=image)
                self.x_class_train_pic[i].image = image
                
                x_test_i = self.x_test[indexes[i]]
                y_test_i = self.y_test[indexes[i]]
                test = [x_test_i,y_test_i]

                if self.method == 'Гистограмма':
                    # Получаем значение гистограммы 
                    hist, bins = get_histogram(test[0]/255, self.parameter)
                    hist = np.insert(hist, 0, 0.0)
                    fig = plt.figure(figsize=(1,1))
                    ax = fig.add_subplot(111)
                    ax.plot(bins, hist)
                    plt.xticks(color='w')
                    plt.yticks(color='w')
                    buf = io.BytesIO()
                    fig.savefig(buf)
                    buf.seek(0)
                    
                    image = Image.open(buf)
                    image = ImageTk.PhotoImage(image)
                    self.x_class_train_res[j].configure(image=image)
                    self.x_class_train_res[j].image = image

                    # for j in range(len(self.x_train)):
                    #     train[0][j] = self.x_train[j]/255
                    index_res = classifier('normal_faces', train, test, get_histogram, 'get_histogram', self.parameter)

                    hist, bins = get_histogram(self.x_train[index_res[0]]/255, self.parameter)
                    hist = np.insert(hist, 0, 0.0)
                    fig = plt.figure(figsize=(1,1))
                    ax = fig.add_subplot(111)
                    ax.plot(bins, hist)
                    plt.xticks(color='w')
                    plt.yticks(color='w')
                    buf = io.BytesIO()
                    fig.savefig(buf)
                    buf.seek(0)
                    
                    image = Image.open(buf)
                    image = ImageTk.PhotoImage(image)
                    self.x_train_res_train[j].configure(image=image)
                    self.x_train_res_train[j].image = image
                    j = j+1

                    
                    image = Image.fromarray(self.x_train[index_res[0]])
                    image = ImageTk.PhotoImage(image)
                    self.x_train_res_pic[i].configure(image=image)
                    self.x_train_res_pic[i].image = image
                    self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

                if self.method == 'DFT':
                    # Получаем значение DFT 
                    fig = plt.figure(figsize=(1,1))
                    ax = fig.add_subplot(111)
                    ax.pcolormesh(range(get_dft(test[0], self.parameter).shape[0]),
                                            range(get_dft(test[0], self.parameter).shape[0]),
                                            np.flip(get_dft(test[0], self.parameter), 0), cmap="Greys")
                    plt.xticks(color='w')
                    plt.yticks(color='w')
                    buf = io.BytesIO()
                    fig.savefig(buf)
                    buf.seek(0)
                    
                    image = Image.open(buf)
                    image = ImageTk.PhotoImage(image)
                    self.x_class_train_res[j].configure(image=image)
                    self.x_class_train_res[j].image = image

                    # Получаем ближайшее значение признака и соответсвующее изображение
                    index_res = classifier('normal_faces', train, test, get_dft, 'get_dft', self.parameter)
                    fig = plt.figure(figsize=(1,1))
                    ax = fig.add_subplot(111)
                    ax.pcolormesh(range(get_dft(self.x_train[index_res[0]], self.parameter).shape[0]),
                                            range(get_dft(self.x_train[index_res[0]], self.parameter).shape[0]),
                                            np.flip(get_dft(self.x_train[index_res[0]], self.parameter), 0), cmap="Greys")
                    plt.xticks(color='w')
                    plt.yticks(color='w')
                    buf = io.BytesIO()
                    fig.savefig(buf)
                    buf.seek(0)
                    
                    image = Image.open(buf)
                    image = ImageTk.PhotoImage(image)
                    self.x_train_res_train[j].configure(image=image)
                    self.x_train_res_train[j].image = image
                    j=j+1

                    
                    image = Image.fromarray(self.x_train[index_res[0]])
                    image = ImageTk.PhotoImage(image)
                    self.x_train_res_pic[i].configure(image=image)
                    self.x_train_res_pic[i].image = image
                    self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

                if self.method == 'DCT':
                    # Получаем значение DCT
                    fig = plt.figure(figsize=(1,1))
                    ax = fig.add_subplot(111)
                    ax.pcolormesh(range(get_dct(test[0], self.parameter).shape[0]),
                                        range(get_dct(test[0], self.parameter).shape[0]),
                                        np.flip(get_dct(test[0], self.parameter), 0), cmap="Greys")
                    plt.xticks(color='w')
                    plt.yticks(color='w')
                    buf = io.BytesIO()
                    fig.savefig(buf)
                    buf.seek(0)
                    
                    image = Image.open(buf)
                    image = ImageTk.PhotoImage(image)
                    self.x_class_train_res[j].configure(image=image)
                    self.x_class_train_res[j].image = image

                    # Получаем ближайшее значение признака и соответсвующее изображение
                    index_res = classifier('normal_faces', train, test, get_dct, 'get_dct', self.parameter)
                    fig = plt.figure(figsize=(1,1))
                    ax = fig.add_subplot(111)
                    ax.pcolormesh(range(get_dct(self.x_train[index_res[0]], self.parameter).shape[0]),
                                        range(get_dct(self.x_train[index_res[0]], self.parameter).shape[0]),
                                        np.flip(get_dct(self.x_train[index_res[0]], self.parameter), 0), cmap="Greys")
                    plt.xticks(color='w')
                    plt.yticks(color='w')
                    buf = io.BytesIO()
                    fig.savefig(buf)
                    buf.seek(0)
                    
                    image = Image.open(buf)
                    image = ImageTk.PhotoImage(image)
                    self.x_train_res_train[j].configure(image=image)
                    self.x_train_res_train[j].image = image
                    j=j+1

                    # Изображения, соответствующие ближайшему значению признака
                    
                    image = Image.fromarray(self.x_train[index_res[0]])
                    image = ImageTk.PhotoImage(image)
                    self.x_train_res_pic[i].configure(image=image)
                    self.x_train_res_pic[i].image = image
                    self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

                if self.method == 'Scale':
                    # Получаем значение Scale
                    res = get_scale(test[0]/255, self.parameter)
                    
                    image = Image.fromarray(cv2.resize(res*255, test[0].shape, interpolation = cv2.INTER_AREA))
                    image = ImageTk.PhotoImage(image)
                    self.x_class_train_res[j].configure(image=image)
                    self.x_class_train_res[j].image = image

                    # Получаем ближайшее значение признака и соответсвующее изображение
                    index_res = classifier('normal_faces', train, test, get_scale, 'get_scale', self.parameter)
                    res = get_scale(self.x_train[index_res[0]]/255, self.parameter)
                    
                    image = Image.fromarray(cv2.resize(res*255, self.x_train[index_res[0]].shape, interpolation = cv2.INTER_AREA))
                    image = ImageTk.PhotoImage(image)
                    self.x_train_res_train[j].configure(image=image)
                    self.x_train_res_train[j].image = image
                    j=j+1

                    # Изображения, соответствующие ближайшему значению признака
                    
                    image = Image.fromarray(self.x_train[index_res[0]])
                    image = ImageTk.PhotoImage(image)
                    self.x_train_res_pic[i].configure(image=image)
                    self.x_train_res_pic[i].image = image
                    self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

                if self.method == 'Градиент':
                    # Получаем значение Градиента
                    fig = plt.figure(figsize=(1,1))
                    ax = fig.add_subplot(111)
                    ax.plot(range(0, len(get_gradient(test[0], self.parameter))), get_gradient(test[0], self.parameter))
                    plt.xticks(color='w')
                    plt.yticks(color='w')
                    buf = io.BytesIO()
                    fig.savefig(buf)
                    buf.seek(0)
                    
                    image = Image.open(buf)
                    image = ImageTk.PhotoImage(image)
                    self.x_class_train_res[j].configure(image=image)
                    self.x_class_train_res[j].image = image

                    # Получаем ближайшее значение признака и соответсвующее изображение
                    index_res = classifier('normal_faces', train, test, get_gradient, 'get_gradient', self.parameter)
                    fig = plt.figure(figsize=(1,1))
                    ax = fig.add_subplot(111)
                    ax.plot(range(0, len(get_gradient(self.x_train[index_res[0]], self.parameter))), get_gradient(self.x_train[index_res[0]], self.parameter))
                    plt.xticks(color='w')
                    plt.yticks(color='w')
                    buf = io.BytesIO()
                    fig.savefig(buf)
                    buf.seek(0)
                    
                    image = Image.open(buf)
                    image = ImageTk.PhotoImage(image)
                    self.x_train_res_train[j].configure(image=image)
                    self.x_train_res_train[j].image = image
                    j=j+1

                    # Изображения, соответствующие ближайшему значению признака
                    
                    image = Image.fromarray(self.x_train[index_res[0]])
                    image = ImageTk.PhotoImage(image)
                    self.x_train_res_pic[i].configure(image=image)
                    self.x_train_res_pic[i].image = image
                    self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

                if self.y_train[index_res[0]] == y_test_i:
                    test_accuracy = test_accuracy + 1
            test_accuracy = test_accuracy/len(indexes)
            test_accuracy = test_accuracy*100
            # print ('train_accuracy:' + str(train_accuracy))
            self.train_accuracy.configure(text = str(toFixed(test_accuracy)) + '%')

    def get_cloaked_results(self):
        test_accuracy = 0
        self.x_train, self.x_test, self.y_train, self.y_test = split_train_test_cv(self.data[0]*255, self.data[1], self.folds)
        train = [self.x_train, self.y_train]
        j=0

        cloaked_faces = get_cloaked_faces()
        for i in range(len(cloaked_faces[1])):
            image = Image.fromarray(cloaked_faces[0][i])
            image = ImageTk.PhotoImage(image)
            self.x_class_train_pic[i].configure(image=image)
            self.x_class_train_pic[i].image = image

            test = [cloaked_faces[0][i],cloaked_faces[1][i]]

            if self.method == 'Гистограмма':
                # Получаем значение гистограммы 
                hist, bins = get_histogram(test[0]/255, self.parameter)
                hist = np.insert(hist, 0, 0.0)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(bins, hist)
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # for j in range(len(self.x_train)):
                #     train[0][j] = self.x_train[j]/255
                index_res = classifier('normal_faces', train, test, get_histogram, 'get_histogram', self.parameter)

                hist, bins = get_histogram(self.x_train[index_res[0]]/255, self.parameter)
                hist = np.insert(hist, 0, 0.0)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(bins, hist)
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j = j+1

                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'DFT':
                # Получаем значение DFT 
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dft(test[0], self.parameter).shape[0]),
                                        range(get_dft(test[0], self.parameter).shape[0]),
                                        np.flip(get_dft(test[0], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_dft, 'get_dft', self.parameter)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dft(self.x_train[index_res[0]], self.parameter).shape[0]),
                                        range(get_dft(self.x_train[index_res[0]], self.parameter).shape[0]),
                                        np.flip(get_dft(self.x_train[index_res[0]], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'DCT':
                # Получаем значение DCT
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dct(test[0], self.parameter).shape[0]),
                                    range(get_dct(test[0], self.parameter).shape[0]),
                                    np.flip(get_dct(test[0], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_dct, 'get_dct', self.parameter)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dct(self.x_train[index_res[0]], self.parameter).shape[0]),
                                    range(get_dct(self.x_train[index_res[0]], self.parameter).shape[0]),
                                    np.flip(get_dct(self.x_train[index_res[0]], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                # Изображения, соответствующие ближайшему значению признака
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'Scale':
                # Получаем значение Scale
                res = get_scale(test[0]/255, self.parameter)
                image = Image.fromarray(cv2.resize(res*255, test[0].shape, interpolation = cv2.INTER_AREA))
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_scale, 'get_scale', self.parameter)
                res = get_scale(self.x_train[index_res[0]]/255, self.parameter)
                image = Image.fromarray(cv2.resize(res*255, self.x_train[index_res[0]].shape, interpolation = cv2.INTER_AREA))
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                # Изображения, соответствующие ближайшему значению признака
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'Градиент':
                # Получаем значение Градиента
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(range(0, len(get_gradient(test[0], self.parameter))), get_gradient(test[0], self.parameter))
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_gradient, 'get_gradient', self.parameter)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(range(0, len(get_gradient(self.x_train[index_res[0]], self.parameter))), get_gradient(self.x_train[index_res[0]], self.parameter))
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                # Изображения, соответствующие ближайшему значению признака
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.y_train[index_res[0]] == cloaked_faces[1][i]:
                test_accuracy = test_accuracy + 1
        test_accuracy = test_accuracy/len(cloaked_faces[1])
        test_accuracy = test_accuracy*100
        self.train_accuracy.configure(text = str(toFixed(test_accuracy)) + '%')

    def get_mask_results(self):
        test_accuracy = 0
        self.x_train, self.x_test, self.y_train, self.y_test = split_train_test_cv(self.data[0]*255, self.data[1], self.folds)
        train = [self.x_train, self.y_train]
        j=0

        mask_faces = get_mask_faces()
        for i in range(len(mask_faces[1])):
            image = Image.fromarray(mask_faces[0][i])
            image = ImageTk.PhotoImage(image)
            self.x_class_train_pic[i].configure(image=image)
            self.x_class_train_pic[i].image = image

            test = [mask_faces[0][i],mask_faces[1][i]]

            if self.method == 'Гистограмма':
                # Получаем значение гистограммы 
                hist, bins = get_histogram(test[0]/255, self.parameter)
                hist = np.insert(hist, 0, 0.0)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(bins, hist)
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # for j in range(len(self.x_train)):
                #     train[0][j] = self.x_train[j]/255
                index_res = classifier('normal_faces', train, test, get_histogram, 'get_histogram', self.parameter)

                hist, bins = get_histogram(self.x_train[index_res[0]]/255, self.parameter)
                hist = np.insert(hist, 0, 0.0)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(bins, hist)
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j = j+1

                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'DFT':
                # Получаем значение DFT 
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dft(test[0], self.parameter).shape[0]),
                                        range(get_dft(test[0], self.parameter).shape[0]),
                                        np.flip(get_dft(test[0], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_dft, 'get_dft', self.parameter)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dft(self.x_train[index_res[0]], self.parameter).shape[0]),
                                        range(get_dft(self.x_train[index_res[0]], self.parameter).shape[0]),
                                        np.flip(get_dft(self.x_train[index_res[0]], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'DCT':
                # Получаем значение DCT
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dct(test[0], self.parameter).shape[0]),
                                    range(get_dct(test[0], self.parameter).shape[0]),
                                    np.flip(get_dct(test[0], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_dct, 'get_dct', self.parameter)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(get_dct(self.x_train[index_res[0]], self.parameter).shape[0]),
                                    range(get_dct(self.x_train[index_res[0]], self.parameter).shape[0]),
                                    np.flip(get_dct(self.x_train[index_res[0]], self.parameter), 0), cmap="Greys")
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                # Изображения, соответствующие ближайшему значению признака
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'Scale':
                # Получаем значение Scale
                res = get_scale(test[0]/255, self.parameter)
                image = Image.fromarray(cv2.resize(res*255, test[0].shape, interpolation = cv2.INTER_AREA))
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_scale, 'get_scale', self.parameter)
                res = get_scale(self.x_train[index_res[0]]/255, self.parameter)
                image = Image.fromarray(cv2.resize(res*255, self.x_train[index_res[0]].shape, interpolation = cv2.INTER_AREA))
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                # Изображения, соответствующие ближайшему значению признака
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.method == 'Градиент':
                # Получаем значение Градиента
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(range(0, len(get_gradient(test[0], self.parameter))), get_gradient(test[0], self.parameter))
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_class_train_res[j].configure(image=image)
                self.x_class_train_res[j].image = image

                # Получаем ближайшее значение признака и соответсвующее изображение
                index_res = classifier('normal_faces', train, test, get_gradient, 'get_gradient', self.parameter)
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(111)
                ax.plot(range(0, len(get_gradient(self.x_train[index_res[0]], self.parameter))), get_gradient(self.x_train[index_res[0]], self.parameter))
                plt.xticks(color='w')
                plt.yticks(color='w')
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.x_train_res_train[j].configure(image=image)
                self.x_train_res_train[j].image = image
                j=j+1

                # Изображения, соответствующие ближайшему значению признака
                image = Image.fromarray(self.x_train[index_res[0]])
                image = ImageTk.PhotoImage(image)
                self.x_train_res_pic[i].configure(image=image)
                self.x_train_res_pic[i].image = image
                self.train_people_num[i].configure(text = str(self.y_train[index_res[0]] + 1))

            if self.y_train[index_res[0]] == mask_faces[1][i]:
                    test_accuracy = test_accuracy + 1
        test_accuracy = test_accuracy/len(mask_faces[1])
        test_accuracy = test_accuracy*100
        self.train_accuracy.configure(text = str(test_accuracy) + '%')


# ПОЛУЧАЕМ ЛУЧШИЕ ЗНАЧЕНИЯ ============================================================

    def get_best_values(self):
        if self.method == 'Гистограмма':
            params = [j for j in range(10,46)]
            method = get_histogram
            method_name = 'get_histogram'
            for_chart = 1
        if self.method == 'DFT':
            params = [j for j in range(3,21)]
            method = get_dft
            method_name = 'get_dft'
            for_chart = 1
        if self.method == 'DCT':
            params = [j for j in range(3,21)]
            method = get_dct
            method_name = 'get_dct'
            for_chart = 1
        if self.method == 'Scale':
            params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            method = get_scale
            method_name = 'get_scale'
            for_chart = 0.1
        if self.method == 'Градиент':
            params = [j for j in range(2,33)]
            method = get_gradient
            method_name = 'get_gradient'
            for_chart = 1

        # Инициализация    
        test_accuracy = [[0] * len(params) for j in range(9)]
        average_folds_accuracy = [0 for j in range(9)]

        for num_folds in range(1,10):
        # for num_folds in range(1,3):
            folds_accuracy_by_param = [0 for j in range(len(params))]
            self.x_train, self.x_test, self.y_train, self.y_test = split_train_test_cv(self.data[0]*255, self.data[1], num_folds)
            train = [self.x_train, self.y_train]
            
            # В качестве тестовой выборки будем использовать выборку размерности 40 для любого числа эталонов (берем из x_test только по одному изображению для каждого человека))
            # self.x_test_one = [0 for j in range(40)]
            # self.y_test_one = [0 for j in range(40)]
            # j = 0
            # for i in range(40):
            #     self.x_test_one[i] = self.x_test2[j]
            #     self.y_test_one[i] = self.y_test2[j]
            #     j = j+(10-num_folds)
            # test_indexes = list(j for j in range(40))


            test_indexes = list(range(400-40*(num_folds)))
            # indexes = rnd.sample(test_indexes, 5)
            indexes = test_indexes
            idx_folds = num_folds-1

            
            for param in params:
                if method_name == 'get_scale':
                    idx_param = int(param*10-1)
                else:
                    idx_param = param-params[0]
                test_accuracy[idx_folds][idx_param] = 0
                for i in range(len(self.x_test)):
                    x_test_i = self.x_test[indexes[i]]
                    y_test_i = self.y_test[indexes[i]]
                    test = [x_test_i,y_test_i]
                    index_res = classifier('normal_faces', train, test, method, method_name, param)
                    if self.y_train[index_res[0]] == y_test_i:
                        test_accuracy[idx_folds][idx_param] = test_accuracy[idx_folds][idx_param] + 1
                # test_accuracy[idx_folds][idx_param] = round(100/len(self.x_test2) * test_accuracy[idx_folds][idx_param])
                # test_accuracy[idx_folds][idx_param] = float('{:.3f}'.format(test_accuracy[idx_folds][idx_param]/len(self.x_test2))) 
                test_accuracy[idx_folds][idx_param] = test_accuracy[idx_folds][idx_param]/len(self.x_test)
                test_accuracy[idx_folds][idx_param] = test_accuracy[idx_folds][idx_param]*100
                # test_accuracy[idx_folds][idx_param] = toFixed(test_accuracy[idx_folds][idx_param])
                print ("[" + str(num_folds) + "][" + str(param) + "] = " + str(test_accuracy[idx_folds][idx_param]))
                average_folds_accuracy[idx_folds] = average_folds_accuracy[idx_folds] + test_accuracy[idx_folds][idx_param]
                # total_folds_accuracy[idx_folds] = toFixed(total_folds_accuracy[idx_folds])
                # print ("total_folds_accuracy [" + str(num_folds) + "] = " + str(total_folds_accuracy[idx_folds]))
                folds_accuracy_by_param[idx_param] = test_accuracy[idx_folds][idx_param]

            average_folds_accuracy[idx_folds] = average_folds_accuracy[idx_folds]/len(params)
            print ("average_folds_accuracy[" + str(num_folds) + "] = " + str(average_folds_accuracy[idx_folds]))
            print ("_____________")


        best_folds_accuracy = 0
        best_folds = list()
        for i in range(num_folds):
            if average_folds_accuracy[i] == best_folds_accuracy:
                best_folds.append(i+1)
            elif average_folds_accuracy[i] > best_folds_accuracy:
                best_folds_accuracy = average_folds_accuracy[i]
                best_folds.clear()
                best_folds.append(i+1)
        print ('best_folds_accuracy = ' + str(toFixed(best_folds_accuracy)))
        print ('best_folds = ' + str(best_folds))


        best_accuracy = 0
        best_param = [list() for i in range(len(best_folds))]
        for i in range(len(params)):
            for j in range(len(best_folds)):

                if method_name == 'get_scale':
                    idx_param = int(params[i]*10-1)
                else:
                    idx_param = params[i]-params[0]

                if test_accuracy[best_folds[j]-1][idx_param] == best_accuracy:
                    best_param[j].append(params[i])
                elif test_accuracy[best_folds[j]-1][idx_param] > best_accuracy:
                    best_accuracy = test_accuracy[best_folds[j]-1][idx_param]
                    best_param[j].clear()
                    best_param[j].append(params[i])
        print ('best_param = ' + str(best_param))
        print ('best_accuracy = ' + str(best_accuracy))


        folds = [j for j in range(1,10)]
        fig = plt.figure(figsize=(5,3))
        fig.subplots_adjust (bottom = 0.2)
        ax = fig.add_subplot(111)
        ax.set(xlim = [0, 10],
            ylim = [30, 100],
            title = 'Средняя точность по числу эталонов',
            xlabel = 'Число эталонов',
            ylabel = 'Точность,%')
        ax.plot(folds, average_folds_accuracy)
        # ax.scatter(folds, total_folds_accuracy, s=9)
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        image = Image.open(buf)
        image = ImageTk.PhotoImage(image)
        self.chart1.configure(image=image)
        self.chart1.image = image


        for i in range(len(best_folds)):
            if best_folds[i] == 1:
                folds_str = ' эталона'
            else:
                folds_str = ' эталонов'
            fig = plt.figure(figsize=(5,3))
            fig.subplots_adjust (bottom = 0.2)
            ax = fig.add_subplot(111)
            ax.set(xlim = [params[0]-for_chart, params[-1]+for_chart],
                ylim = [20, 100],
                title = 'Точность по параметрам для ' + str(best_folds[i]) + folds_str,
                xlabel = 'Значение параметра',
                ylabel = 'Точность,%')
            ax.plot(params, test_accuracy[best_folds[i]-1])
            # ax.scatter(params, test_accuracy[best_folds[i]-1], s=9)
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            image = Image.open(buf)
            image = ImageTk.PhotoImage(image)
            self.chart2[i].configure(image=image)
            self.chart2[i].image = image


        # if self.method == 'Гистограмма':
        #     best_folds = [9]
        #     best_param = [[30, 38, 43]]
        # if self.method == 'DFT':
        #     best_folds = [8]
        #     best_param = [[13, 14, 15, 16, 17, 18, 19, 20]]
        # if self.method == 'DCT':
        #     best_folds = [9]
        #     best_param = [[7]]
        # if self.method == 'Scale':
        #     best_folds = [9]
        #     best_param = [[0.1, 0.2, 0.3]]
        # if self.method == 'Градиент':
        #     best_folds = [7]
        #     best_param = [[3, 5]]
        

        k = 0
        for i in range(len(best_folds)):
            for j in range(len(best_param[i])):
                num_of_tests_accuracy = accuracy_test_num('for_best_param', method, method_name, best_folds[i], best_param[i][j])
                num_of_tests = [i for i in range(1,401-40*(best_folds[0]))]
                fig = plt.figure(figsize=(5,3))
                fig.subplots_adjust (bottom = 0.2)
                ax = fig.add_subplot(111)
                ax.set(xlim = [0, 400-40*(best_folds[0])],
                    ylim = [-10, 110],
                    title = 'Точность при ' + str(best_folds[i]) + ' эталонах и параметре ' + str(best_param[i][j]),
                    xlabel = 'Число тестовых изображений',
                    ylabel = 'Точность,%')
                ax.plot(num_of_tests, num_of_tests_accuracy)
                # ax.scatter(folds, total_folds_accuracy, s=9)
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                image = Image.open(buf)
                image = ImageTk.PhotoImage(image)
                self.chart3[k].configure(image=image)
                self.chart3[k].image = image
                k=k+1


        best_folds_str = delStaples(str(best_folds))
        best_param_str = delStaples(str(best_param))

        self.best_folds.configure(text = best_folds_str)
        self.best_param.configure(text = best_param_str)
        self.best_accuracy.configure(text = str(toFixed(best_accuracy)) + '%')
        


    def resize_s(self, event):
        region = self.canvas.bbox(tki.ALL)
        self.canvas.configure(scrollregion=region)

    def onClose(self):
        print("[INFO] closing...")
        self.root.quit()

pba = PhotoBoothApp()
pba.root.mainloop()