import os
import tkinter as tki
from ClassificationFunctions import *
from PIL import Image, ImageTk
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
        self.root.title("FaceClassifier2.ParallelSystem")
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

        lbl1 = tki.Label(self.frame_settings, text="Введите количество эталонов", font=('Helvetica', 12, 'bold'))
        lbl1.grid(row=1, column=0, padx = 40, pady = 10, sticky='W')

        self.e1 = tki.Entry(self.frame_settings, width=15, bg='floral white')
        self.e1.grid(row=2, column=0, padx = 40, sticky='W')

        btn1 = tki.Button(self.frame_settings, text="Готово", bg='floral white', width=5, command=self.get_folds)
        btn1.grid(row=2, column=0, pady = 8, sticky='E')

        #--------------

        lbl2 = tki.Label(self.frame_settings, text="Введите значения параметров", font=('Helvetica', 12, 'bold'))
        lbl2.grid(row=3, column=0, padx = 40, pady = 10, sticky='W')      

        lbl3 = tki.Label(self.frame_settings, text="Гистограмма: ")
        lbl3.grid(row=4, column=0, padx = 40, pady = 10, sticky='W') 

        self.e2 = tki.Entry(self.frame_settings, width=15, bg='floral white')
        self.e2.grid(row=4, column=0, pady = 8, sticky='E')

        lbl4 = tki.Label(self.frame_settings, text="DFT: ")
        lbl4.grid(row=5, column=0, padx = 40, pady = 10, sticky='W') 

        self.e3 = tki.Entry(self.frame_settings, width=15, bg='floral white')
        self.e3.grid(row=5, column=0, pady = 8, sticky='E')

        lbl5 = tki.Label(self.frame_settings, text="DCT: ")
        lbl5.grid(row=6, column=0, padx = 40, pady = 10, sticky='W') 

        self.e4 = tki.Entry(self.frame_settings, width=15, bg='floral white')
        self.e4.grid(row=6, column=0, pady = 8, sticky='E')

        lbl6 = tki.Label(self.frame_settings, text="Градиент: ")
        lbl6.grid(row=7, column=0, padx = 40, pady = 10, sticky='W') 

        self.e5 = tki.Entry(self.frame_settings, width=15, bg='floral white')
        self.e5.grid(row=7, column=0, pady = 8, sticky='E')

        lbl7 = tki.Label(self.frame_settings, text="Scale: ")
        lbl7.grid(row=8, column=0, padx = 40, pady = 10, sticky='W') 

        self.e6 = tki.Entry(self.frame_settings, width=15, bg='floral white')
        self.e6.grid(row=8, column=0, pady = 8, sticky='E')

        btn2 = tki.Button(self.frame_settings, text="Готово", bg='floral white', command=self.get_parameters)
        btn2.grid(row=9, column=0, pady = 8, sticky='E')

        lbl8 = tki.Label(self.frame_settings, text="Введите номер изображения", font=('Helvetica', 12, 'bold'))
        lbl8.grid(row=10, column=0, padx = 40, pady = 10, sticky='W')

        self.label_diap = tki.Label(self.frame_settings, text="", font=('Helvetica', 8))
        self.label_diap.grid(row=11, column=0, padx = 40, sticky='W')
        
        self.e7 = tki.Entry(self.frame_settings, width=15, bg='floral white')
        self.e7.grid(row=12, column=0, padx = 40, sticky='W')
        
        btn3 = tki.Button(self.frame_settings, text="Старт", bg='floral white', command=self.get_pic_number_and_start)
        btn3.grid(row=12, column=0, pady = 8, sticky='E')

        self.label_diap2 = tki.Label(self.frame_settings, text="Измененное изображение можно выбрать из [0,39]", font=('Helvetica', 8))
        self.label_diap2.grid(row=13, column=0, padx = 40, sticky='W')
        
        self.e8 = tki.Entry(self.frame_settings, width=15, bg='floral white')
        self.e8.grid(row=14, column=0, padx = 40, sticky='W')
        
        btn4 = tki.Button(self.frame_settings, text="Старт", bg='floral white', command=self.get_cloaked_and_start)
        btn4.grid(row=14, column=0, pady = 8, sticky='E')

        self.label_diap3 = tki.Label(self.frame_settings, text="Изображение в маске можно выбрать из [0,39]", font=('Helvetica', 8))
        self.label_diap3.grid(row=15, column=0, padx = 40, sticky='W')
        
        self.e9 = tki.Entry(self.frame_settings, width=15, bg='floral white')
        self.e9.grid(row=16, column=0, padx = 40, sticky='W')
        
        btn5 = tki.Button(self.frame_settings, text="Старт", bg='floral white', command=self.get_mask_and_start)
        btn5.grid(row=16, column=0, pady = 8, sticky='E')

        btn6 = tki.Button(self.frame_settings, text="Рассчитать лучшее число эталонов", bg='floral white', command=self.get_best_folds_num)
        btn6.grid(row=17, column=0, pady = 8, sticky='E')

        # =====================================================

        lbl23 = tki.Label(self.frame_settings, text="________________________", font=('Helvetica', 16, 'bold'))
        lbl23.grid(row=18, column=0,sticky='E')

        lbl24 = tki.Label(self.frame_settings, text="     ", font=('Helvetica', 12, 'bold'))
        lbl24.grid(row=19, column=0, padx = 40, pady = 8, sticky='W')

        lbl25 = tki.Label(self.frame_settings, text="Лучшее число эталонов:", font=('Helvetica', 12, 'bold'))
        lbl25.grid(row=20, column=0, padx = 40, pady = 8, sticky='W')

        lbl26 = tki.Label(self.frame_settings, text="Лучшее значение точности:", font=('Helvetica', 12, 'bold'))
        lbl26.grid(row=21, column=0, padx = 40, pady = 8, sticky='W')

        lbl27 = tki.Label(self.frame_settings, text="Графики", font=('Helvetica', 12, 'bold'))
        lbl27.grid(row=22, column=0, padx = 40, pady = 8, sticky='W')

        #---------------

        lbl9 = tki.Label(self.frame_settings, text=" ")
        lbl9.grid(row=1, column=1, padx = 10, pady = 10, sticky='W')

        lbl10 = tki.Label(self.frame_settings, text="Изображение:", font=('Helvetica', 12, 'bold'))
        lbl10.grid(row=1, column=2, padx = 40, pady = 10, sticky='W')

        lbl11 = tki.Label(self.frame_settings, text="Голоса методов:", font=('Helvetica', 12, 'bold'))
        lbl11.grid(row=3, column=2, padx = 40, pady = 10, sticky='W')

        lbl12 = tki.Label(self.frame_settings, text="Итог:", font=('Helvetica', 12, 'bold'))
        lbl12.grid(row=5, column=2, padx = 40, pady = 10, sticky='W')

        lbl13 = tki.Label(self.frame_settings, text="Человек под номером:", font=('Helvetica', 12, 'bold'))
        lbl13.grid(row=7, column=2, padx = 40, pady = 10, sticky='W')

        #-------------
        
        self.data = get_faces()
        self.cloaked_data = get_cloaked_faces()
        self.mask_data = get_mask_faces()

        self.parameters = []
        self.res = []
        self.folds = 0
        self.pic_number = 0
        self.res_number = 0
        self.img_massiv = []


        self.pic = tki.Label(self.frame_settings)
        self.pic.grid(row = 0, column = 3, sticky='N', padx=10, pady=2, rowspan=3)

        self.img_results = [tki.Label(self.frame_settings), tki.Label(self.frame_settings), tki.Label(self.frame_settings), tki.Label(self.frame_settings),
        tki.Label(self.frame_settings)]
        for i in range(len(self.img_results)):
                self.img_results[i].grid(row = 3, column = i+3, sticky='N', padx=10, pady=2, rowspan=3)
        
        # lbl9 = tki.Label(self.frame_settings, text="Результат: ")
        # lbl9.grid(row=6, column=2, padx = 20, pady = 10, rowspan=3) 

        self.pic_res = tki.Label(self.frame_settings)
        self.pic_res.grid(row = 5, column = 3, sticky='N', padx=10, pady=2, rowspan=3)

        # lbl10 = tki.Label(self.frame_settings, text="№")
        # lbl10.grid(row=6, column=2, padx = 25, sticky='W', rowspan=3)

        # Вывод номера человека
        self.label_pic_res = tki.Label(self.frame_settings, text="")
        self.label_pic_res.grid(row=7, column=3)

        # =================================================================

        # Лучшее число эталонов (с максимальной средней точностью)
        self.best_folds = tki.Label(self.frame_settings, text="")
        self.best_folds.grid(row=20, column=0, sticky='E')

        # Лучшее значение точности
        self.best_folds_accuracy = tki.Label(self.frame_settings, text="")
        self.best_folds_accuracy.grid(row=21, column=0, sticky='E')

        # График
        self.chart = tki.Label(self.frame_settings, text="")
        self.chart.grid(row=23, column=0, padx = 40, pady = 8, sticky='W', columnspan = 3, rowspan = 20)

        self.chart2 = tki.Label(self.frame_settings, text="")
        self.chart2.grid(row=43, column=0, padx = 40, pady = 8, sticky='W', columnspan = 3, rowspan = 20)


    def get_folds(self):
        f1 = float(self.e1.get())
        self.folds = int(f1)
        self.label_diap.configure(text = "Тестовое изображение можно выбрать из [" + str(0) + "," + str(399 - 40*(self.folds))+"]")
        print("folds = " + str(self.folds))

    def get_parameters(self):
        p1 = float(self.e2.get())      
        p2 = float(self.e3.get()) 
        p3 = float(self.e4.get())  
        p4 = float(self.e5.get())
        p5 = float(self.e6.get())
        self.parameters = []
        self.parameters.append(int(p1)) 
        self.parameters.append(int(p2))
        self.parameters.append(int(p3)) 
        self.parameters.append(int(p4))   
        self.parameters.append(p5)  
        print('params = ' + str(self.parameters))

    def get_pic_number_and_start(self):
        n1 = float(self.e7.get())
        self.pic_number = int(n1)
        self.res = parallel_classifier('for_best_param', self.folds, self.pic_number, self.parameters)   
        # Массив изображений, за которые проголосовали каждый из методов 
        self.img_massiv = self.res[0]     
        # Индекс результирующего изображения   
        self.res_number = self.res[1]
        
        #showing results
        pic_index = self.pic_number + self.folds * ((self.pic_number // (10 - self.folds))+1)
        img_res = self.res_number*10
        image = Image.fromarray(self.data[0][pic_index]*255)
        image = ImageTk.PhotoImage(image)
        self.pic.configure(image=image)
        self.pic.image = image
        for i in range(len(self.img_massiv)):
            image = Image.fromarray(self.data[0][self.img_massiv[i]]*255)
            image = ImageTk.PhotoImage(image)
            self.img_results[i].configure(image=image)
            self.img_results[i].image = image
        image = Image.fromarray(self.data[0][img_res]*255)
        image = ImageTk.PhotoImage(image)
        self.pic_res.configure(image=image)
        self.pic_res.image = image
        self.label_pic_res.configure(text = str(self.res_number + 1))

    def get_cloaked_and_start(self):
        n1 = float(self.e8.get())
        self.pic_number = int(n1)

        self.res = parallel_classifier('cloaked', self.folds, self.pic_number, self.parameters)   
        # Массив изображений, за которые проголосовали каждый из методов 
        self.img_massiv = self.res[0]     
        # Индекс результирующего изображения   
        self.res_number = self.res[1]
        print('self.res[1] = ' + str(self.res[1]))
        
        #showing results
        img_res = self.res_number*10
        image = Image.fromarray(self.cloaked_data[0][self.pic_number])
        image = ImageTk.PhotoImage(image)
        self.pic.configure(image=image)
        self.pic.image = image
        for i in range(len(self.img_massiv)):
            image = Image.fromarray(self.data[0][self.img_massiv[i]]*255)
            image = ImageTk.PhotoImage(image)
            self.img_results[i].configure(image=image)
            self.img_results[i].image = image
        image = Image.fromarray(self.data[0][img_res]*255)
        image = ImageTk.PhotoImage(image)
        self.pic_res.configure(image=image)
        self.pic_res.image = image
        self.label_pic_res.configure(text = str(self.res_number + 1))

    def get_mask_and_start(self):
        n1 = float(self.e9.get())
        self.pic_number = int(n1)

        self.res = parallel_classifier('mask', self.folds, self.pic_number, self.parameters)   
        # Массив изображений, за которые проголосовали каждый из методов 
        self.img_massiv = self.res[0]     
        # Индекс результирующего изображения   
        self.res_number = self.res[1]
        print('self.res[1] = ' + str(self.res[1]))
        
        #showing results
        img_res = self.res_number*10
        image = Image.fromarray(self.mask_data[0][self.pic_number])
        image = ImageTk.PhotoImage(image)
        self.pic.configure(image=image)
        self.pic.image = image
        for i in range(len(self.img_massiv)):
            image = Image.fromarray(self.data[0][self.img_massiv[i]]*255)
            image = ImageTk.PhotoImage(image)
            self.img_results[i].configure(image=image)
            self.img_results[i].image = image
        image = Image.fromarray(self.data[0][img_res]*255)
        image = ImageTk.PhotoImage(image)
        self.pic_res.configure(image=image)
        self.pic_res.image = image
        self.label_pic_res.configure(text = str(self.res_number + 1))


    def get_best_folds_num(self):
        # Инициализация    
        folds_accuracy = [0 for j in range(9)]

        for num_folds in range(1,10):
            idx_folds = num_folds-1
            folds_accuracy[idx_folds] = parallel_fold_accuracy('for_best_param', num_folds, self.parameters)
            print('folds_accuracy[' + str(num_folds) + '] = ' + str(folds_accuracy[idx_folds]))

        best_folds_accuracy = 0
        best_folds = list()
        for i in range(9):
            if folds_accuracy[i] == best_folds_accuracy:
                best_folds.append(i+1)
            elif folds_accuracy[i] > best_folds_accuracy:
                best_folds_accuracy = folds_accuracy[i]
                best_folds.clear()
                best_folds.append(i+1)
        print ('best_folds_accuracy = ' + str(best_folds_accuracy))
        print ('best_folds = ' + str(best_folds))

        folds = [j for j in range(1,10)]
        fig = plt.figure(figsize=(5,3))
        fig.subplots_adjust (bottom = 0.2)
        ax = fig.add_subplot(111)
        ax.set(xlim = [0, 10],
            ylim = [50, 100],
            title = 'Средняя точность по числу эталонов',
            xlabel = 'Число эталонов',
            ylabel = 'Точность,%')
        ax.plot(folds, folds_accuracy)
        # ax.scatter(folds, total_folds_accuracy, s=9)
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        image = Image.open(buf)
        image = ImageTk.PhotoImage(image)
        self.chart.configure(image=image)
        self.chart.image = image

        num_test_accuracy = parallel_accuracy_test_num('for_best_param', best_folds[0], self.parameters)

        # num_test_accuracy = [0 for j in range(80)]
        # num_test_accuracy[0] = 1.0
        # num_test_accuracy[1] = 1.0
        # num_test_accuracy[2] = 1.0
        # num_test_accuracy[3] = 1.0
        # num_test_accuracy[4] = 1.0
        # num_test_accuracy[5] = 1.0
        # num_test_accuracy[6] = 1.0
        # num_test_accuracy[7] = 1.0
        # num_test_accuracy[8] = 1.0
        # num_test_accuracy[9] = 0.0
        # num_test_accuracy[10] = 0.09090909090909091
        # num_test_accuracy[11] = 0.16666666666666666
        # num_test_accuracy[12] = 0.23076923076923078
        # num_test_accuracy[13] = 0.2857142857142857
        # num_test_accuracy[14] = 0.0
        # num_test_accuracy[15] = 0.0625
        # num_test_accuracy[16] = 0.11764705882352941
        # num_test_accuracy[17] = 0.16666666666666666
        # num_test_accuracy[18] = 0.21052631578947367
        # num_test_accuracy[19] = 0.0
        # num_test_accuracy[20] = 0.047619047619047616
        # num_test_accuracy[21] = 0.09090909090909091
        # num_test_accuracy[22] = 0.13043478260869565
        # num_test_accuracy[23] = 0.16666666666666666
        # num_test_accuracy[24] = 0.2
        # num_test_accuracy[25] = 0.23076923076923078
        # num_test_accuracy[26] = 0.25925925925925924
        # num_test_accuracy[27] = 0.2857142857142857
        # num_test_accuracy[28] = 0.3103448275862069
        # num_test_accuracy[29] = 0.3333333333333333
        # num_test_accuracy[30] = 0.3548387096774194
        # num_test_accuracy[31] = 0.375
        # num_test_accuracy[32] = 0.3939393939393939
        # num_test_accuracy[33] = 0.4117647058823529
        # num_test_accuracy[34] = 0.42857142857142855
        # num_test_accuracy[35] = 0.4444444444444444
        # num_test_accuracy[36] = 0.4594594594594595
        # num_test_accuracy[37] = 0.47368421052631576
        # num_test_accuracy[38] = 0.48717948717948717
        # num_test_accuracy[39] = 0.5
        # num_test_accuracy[40] = 0.5121951219512195
        # num_test_accuracy[41] = 0.5238095238095238
        # num_test_accuracy[42] = 0.5348837209302325
        # num_test_accuracy[43] = 0.5454545454545454
        # num_test_accuracy[44] = 0.5555555555555556
        # num_test_accuracy[45] = 0.5652173913043478
        # num_test_accuracy[46] = 0.574468085106383
        # num_test_accuracy[47] = 0.5833333333333334
        # num_test_accuracy[48] = 0.5918367346938775
        # num_test_accuracy[49] = 0.6
        # num_test_accuracy[50] = 0.6078431372549019
        # num_test_accuracy[51] = 0.6153846153846154
        # num_test_accuracy[52] = 0.6226415094339622
        # num_test_accuracy[53] = 0.6296296296296297
        # num_test_accuracy[54] = 0.6363636363636364
        # num_test_accuracy[55] = 0.6428571428571429
        # num_test_accuracy[56] = 0.6491228070175439
        # num_test_accuracy[57] = 0.6551724137931034
        # num_test_accuracy[58] = 0.6610169491525424
        # num_test_accuracy[59] = 0.6666666666666666
        # num_test_accuracy[60] = 0.6721311475409836
        # num_test_accuracy[61] = 0.6774193548387096
        # num_test_accuracy[62] = 0.6825396825396826
        # num_test_accuracy[63] = 0.6875
        # num_test_accuracy[64] = 0.6923076923076923
        # num_test_accuracy[65] = 0.696969696969697
        # num_test_accuracy[66] = 0.7014925373134329
        # num_test_accuracy[67] = 0.7058823529411765
        # num_test_accuracy[68] = 0.7101449275362319
        # num_test_accuracy[69] = 0.7142857142857143
        # num_test_accuracy[70] = 0.7183098591549296
        # num_test_accuracy[71] = 0.7222222222222222
        # num_test_accuracy[72] = 0.726027397260274
        # num_test_accuracy[73] = 0.7297297297297297
        # num_test_accuracy[74] = 0.7333333333333333
        # num_test_accuracy[75] = 0.7368421052631579
        # num_test_accuracy[76] = 0.7402597402597403
        # num_test_accuracy[77] = 0.7435897435897436
        # num_test_accuracy[78] = 0.7468354430379747
        # num_test_accuracy[79] = 0.75
        # best_folds = [8,9]

        num_of_tests = [i for i in range(1,401-40*(best_folds[0]))]
        fig = plt.figure(figsize=(5,3))
        fig.subplots_adjust (bottom = 0.2)
        ax = fig.add_subplot(111)
        ax.set(xlim = [0, 400-40*(best_folds[0])],
            ylim = [-10, 110],
            title = 'Точность при 8 эталонах',
            xlabel = 'Число тестовых изображений',
            ylabel = 'Точность,%')
        ax.plot(num_of_tests, num_test_accuracy)
        # ax.scatter(folds, total_folds_accuracy, s=9)
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        image = Image.open(buf)
        image = ImageTk.PhotoImage(image)
        self.chart2.configure(image=image)
        self.chart2.image = image

        best_folds_str = delStaples(str(best_folds))

        self.best_folds.configure(text = best_folds_str)
        self.best_folds_accuracy.configure(text = str(best_folds_accuracy) + '%') 


    def resize_s(self, event):
        region = self.canvas.bbox(tki.ALL)
        self.canvas.configure(scrollregion=region)

    def onClose(self):
        print("[INFO] closing...")
        self.root.quit()

pba = PhotoBoothApp()
pba.root.mainloop()