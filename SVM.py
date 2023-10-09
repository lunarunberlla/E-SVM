import os
import cv2
import numpy as np
from colorama import Fore, init
from sklearn.svm import  SVC
import matplotlib.pyplot as plt

init()

class SupportVectorMachines:
    def __init__(self,trainpath='./西瓜数据集3.0α.csv',testpath='./西瓜数据集3.0α.csv',mode='rbf'):
        self.trainpath=trainpath
        self.testpath=testpath
        self.mode=mode

    def Dataset(self):
        Train_X,Train_Y,Test_X,Test_Y=[],[],[],[]
        Trainpath,Testpath=self.trainpath,self.testpath
        ########################加载训练数据集#################################
        source_data_file = open(Trainpath)  # 打开数据集文件
        Data = source_data_file.read()  # 将文件数据读取到Data中
        Data = Data.split('\n')
        Data = Data[1:len(Data) - 1]
        _x, _y = [], []
        for i in Data:
            _x_middle = []
            _x_middle.append(float(i.split(',')[1]))  # 将第一个特征加入
            _x_middle.append(float(i.split(',')[2]))  # 将第二个特征加入
            _x.append(_x_middle)
            if i.split(',')[3] == '是':  # 将分类的情况转换为 0和1 并将其放在y中
                _y.append(1)
            else:
                _y.append(0)
        Train_X,Train_Y=_x,_y
        #######################训练集加载完成################################
        #######################加载测试数据集################################
        source_data_file = open(Testpath)  # 打开数据集文件
        Data = source_data_file.read()  # 将文件数据读取到Data中
        Data = Data.split('\n')
        Data = Data[1:len(Data) - 1]
        _x, _y = [], []
        for i in Data:
            _x_middle = []
            _x_middle.append(float(i.split(',')[1]))  # 将第一个特征加入
            _x_middle.append(float(i.split(',')[2]))  # 将第二个特征加入
            _x.append(_x_middle)
            if i.split(',')[3] == '是':  # 将分类的情况转换为 0和1 并将其放在y中
                _y.append(1)
            else:
                _y.append(0)
        Test_X,Test_Y=_x,_y
        if len(Train_X)==len(Train_Y) and len(Test_X)==len(Test_Y):
            print(Fore.YELLOW,'文件加载完成，训练样本{}个，测试样本{}个'.format(len(Train_X),len(Test_X)))
        else:
            print(Fore.RED,"文件加载错误")
        return Train_X,Train_Y,Test_X,Test_Y

    def Trainer(self):
        if self.mode=='rbf':
            X,Y,Test_X,Test_Y=SupportVectorMachines.Dataset(self)
            model=SVC(kernel='rbf')
            model.fit(X,Y)
            print(Fore.GREEN,'训练完成，正在评估得分：')
            scor=model.score(Test_X,Test_Y)
            print(Fore.BLUE,"得分评定成功，该模型得分为：{}".format(scor))
            Prediect_Y=model.predict(Test_X)
            print(Fore.CYAN,"正在画图，请等待...")
            #################画图##########################################
            plot_x=np.arange(0,1000,1000/len(Prediect_Y))
            plt.scatter(plot_x,Prediect_Y+0.02,marker='*',label='prediect')
            plt.scatter(plot_x,Test_Y,marker='.',label='source')
            plt.xlabel('number')
            plt.ylabel('value')
            plt.legend()
            plt.title("SVM(gbf)")
            plt.show()
            print(Fore.LIGHTRED_EX,"本次训练结束")
            ################################################################
            return model
        else:
            X, Y, Test_X, Test_Y = SupportVectorMachines.Dataset(self)
            model = SVC(kernel='linear')
            model.fit(X, Y)
            print(Fore.GREEN, '训练完成，正在评估得分：')
            scor = model.score(Test_X, Test_Y)
            print(Fore.BLUE, "得分评定成功，该模型得分为：{}".format(scor))
            Prediect_Y = model.predict(Test_X)
            print(Fore.CYAN, "正在画图，请等待...")
            #################画图##########################################
            plot_x = np.arange(0, 1000, 1000 / len(Prediect_Y))
            plt.scatter(plot_x, Prediect_Y + 0.02, marker='*', label='prediect')
            plt.scatter(plot_x, Test_Y, marker='.', label='source')
            plt.xlabel('number')
            plt.ylabel('value')
            plt.legend()
            plt.title("SVM(linear)")
            plt.show()
            print(Fore.LIGHTRED_EX, "本次训练结束")
            ################################################################
            return model
    def User(self,sample=[0.697,0.456]):
        if sample==None:
            print(Fore.RED,"请输入测试样本!!!")
        else:
            model=SupportVectorMachines.Trainer(self)
            result=model.predict([sample])
            if int(result[0])=='0':
                print(Fore.RED,"这是一个坏瓜")
            else:
                print(Fore.GREEN,"这是一个好瓜")



if __name__ == '__main__':
    A=SupportVectorMachines(mode='Linear')
    A.User()
    B=SupportVectorMachines(mode='rbf')
    B.User()