#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author imai takeaki
#date Aug 26th
#scikit learnに使われているdigitをベクトルデータに変換した後にアレする
#層は一層
import numpy as np
from numpy.random import *
import pylab as plt
import digits
#fetch the training data
image = digits.Image(0) #initialize
data = image.makedata(10)
##訓練データ
X = data[0]
t = data[1]
##テストデータ
X_test = data[2]
t_test = data[3]

class NN:
    def __init__(self, num_input, num_hidden, num_output, learning_rate):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.learning_rate = learning_rate

        self.w_input2hidden = (np.random.random((self.num_hidden, self.num_input))-0.5)
        self.w_hidden2output = (np.random.random((self.num_output, self.num_hidden))-0.5)
        self.b_input2hidden = np.zeros((self.num_hidden))
        self.b_hidden2output = np.zeros((self.num_output))

    ##活性化関数（シグモイド関数）
    def activate_func(self, x):
        return 1/(1+np.exp(-x))
    ##活性化関数の微分
    def dactivate_func(self,x):
        return self.activate_func(x)*(1-self.activate_func(x))
    ##ソフトマックス関数
    def softmax_func(self,x):
        s = 0.0
        t = np.exp(x)
        for i in range(len(x)):
            s += np.exp(x[i])
        return t/s
    ##順伝播計算_
    def forward_propagation(self, x):
        u_hidden = np.dot(self.w_input2hidden, x) + self.b_input2hidden
        z_hidden = self.activate_func(u_hidden)
        u_output = np.dot(self.w_hidden2output, z_hidden) + self.b_hidden2output
        z_output = self.softmax_func(u_output)
        return u_hidden, u_output, z_hidden, z_output
    ##逆伝播でδを求める
    def backward_propagation(self,t,u):
        delta_output =  u[3]-t
        delta_hidden = np.dot(delta_output, self.w_hidden2output * self.dactivate_func(u[0]))
        return delta_hidden, delta_output
    ##損失関数のパラメータwに関する勾配
    def calc_gradient(self,delta,z):
        dW = np.zeros((len(delta), len(z)))
        for i in range(len(delta)):
            for j in range(len(z)):
                dW[i][j] = delta[i] * z[j]
        return dW
    ##重みをアップデートする
    def update_weight(self,w0,gradE):
        return w0 - self.learning_rate*gradE

if __name__ == '__main__':
    nn = NN(len(X[0]),32,len(t[0]),0.1)
    epoch = 0
    while 1:
        grad_i2h = 0
        grad_h2o = 0
        gradbias_i2h = 0
        gradbias_h2o = 0
        n = 0
        err = 0
        rand = randint(0,len(X),10)
        for i in range(len(rand)):
            a = nn.forward_propagation(X[rand[i]])
            b = nn.backward_propagation(t[rand[i]], a)
            grad_i2h += nn.calc_gradient(b[0], X[rand[i]])
            grad_h2o += nn.calc_gradient(b[1], a[2])
            gradbias_i2h += b[0]
            gradbias_h2o += b[1]

        nn.w_input2hidden = nn.update_weight(nn.w_input2hidden, grad_i2h / len(rand))
        nn.w_hidden2output = nn.update_weight(nn.w_hidden2output, grad_h2o / len(rand))
        nn.b_input2hidden = nn.update_weight(nn.b_input2hidden, gradbias_i2h / len(rand))
        nn.b_hidden2output = nn.update_weight(nn.b_hidden2output, gradbias_h2o / len(rand))

        epoch += 1
        if epoch%100 == 0:
            print "training... %d"%(epoch)
            #####学習したパラメータでテスト####
            tmp = 0.0
            for i in range(len(X_test)):
                fp = nn.forward_propagation(X_test[i])
                if np.argmax(fp[3]) == np.argmax(t_test[i]):
                    tmp += 1
            accuracy = tmp/len(X_test)*100.0
            plt.plot(epoch/100, accuracy, "bx")
            print "testing...accuracy:%f"%(accuracy)
        if epoch > 50000:
            break
    plt.show()
    #####学習したパラメータでテスト(まちがった数字を表示）####
    miss_list = []
    prediction=[]
    print "Extracting the mistaken figures..."
    for i in range(len(X_test)):
        fp = nn.forward_propagation(X_test[i])
        prediction.append(fp[3])
        if np.argmax(fp[3]) != np.argmax(t_test[i]):
            miss_list.append(i)
            print "%d is mistakenly classified %d as %d" %(i,np.argmax(fp[3]), np.argmax(t_test[i]))
            ###ミスした画像とどうミスしたかを表示
    image.extract_mistaken_image(X_test,t_test,prediction,miss_list)
   # print miss_list
    print "final mistakes = %d within %d data" %(len(miss_list),len(X_test))   
    
    
