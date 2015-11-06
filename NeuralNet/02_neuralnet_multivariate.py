#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author imai takeaki
#出力がベクトル（１次元じゃない）風にNNを拡張
#出力層の最後にソフトマックス関数を使うのがポイントっぽい

import numpy as np
from numpy.random import *
import pylab as plt
#create the training data
N = 200
cls1 = []
cls2 = []
cls3 = []
cls4 = []
cls5 = []

mean1 = [3, 3]
mean2 = [3, -3]
mean3 = [-3, 3]
mean4 = [-3, -3]
mean5 = [0,0]
#common covariance matrix
cov = [[0.5, 0.0],[0.0,0.5]]
#random data following the mean and cov
cls1.extend(np.random.multivariate_normal(mean1,cov,N/5))
cls2.extend(np.random.multivariate_normal(mean4,cov,N/5))
cls3.extend(np.random.multivariate_normal(mean2,cov,N/5))
cls4.extend(np.random.multivariate_normal(mean3,cov,N/5))
cls5.extend(np.random.multivariate_normal(mean5,cov,N/5))


#data matrix
X = np.vstack((cls1,cls2,cls3,cls4,cls5))
#print X, len(X)
#label t
#３クラスで分類してみたい．
#200*3の行列を作っている．
t = np.zeros((N,5))
for i in range(N/5):
    t[i][0] = 1
for i in range(N/5,N/5*2):
    t[i][1] = 1
for i in range(N/5*2,N/5*3):
    t[i][2] = 1
for i in range(N/5*3,N/5*4):
    t[i][3] = 1
for i in range(N/5*4,N):
    t[i][4] = 1

#paint the training data
x1, x2 = np.array(cls1).transpose()
plt.plot(x1,x2,'ro')
x1, x2 = np.array(cls2).transpose()
plt.plot(x1,x2,'go')
x1, x2 = np.array(cls3).transpose()
plt.plot(x1,x2,'bo')
x1, x2 = np.array(cls4).transpose()
plt.plot(x1,x2,'yo')
x1, x2 = np.array(cls5).transpose()
plt.plot(x1,x2,'co')


class NN:
    def __init__(self, num_input, num_hidden, num_output, learning_rate):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.learning_rate = learning_rate

        self.w_input2hidden = np.random.random((self.num_hidden, self.num_input))
        self.w_hidden2output = np.random.random((self.num_output, self.num_hidden))
        self.b_input2hidden = np.ones((self.num_hidden))
        self.b_hidden2output = np.ones((self.num_output))

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
        delta_output = u[3]-t
        delta_hidden = np.dot(delta_output, self.w_hidden2output * self.dactivate_func(u[0]))
        #print delta_output, delta_hidden
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
    nn = NN(2,20,5,0.2)
    epoch = 0
    while 1:
        grad_i2h = 0
        grad_h2o = 0
        gradbias_i2h = 0
        gradbias_h2o = 0
        n = 0
        err = 0
        rand = randint(0,len(X),100)
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
        #print grad_h2o
        epoch += 1
        if epoch%100 == 0:
            print epoch
        if epoch > 1500:
            break
    print "plotting..."
    for i in range(160):
        for j in range(160):
            INPUT = [i*1.0/10-8.0,j*1.0/10-8.0]
            a = nn.forward_propagation(INPUT)
            if np.argmax(a[3]) == 0:
                plt.plot(i*1.0/10-8.0,j*1.0/10-8.0,'rx')
            elif np.argmax(a[3]) == 1:  
                plt.plot(i*1.0/10-8.0,j*1.0/10-8.0,'gx')
            elif np.argmax(a[3]) == 2:
                plt.plot(i*1.0/10-8.0,j*1.0/10-8.0,'bx')
            elif np.argmax(a[3]) == 3:
                plt.plot(i*1.0/10-8.0,j*1.0/10-8.0,'yx')
            elif np.argmax(a[3]) == 4:
                plt.plot(i*1.0/10-8.0,j*1.0/10-8.0,'cx')
        print i
    
    plt.show()
    print "finish plotting"