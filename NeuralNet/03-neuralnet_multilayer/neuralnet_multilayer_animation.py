#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author imai takeaki
#date 2nd, Feb, 2016
#アニメーション

import numpy as np
from numpy.random import *
import pylab as plt
import matplotlib.animation as ArtistAnimation
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


class NN:
    ###隠れ層の数を2層に
    def __init__(self, num_input, num_hidden, num_hidden2, num_output, learning_rate):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_hidden2 = num_hidden2
        self.num_output = num_output
        self.learning_rate = learning_rate

        self.w_input_to_hidden = np.random.random((self.num_hidden, self.num_input))
        self.w_hidden_to_hidden2 = np.random.random((self.num_hidden2, self.num_hidden))
        self.w_hidden2_to_output = np.random.random((self.num_output, self.num_hidden2))

        self.b_input_to_hidden = np.ones((self.num_hidden))
        self.b_hidden_to_hidden2 = np.ones((self.num_hidden2))
        self.b_hidden2_to_output = np.ones((self.num_output))

    ##活性化関数（シグモイド関数）
    def activate_func(self, x):
        return 1/(1+np.exp(-x))
        # return max(0,x)
    ##活性化関数の微分
    def dactivate_func(self,x):
        return self.activate_func(x)*(1-self.activate_func(x))
        #if x>0:return 1
        #else : return 0
    ##ソフトマックス関数
    def softmax_func(self,x):
        s = 0.0
        t = np.exp(x)
        for i in range(len(x)):
            s += np.exp(x[i])
        return t/s
    ##順伝播計算_
    def forward_propagation(self, x):
        u_hidden = np.dot(self.w_input_to_hidden, x) + self.b_input_to_hidden
        z_hidden = self.activate_func(u_hidden)
        u_hidden2 = np.dot(self.w_hidden_to_hidden2, z_hidden) + self.b_hidden_to_hidden2
        z_hidden2 = self.activate_func(u_hidden2)
        u_output = np.dot(self.w_hidden2_to_output, z_hidden2) + self.b_hidden2_to_output
        z_output = self.softmax_func(u_output)
        return u_hidden, u_hidden2, u_output, z_hidden, z_hidden2, z_output
    ##逆伝播でδを求める
    def backward_propagation(self,t,fp):
        delta_output = fp[5]-t
        delta_hidden2 = np.dot(delta_output, self.w_hidden2_to_output * self.dactivate_func(fp[1]))
        delta_hidden = np.dot(delta_hidden2, self.w_hidden_to_hidden2 * self.dactivate_func(fp[0]))
        return delta_hidden, delta_hidden2, delta_output
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
    list = [] #アニメーション用のリスト
    nn = NN(2,10,10,5,0.2)
    epoch = 0
    while 1:
        grad_i_to_h = 0
        grad_h_to_h2 = 0
        grad_h2_to_o = 0
        gradbias_i_to_h = 0
        gradbias_h_to_h2 = 0
        gradbias_h2_to_o = 0
        n = 0
        err = 0
        rand = randint(0,len(X),100)
        for i in range(len(rand)):
            fp = nn.forward_propagation(X[rand[i]])
            bp = nn.backward_propagation(t[rand[i]], fp)
            grad_i_to_h  += nn.calc_gradient(bp[0], X[rand[i]])
            grad_h_to_h2 += nn.calc_gradient(bp[1], fp[3])
            grad_h2_to_o += nn.calc_gradient(bp[2], fp[4])
            gradbias_i_to_h += bp[0]
            gradbias_h_to_h2 += bp[1]
            gradbias_h2_to_o += bp[2]
        nn.w_input_to_hidden = nn.update_weight(nn.w_input_to_hidden, grad_i_to_h / len(rand))
        nn.w_hidden_to_hidden2 = nn.update_weight(nn.w_hidden_to_hidden2, grad_h_to_h2 / len(rand))
        nn.w_hidden2_to_output = nn.update_weight(nn.w_hidden2_to_output, grad_h2_to_o / len(rand))
        nn.b_input_to_hidden = nn.update_weight(nn.b_input_to_hidden, gradbias_i_to_h / len(rand))
        nn.b_hidden_to_hidden2 = nn.update_weight(nn.b_hidden_to_hidden2, gradbias_h_to_h2 / len(rand))
        nn.b_hidden2_to_output = nn.update_weight(nn.b_hidden2_to_output, gradbias_h2_to_o / len(rand))

        #print grad_h2o
        epoch += 1
        if epoch%50 == 0:
            print epoch
            
            #####学習したパラメータで境界線を図示
            print "plotting...%d"%(epoch)
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
            for i in range(80):
                for j in range(80):
                    INPUT = [i*2.0/10-8.0,j*2.0/10-8.0]
                    a = nn.forward_propagation(INPUT)
                    if np.argmax(a[5]) == 0:
                        plt.plot(i*2.0/10-8.0,j*2.0/10-8.0,'rx')
                    elif np.argmax(a[5]) == 1:  
                        plt.plot(i*2.0/10-8.0,j*2.0/10-8.0,'gx')
                    elif np.argmax(a[5]) == 2:
                        plt.plot(i*2.0/10-8.0,j*2.0/10-8.0,'bx')
                    elif np.argmax(a[5]) == 3:
                        plt.plot(i*2.0/10-8.0,j*2.0/10-8.0,'yx')
                    elif np.argmax(a[5]) == 4:
                        plt.plot(i*2.0/10-8.0,j*2.0/10-8.0,'cx')
            #frameに
            list.append(plt)
            plt.cla()
        if epoch > 5000:
            break
    fig = plt.figure()
    ani = animation.ArtistAnimation(fig,list)
    ani.save('result.mp4', writer=writer)
    plt.show()
    print "finish plotting"