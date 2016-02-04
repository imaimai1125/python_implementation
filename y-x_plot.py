#!/usr/bin/python
# coding: utf-8
# 20151105
import numpy as np
import pylab as plt


##式
def f(x):
	# return np.log(1+np.exp(x)) #ReLU	
	# return np.sin(x)*np.exp(-x/2) #e^(-1/2)sinx
	# return np.abs(x)*np.sin(x)
	return np.max(x,0)
##式（ガウス分布）
def gaussian(x,mu,sigma):
	return 1/(np.sqrt(2*np.pi*sigma))*np.exp(-((x-mu)**2)/(2*sigma**2))	
##微分
def differential(x,y,i):
	return (y[i+1]-y[i])/(x[i+1]-x[i])

##x座標の範囲
xmin = -10
xmax = 10
x = np.linspace(xmin,xmax,200)
##ベースライン
y0 = np.linspace(0,0,200)

# 範囲の設定
plt.xlim(xmin,xmax)

##ｙ座標の値
# y = gaussian(x,2,1)+gaussian(x,-2,1)
y= f(x)
##微分値
dy0 = []
for i in range(len(x)-1):
	dy0.append(differential(x,y,i))
dy0.append(dy0[98])

##実際にプロット
plt.plot(x,y0,color = "g")
plt.plot(x,y,color = "b")
plt.plot(x,dy0,color = "r")

plt.show()