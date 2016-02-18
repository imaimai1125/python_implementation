#!/usr/bin/python
# coding: utf-8
# メトロポリス法で1Dガウシアンのサンプリング
import numpy as np
import numpy.random as rnd
import pylab as plt

# 式（ガウス分布）


def g(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi*sigma))*np.exp(-((x-mu)**2)/(2*sigma**2))
    # return 1/(np.sqrt(2*np.pi*sigma))*np.exp(-((x-mu-2)**2)/(2*sigma**2))
    # +1/(np.sqrt(2*np.pi*sigma))*np.exp(-((x-mu+2)**2)/(2*sigma**2))

"""
初期設定
"""

mu = 0
sigma = 1.0
xmin, xmax = -10, 10
ymin, ymax = 0, 3
x_now = -5+rnd.random()*10
# ヒストグラム用配列
hist_num = 100
T_max = 100000

if __name__ == '__main__':
    """
    gaussianのプロット
    """
    x = np.linspace(-10, 10, 200)
    plt.plot(x, g(x, mu, sigma), "b")
    g_max = max(g(x,mu,sigma))
    plt.xlim = (xmin, xmax)
    plt.ylim = (ymin, ymax)
    """
	sampling(metropolis)
	"""
    # 一様分布からサンプリング(montecarlo)
    histogram = np.zeros(hist_num)
    m = hist_num/(xmax-xmin)
    # 値を更新（確率的に選んだ幅）
    for i in range(T_max):
        x_new = x_now + m*rnd.random()-m/2.0
        q = rnd.random()
        if (q<g(x_new, mu, sigma)/g(x_now, mu, sigma)):
            x_now = x_new
        histogram[int(x_now*m+hist_num)-hist_num/2] += 1
        print i, x_now, x_new  # ,int(x_now*m)+hist_num/2

        # sum = 0
        # for i in range(len(histogram)):
        #     sum += histogram[i]
        # print sum
    n = np.linspace(-10, 10, hist_num)
    # 正規化
    hist_normal = histogram/max(histogram)*g_max
    print hist_normal
    plt.plot(n, hist_normal, 'go')
    plt.show()
    # plt.show()
