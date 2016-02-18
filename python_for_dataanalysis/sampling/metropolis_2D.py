#!/usr/bin/python
# coding: utf-8
# 2016/2/7
# メトロポリス法で2Dガウシアンのサンプリング
import numpy as np
import numpy.random as rnd
import pylab as plt
import matplotlib.animation as animation

# set up the distinguished line


def mnd(_x, _mu, _sigma):
    x = np.matrix(_x)
    mu = np.matrix(_mu)
    sigma = np.matrix(_sigma)
    a = np.sqrt(np.linalg.det(sigma)*(2*np.pi)**sigma.ndim)
    b = np.linalg.det(-0.5*(x-mu)*sigma.I*(x-mu).T)
    return np.exp(b)/a
"""
初期設定
"""
# gaussianの初期値
mean = [0.0, 0.0]
cov = [[1.0, 0.5], [1.5, 3.0]]
xmin, ymin, xmax, ymax = -10, -10, 10, 10
T_max = 500
n = 3.0  # 更新幅
plt.xlim = (xmin, xmax)
plt.ylim = (ymin, ymax)
num = 0

# for animation
list = []
figure = plt.figure()


if __name__ == '__main__':
    # x_now = [-xmax/2+rnd.random()*xmax, -ymax/2+rnd.random()*ymax]
    x_now = [5,5]
    x_new = [0, 0]
    for i in range(T_max):
        q = rnd.random()
        tmp1 = x_now[0]
        tmp2 = x_now[1]
        x_new[0] = x_now[0]+n*rnd.random()-n/2.0
        x_new[1] = x_now[1]+n*rnd.random()-n/2.0
        # print x_now,x_new
        # print mnd(x_new, mean, cov), mnd(x_now, mean, cov), mnd(x_new, mean, cov)/mnd(x_now, mean, cov)
        # 棄却 or not

        if(q < mnd(x_new, mean, cov)/mnd(x_now, mean, cov)):
            x_now[0] = x_new[0]
            x_now[1] = x_new[1]
            plt.plot(x_now[0], x_now[1], 'bo',alpha = 0.1)
            frame = plt.plot(np.array([tmp1,x_new[0]]), np.array([tmp2,x_new[1]]), '-k', linewidth = 3)
            list.append(frame)
            print i
            num += 1
    print num*1.0/T_max
    ani = animation.ArtistAnimation(figure,list)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('metropolis.mp4', writer=writer)

    plt.show()
"""
x_now = x_new はダメ？
ポインタとして認識する？？
"""
