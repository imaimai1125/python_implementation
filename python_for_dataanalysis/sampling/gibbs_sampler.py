#!/usr/bin/python
# coding: utf-8
# 2016/2/7
# gibbs sampling で2Dガウシアンのサンプリング
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
cov = [[1.0, 0.5], [0.5, 1.0]]
xmin, ymin, xmax, ymax = -10, -10, 10, 10
T_max = 300
plt.xlim = (xmin, xmax)
plt.ylim = (ymin, ymax)
num = 0
# for animation
list = []
figure = plt.figure()
tmp1 = 0
tmp2 = 0

if __name__ == '__main__':
    x_new = [5, 5]

    for i in range(T_max):
        tmp1 = rnd.normal((cov[0][1]+cov[1][0]) /
                          (2*cov[0][0])*tmp2, 1/np.sqrt(cov[0][0]))
        plt.plot(tmp1, tmp2, 'bo', alpha=0.1)
        frame = plt.plot(
            np.array([tmp1, x_new[0]]), np.array([tmp2, tmp2]), '-b', linewidth=3)
        list.append(frame)

        tmp2 = rnd.normal((cov[0][1]+cov[1][0]) /
                          (2*cov[1][1])*tmp1, 1/np.sqrt(cov[1][1]))
        frame = plt.plot(
            np.array([tmp1, tmp1]), np.array([tmp2, x_new[1]]), '-g', linewidth=3)
        list.append(frame)
        plt.plot(tmp1, tmp2, 'go', alpha=0.1)
        x_new = [tmp1, tmp2]
        print i, x_new
    ani = animation.ArtistAnimation(figure, list)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('gibbs.mp4', writer=writer)
    plt.show()
