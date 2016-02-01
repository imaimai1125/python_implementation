# coding: utf-8
# matplotlibを使ったアニメーション
# ガウス分布を右にずらす

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
# ガウス分布
def gaussian(x,mu,sigma):
	return 1/(np.sqrt(2*np.pi*sigma))*np.exp(-((x-mu)**2)/(2*sigma**2))	
# 座標のmax,min設定
xmin = -10
xmax = 10
x = np.linspace(xmin,xmax,200)
# 範囲
plt.xlim(xmin,xmax);


# アニメーションの設定
list = []
for i in range(0,20):
	y = gaussian(x,-5+i/2.0,1)
	frame = plt.plot(x,y,color = "g")
	# いちいちフレームに結果を入れておく 
	list.append(frame)
#最終的にフレームをパラパラ漫画のように
ani = animation.ArtistAnimation(fig, list)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save('gaussian.mp4', writer=writer)
# plt.show()