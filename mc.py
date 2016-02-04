#!/usr/bin/python
# coding: utf-8
# モンテカルロ法で円周率出す
import numpy as np
import numpy.random as rnd
import pylab as plt

err = 0.001
t = 1.0
p = 1.0
while (np.abs(np.pi-p/t*4.0)>err):
	x = rnd.random()-0.5
	y = rnd.random()-0.5
	# 円内だけプロット
	if(x*x+y*y<=1.0/4.0):
		p+=1
		plt.plot(x,y,'bo')
	t+=1
	if(t % 1000 == 0):
		print t,(p/t)*4.0
print t,p/t*4.0

# plot
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.show()