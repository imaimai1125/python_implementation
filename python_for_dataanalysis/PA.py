
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

N = 500
eps = 0.01 #sensitivity
T_max = 100000

if __name__ == "__main__":

	#generate taining data
	cls1 = []
	cls2 = []
	t1 = []
	t2 = []

	#set up the distinguished line
	def h(x):
		return -x+5
	#generate date following gaussian
	mean = [2.5, 2.5]
	cov = [[1.0,0.0],[0.0,1.0]] #cov matrix (equall in all classes)
	#generate the random data (2D)
	Rand = np.random.multivariate_normal(mean,cov,N)
	for i in range(N):
		if h(Rand[i][0])<Rand[i][1]:
			cls1.append(Rand[i])
			t1.append(+1) #class1
		else:
			cls2.append(Rand[i])
			t2.append(-1) #class2
	#class1
	x1,x2 = np.transpose(np.array(cls1))
	plt.plot(x1,x2,'bo')
	#class2
	x1, x2 = np.transpose(np.array(cls2))
	plt.plot(x1,x2,'ro')
	#distinctive line
	x = linspace(0,5.0,50)
	y = h(x)
	plt.plot(x,y,'g-')


##################Passive aggressive Algorithm##########################
	#merge the data in class 1 and class 2
	x1, x2 = np.array(cls1+cls2).transpose()
	t = np.array(t1+t2).transpose()
	print t
	w = np.array([1.0, 1.0, 5.0]) #initial condition
	turn = 0
	correct = 0 #data classified appropriately
	#############Algorithm###############
	while correct < N: #continue until it can classify all data appropriately.
		correct = 0
		for i in range(N): #consider all data
			v = np.array([1,x1[i],x2[i]])*t[i]
			l = eps - np.dot(w,[1,x1[i],x2[i]])*t[i]
			if np.dot(w,[1,x1[i],x2[i]])*t[i] >= 0 :#do nothing if classified well
				correct += 1
			else: #if wrong, adjust the parameters
				w += (l/(np.dot(v,v)))*v
			turn += 1
		print turn, w, correct
		x = linspace(0,5.0,50)
		#\draw the distinctive line
		y = -w[1]/w[2]*x - w[0]/w[2]
		plt.plot (x,y,'-y')
		if turn>T_max:
			break
			
	x = linspace(0,5.0,50)
	#draw the distinctive line
	y = -w[1]/w[2]*x - w[0]/w[2]
	plt.plot (x,y,'-b')
	xlim(0,5)
	ylim(0,5)
	show()