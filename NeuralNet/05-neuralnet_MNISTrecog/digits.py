#coding: utf-8


import numpy as np
import pylab as plt
import cPickle,gzip
class Image:
	##MNISTのデータを使って数字認識をする(28*28)
	##DL:http://deeplearning.net/tutorial/gettingstarted.html
	##生データはパースがめんどいからすでに整備されたデータを使う。
	def __init__(self,a):
		# Load the dataset
		f = gzip.open('mnist.pkl.gz', 'rb')
		self.train_set, self.valid_set, self.test_set = cPickle.load(f)
		f.close()
		#横一列分の画素数（一次元配列を再び二次元の配列に整形しなおすときに用いる）
		self.n = 28
		self.ntrain = len(self.train_set[1])
		self.ntest  = len(self.test_set[1])
		print self.ntrain, self.ntest
	def makedata(self): 
		#画像をそれぞれ格納
		self.train_images = self.train_set[0]
		self.test_images = self.test_set[0]
		#ラベルを格納
		#ターゲットデータを一次元配列（10次元）に変換
		self.train_labels = np.zeros((self.ntrain, 10))
		self.test_labels  = np.zeros((self.ntest, 10))
		#1 to K 記法
		for i in range (self.ntrain):
			self.train_labels[i][self.train_set[1][i]] = 1
		for i in range (self.ntest):
			self.test_labels[i][self.test_set[1][i]] = 1
		return self.train_images, self.train_labels, self.test_images, self.test_labels

	####ミスしたイメージを抽出・可視化
	def extract_mistaken_image(self,X,t,prediction,miss_list):
		miss_list = np.array(miss_list)
		#ミスをしているやつだけ書き出す。
		for i in miss_list:
			#画像情報の直し（１Dから２Dに再度戻す）
			image = np.hsplit(X[i],self.n)
		    #画像の並べ方を指定(縦列，横列，サブプロットの場所)
			plt.subplot(len(miss_list)/5+1, 5,np.where(miss_list==i)[0] +1)
		    #軸（off)
			plt.axis('off')
		    #画像表示（interpolationとは？）
			plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
		    #タイトル（どう間違ったか)
			plt.title("#%d miss:%d correct:%d" %(i,np.argmax(prediction[i]), np.argmax(t[i])))
		plt.show()
	

if __name__=='__main__':
	image = Image(0)
	print image.makedata()[3][1]
	#print np.hsplit(image.makedata(10)[0][1],len(image.images[0]))
