#coding: utf-8
import numpy as np
import pylab as plt
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer


class Image:
	# scikit-learnの手書き数字データをロード
	# 1797サンプル、8x8ピクセル
	def __init__(self,a):
		self.digits = load_digits()
		self.N = len(self.digits.target)
		# 扱いやすいようにnumpy列に変換
		self.images = np.array(self.digits.images)
		self.target = np.array(self.digits.target)		
	def makedata(self,split_rate):
		# 二次元配列（8*8）画像データを1次元配列（64次元）に変換（[0,0]→[0,8]→[1,0]...の順）
		self.images1D = np.ndarray((self.N,(len(self.images[0])*len(self.images[0][0]))))
		for i in range (self.N):
			array1D = self.images[i][0] 
			for j in range(len(self.images[0])-1):
				array1D = self.array2D_to_array1D(array1D,self.images[i][j+1])
			self.images1D[i] = array1D
		#　二次元配列を正規化する。
		self.images1D /= 255
		#print self.images1D[1]
		###print self.images1D.shape
		# ターゲットデータを一次元配列（10次元）に変換
		self.target1D = np.zeros((self.N, 10))
		for i in range (self.N):
			self.target1D[i][self.target[i]] = 1

		# 教師データと訓練データに分類
		self.num_of_traindata = self.N - self.N/split_rate
		self.train_images, self.test_images = np.vsplit(self.images1D,[self.num_of_traindata])
		self.train_labels, self.test_labels = np.vsplit(self.target1D,[self.num_of_traindata])

		
		return self.train_images, self.train_labels, self.test_images, self.test_labels
	# 2次元配列を1次元に統合、正規化(List型のやつ)
	def array2D_to_array1D(self,x,y):
		return np.hstack([x,y])

	####ミスしたイメージを抽出・可視化
	def extract_mistaken_image(self,X,t,prediction,miss_list):
		miss_list = np.array(miss_list)
		#ミスをしているやつだけ書き出す。
		for i in miss_list:
			#画像情報の直し（１Dから２Dに再度戻す）
			image = np.hsplit(X[i],len(self.images[0]))
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
	#print image.makedata(10)[0][1]
	#print np.hsplit(image.makedata(10)[0][1],len(image.images[0]))
	mnist = fetch_mldata('MNIST orginal')