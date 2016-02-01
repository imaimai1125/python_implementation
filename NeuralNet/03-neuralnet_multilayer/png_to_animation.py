#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2016/2/1
吐き出した画像ファイルを集めてアニメーションにする
http://d.hatena.ne.jp/white_wheels/20100322/p1
http://jn1inl.blog77.fc2.com/blog-entry-2120.html
http://www.waw-project.com/wordpress/2014/01/31/opencvを使ってカメラから動画として保存/
"""
# import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import os

epoch_range = 50
fps = 30

img_init = cv2.imread("epoch_50.png")
height, width, layers = img_init.shape
video = cv2.VideoWriter("animation.avi",
						cv2.cv.CV_FOURCC('D','I','V','X'),
						fps,
						(width,height))
print height,width,layers
for i in range(1,101):
    text = "epoch_" + str(i * epoch_range) + ".png"
    print text
    img = cv2.imread(text)
    # cv2.imshow(text,img)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()
    video.write(img)

cv2.destroyAllWindows()
video.release()

