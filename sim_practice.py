#!/usr/bin/python
# coding: utf-8

from __future__ import unicode_literals, division

import simpy
import numpy as np
np.random.seed(1)

#1.シミュレーション環境を作る
env = simpy.Environment()
"""
env.process(f) プロセスfを追加する
env.run(until = 15) 15回動作実行
"""
def car(env):
	velocity = 72/3600 * 1000
	location = 0.0
	while True:
		if(env.now%300 ==0):
			print ("time:{0:2d} location:{1}m, velocity:{2}km/h".format(env.now, location,velocity*3.6))
		location += velocity
		#ランダムな速度
		velocity = np.random.normal(72,10)/3600*1000 
		#次にプロセスを実行するまでのタイムアウト
		yield env.timeout(1)

#2. シミュレーション対象のプロセス(generator)を作成
car(env)
#3. シミュレーション環境にプロセスを追加
env.process(car(env))
#4. シミュレーション環境の実行
env.run(until = 3601)

