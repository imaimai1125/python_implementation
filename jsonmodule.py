#!/usr/bin/python
# coding: utf-8

## 参考　http://pod.hatenablog.com/entry/2015/12/03/092403
## jsonモジュール

"""
jsonモジュールはJSON形式の文字列表現とpython object間のserialization・deserializationを担うライブラリです。通常、pythonユーザーたちの間であるモジュールがdeserialization・serializationの機能を持つモジュールだと言われた場合には、そのモジュールが以下のようなインターフェイスを持つことを想定していることが多い気がします。

deserialization -- ある表現からpython objectへの変換。 load() , loads() が存在
serialization -- python objectからある表現への変換。 dump() , dumps() が存在


JSONとpython objectの関係
JSON	Python
object	dict
array	list
string	str
number	(int)|int
number	(real)|float
true	True
false	False
null	None


"""


print 'hello'

from collections import Counter
import random
import json

class RandomItemDict(dict):
	def items(self):
		return iter(sorted(super(RandomItemDict,self).items(),key=lambda x : random.random()))

def make_data():
	return RandomItemDict({
		"person":RandomItemDict({"name":"foo","age":20}),
		"address":RandomItemDict({"foo":"bar"}),
	})


c = Counter()
for _ in range(1000):
	data = make_data()
	c[json.dumps(data, sort_keys=True)] += 1
print(c.values())

#dict_values([231,267,241,261])
