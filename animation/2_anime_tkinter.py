# coding: utf-8
# http://qiita.com/nnahito/items/ad1428a30738b3d93762
# http://qiita.com/nnahito/items/2ab3ad0f3adacc3314e6

import sys
import Tkinter
"""
ソフトウェアの実行内容の処理は
root = Tkinter.Tk()と
root.mainloop()
の間に収める
"""
### Tkinterしますよ
root = Tkinter.Tk()
### ウィンドウのタイトルバー
root.title(u"Software Title")
### ウィンドウサイズ
root.geometry("500x500")
### ラベルを作って表示
Static = Tkinter.Label(text=u'test', foreground='#ff0000')
# Static.pack()

### roopしますよ
root.mainloop()
