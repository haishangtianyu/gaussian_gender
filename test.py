# -*- coding: utf-8 -*-

from gaussian_gender import main_
import matplotlib.pyplot as plt 
import numpy as np 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

#记得在函数后加括号
accuracy_rate = []
times = 20
for i in range (times):
    accuracy_rate.append(main_())
a = np.arange(20)
x = np.array(accuracy_rate)
plt.plot(a,x[:,0],label=u'男声正确率')
plt.plot(a,x[:,1],label=u'女声正确率')
plt.title(u'声音性别识别正确率与性别的关系')
plt.xlabel(u'次数')
plt.ylabel(u'正确率')
plt.axis([0,20,0,1])
plt.legend(loc='best')
plt.show()
male_accu=x[:,0].mean()
female_accu=x[:,1].mean()
print('------------------20轮平均数据------------------')
print('男声平均正确率：%.2f%%   女生平均正确率：%.2f%%'%(male_accu*100,female_accu*100))
print('男声平均错误率：%.2f%%   女生平均错误率：%.2f%%'%(100-male_accu*100,100-female_accu*100))