# -*- coding: utf-8 -*-
####下面的程序只是在训练感知器，并没有真正的返回分类结果####
'''
1.权重函数的计算，通过计算每一个输入样本的计算值和实际值的比较来计算权重。
2.引入损失函数：J或者代价函数：C（其实都是一个东西，叫法不同而已）
2.1 批量梯度下降法：最小化所有训练样本的损失函数，使得最终的求解是全局的最优解（使得风险函数最小）
2.2 随机提督下降法：最小化每一个样本的损失函数，每次的计算得到的损失函数不一定是最优解，但是整体方向是最优方向
    得到的结果往往是全局最优解的附近。
'''
import numpy as np
class Perceptron(object):

    def __init__(self,eta=0.01,epochs=50):#这里是变量的默认值，是可以修改的
        self.eta = eta#实际上这里的eta的值也是可以更改的
        self.epochs = epochs

    def train(self,X,y):
        self.w_=np.zeros(1+X.shape[1])#这里+1的原因是w*x+w0来表示完整的条件表达式，这里是获取列数
        self.errors_=[]
        for i in range(self.epochs):
            errors=0
            for xi,target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors+=int(update!=0.0)
            self.errors_.append(errors)
        return self #很明显这里返回的是训练的感知器
    def net_input(self,X):
        '''
        这里的计算实际上是带入相应的w值计算对应的 w*x+w0 得到的最后的结果，因为后面还要和实际上Y值进行比较
        '''
        return np.dot(X,self.w_[1:])+self.w_[0]#因为这里计算的是属性（列）的权重，所以这这样的乘法

    def predict(self,X):
        '''
        这里表示的意思是前面计算后得到的结果若是大于0的全部等于1，若是小于1的全部为-1，
        因为前面我们会定义标准是当某个条件为满足后为1不满足的时候就为-1
        '''
        return np.where(self.net_input(X)>=0.0,1,-1)

#########导入数据源中的数据
import pandas as pd
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#在线获取鸢尾花数据集
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)#很显然这里对应的是前面的设置的目标的
X=df.iloc[0:100,[0,2]].values

import matplotlib.pyplot as plt
from mlxtend.evaluate import plot_decision_regions

ppn=Perceptron(epochs=10,eta=0.1)

ppn.train(X,y)
print 'Weight:%s',ppn.w_
plot_decision_regions(X,y,clf=ppn)
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('penal length [cm]')
plt.show()

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='+')
plt.xlabel('Iterations')
plt.ylabel('Missclassifications')
plt.show()

########################################
y2=df.iloc[50:150,4].values
y2=np.where(y2=='Iris-setosa',-1,1)
X2=df.iloc[50:150,[1,3]].values
ppn=Perceptron(epochs=25,eta=0.01)
ppn.train(X2,y2)
plot_decision_regions(X2,y2,clf=ppn)
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('penal length [cm]')
plt.show()

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='+')
plt.xlabel('Iterations')
plt.ylabel('Missclassifications')
plt.show()

########################################
import numpy as np

class AdalineGD(object):

    def __init__(self,eta=0.01,epochs=50):
        self.eta=eta
        self.epochs=epochs

    def train(self,X,y):
        self.w_=np.zeros(1+X.shape[1])#这里需要解释一下
        '''我们一般情况下都是使用列来表示所谓的属性，使用行来表示个数，
        所以后面我们的感知器的计算都是通过计算属性的加权'''
        self.cost_=[]
        '''权值的更新是通过计算数据集中所有的样本（而不是随着每个样本的增加而更新），
        这也是此方法又被称作“批量”梯度下降。注意这里不是随机梯度下降'''
        for i in range(self.epochs):#这里的计算方法和前面的有所不同，前面是通过嵌套两个循环来计算权重的值
            output=self.net_input(X)
            errors=(y-output)
            self.w_[1:]+=self.eta*X.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2.0
            self.cost_.append(cost)
        return self#返回的是整个类，我们后面在调用其中输入参数后的结果

    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]

    def activation(self,X):#不明白为什么要定义另个函数
        return self.net_input(X)

    def predict(self,X):
        return np.where(self.activation(X)>=0.0,1,-1)

#学习速率较大的情况：会出现直接越过最大值的
ada=AdalineGD(epochs=10,eta=0.01).train(X,y)
plt.plot(range(1,len(ada.cost_)+1),np.log10(ada.cost_),marker='o')
plt.xlabel('Iterations')
plt.ylabel('log(Sum-squared-error)')
plt.title('Adaline - Learning rate 0.01')
plt.show()
#学习速率较小的情况：会出现耗时较长，也有可能会陷在局部最小值中
ada = AdalineGD(epochs=10, eta=0.0001).train(X, y)
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Sum-squared-error')
plt.title('Adaline - Learning rate 0.0001')
plt.show()
'''最好的方法是学习速率是变化的，即刚开始的时候比较大，后面慢慢的变小，类似于DTV的算法'''

#特征标准化：即通过减去平均值除以标准差来实现标准化。
X_std=np.copy(X)
#这里有两个属性所以需要算两遍
X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()

########
import matplotlib.pyplot as plt
from mlxtend.evaluate import plot_decision_regions

ada=AdalineGD(epochs=15,eta=0.01)

ada.train(X_std,y)
plot_decision_regions(X_std,y,clf=ada)
plt.title('Adaline - Gradient Descent')
plt.ylabel('petal length [standardized]')
plt.show()

plt.plot(range(1, len( ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Sum-squared-error')
plt.show()

#########SGD随机梯度下降
import numpy as np
class AdalineSGD(object):

    def __init__(self,eta=0.01,epochs=50):
        self.eta=eta
        self.epochs=epochs

    def train(self,X,y,reinitialize_weights=True):

        if reinitialize_weights:
            self.w_=np.zeros(1+X.shape[1])
        self.cost_=[]
        for i in range(self.epochs):
            '''权值的更新是通过随着每个样本的增加而更新，
            “批量”计算代价。
            随机性是体现在theta值得更新上，一个是使用批量计算，一个是使用单个样本计算'''
            for xi, target in zip(X,y):
                output=self.net_input(xi)
                error=(target-output)
                self.w_[1:]+=self.eta * xi.dot(error)
                self.w_[0] += self.eta * error
            cost = ((y - self.activation(X))**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    def net_input(self,X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self,X):
        return self.net_input(X)

    def predict(self,X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

#######
ada = AdalineSGD(epochs=15, eta=0.01)

# shuffle data
np.random.seed(123)
idx = np.random.permutation(len(y))
X_shuffled, y_shuffled =  X_std[idx], y[idx]

# train and adaline and plot decision regions
ada.train(X_shuffled, y_shuffled)
plot_decision_regions(X_shuffled, y_shuffled, clf=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.show()

plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Sum-squared-error')
plt.show()

