#coding=utf-8
import random
import numpy as np

sizes=[64,30,10]#二层神经网络
num_layers=len(sizes)
biases=[np.random.randn(y,1) for y in sizes[1:]]#按照神经网络层次切分每层阈值
weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]#按照神经网络层次切分每层权重

def SGD(training_dta,epochs,mini_batch_size,eta,test_data=None):
    if test_data:
        n_test=len(test_data)
    n=len(training_dta)
    for j in xrange(epochs):#训练的循环次数
        random.shuffle(training_dta)#打乱样本
        mini_batches=[training_dta[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]#样本切片 连续

        for mini_batch in mini_batches:#切片循环运算
            nabla_b=[np.zeros(b.shape) for b in biases]
            nabla_w=[np.zeros(w.shape) for w in weights]

            for x,y in mini_batch:#对于每一个切片中的样本训练

                delta_nabla_b=[np.zeros(b.shape) for b in biases]#设置对于每一个样本进行计算时的初始权重为零值
                delta_nabla_w=[np.zeros(w.shape) for w in weights]
                activation=x#输入特征值
                activations=[x]#数组化特征值
                zs=[]
                for b,w in zip(biases,weights):#把每层的输出值都计算出来
                    z=np.dot(w,activation)+b
                    zs.append(z)
                    activation=1.0 / (1.0 + np.exp(-z))
                    activations.append(activation)

                delta=(activations[-1]-y)*sigmoid_prime(zs[-1])#计算最后一层的偏差值
                delta_nabla_b[-1]=delta#计算最后一层的b偏差

                delta_nabla_w[-1]=np.dot(delta,activations[-2].transpose())#计算最后一层的W偏差

                for layer in xrange(2,num_layers):#这里才是核心，倒着计算每层的偏差

                    delta=np.dot(weights[-layer+1].transpose(),delta)*sigmoid_prime(zs[-layer])#倒着计算，注意这里的delta是循环更新的

                    delta_nabla_b[-layer]=delta#

                    delta_nabla_w[-layer]=np.dot(delta,activations[-layer-1].transpose())

                nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]#这里是不停的迭代跟新

                nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]

            weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(weights,nabla_w)]
            biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(biases,nabla_b)]


        if test_data:
            print 'Epoch {0}:{1}/{2}'.format(j, evaluate(test_data), n_test)
        else:
            print 'Epoch {0} complete'.format(j)

def sigmoid_prime(z):
    """sigmoid的求导."""
    return sigmoid(z) * (1 - sigmoid(z))
def evaluate(self, test_data):
    test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

from sklearn import datasets
traing_data=datasets.load_digits()

SGD(traing_data,10,10,1)