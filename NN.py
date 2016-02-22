# coding=utf-8
'''
network.py
~~~~~~~~~~~~~~~~~~~~~~
这个模型的宗旨是根据随机梯度下降法制作一个前馈式神经网络
梯度的计算是通过逆向（BP）传播的方法。
'''
###库函数
# 标准库函数
import random
# 第三方库函数
import numpy as np


class Network(object):
    def __init__(self, sizes):  # 初始化变量
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):  # 前馈函数
        '''返回输入a对应的网络'''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''使用最小梯度下降法训练神经网络。
        其中训练数据（training_data）的格式为：（x，y）的形式，x为训练数据，y为期望得到的结果
        测试数据（test_data）默认是没有的，如果有测试数据将会被计算并被部分输出
        其中epochs 表示的是训练的回合数
        其中mini_batch_size表示的是用于梯度下降法的最小样本数量
        其中eta表示步长（学习速度）'''
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            # 将训练数据打乱随机排序（注意（x,y）的形式没变）
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]
            # xrange(0,**)是生成一个对象：使用xrange()进行遍历，每次遍历只返回一个值。
            # range()返回的是一个列表，一次性计算并返回所有的值。
            # 因此，xrange()的执行效率要高于range()
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print 'Epoch {0}:{1}/{2}'.format(j, self.evaluate(test_data), n_test)
            else:
                print 'Epoch {0} complete'.format(j)

    def update_mini_batch(self, mini_batch, eta):
        '''更新训练网络中的 权重 和 偏移 的值，通过梯度下降法，逆向传递一个最小样本数，
        eta是步长（学习速度）'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        """返回梯度值（nable_b,nable_w）表示C-x的梯度值，可以看做是cost函数对w,b的求导结果"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """测试数据的正确性"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """返回cost值，也就是计算出的值和想要得到的结果的值"""
        return (output_activations - y)


#### Miscellaneous functions
def sigmoid(z):
    """sigmoid方法"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """sigmoid的求导."""
    return sigmoid(z) * (1 - sigmoid(z))

