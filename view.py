import numpy as np
import matplotlib.pylab as plt
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

from ch04.two_layer_net import TwoLayerNet

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2+0.1*x

def function_2(x):
    return x[0]**2+x[1]**2

def relu(x):
    return np.maximum(0, x)

def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = tmp +h
        fxh1 = f(x)

        x[idx] = tmp -h
        fxh2 = f(x)
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp
    return grad

def binary_cross_entropy(y,y_hot):
    return -y*np.log(y_hot)-(1-y)*np.log(1-y_hot)

def loadData():
    (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False,one_hot_label=True)
    img = x_train[0]
    print(x_train.shape)
    print(t_train[0])
    img = img.reshape(28,28)
    img_show(img)
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()




if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    print(relu(x0))

    w= 1* np.random.rand(20,5)
    print(w.size)
    """x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-1.0, 5.5)
    plt.show()"""

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num = 10000  # 繰り返しの回数を適宜設定する
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        grad = network.numerical_gradient(x_batch, t_batch)
        #grad = network.gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    # グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

#x1 = np.arange(-2, 2.5, 0.25)
    #x = x.reshape(6,2)
    #print(x.size)
    #grad = np.zeros_like(x)
    #for idx in range(x.size):
    #    grad[idx] = function_2(x[idx])
    #print(grad)
    #y = numerical_gradient(function_2,x)
    #y = sigmoid(x)
    #print(softmax(y))
    #plt.plot(x,grad)
    #plt.show()
    #loadData()


