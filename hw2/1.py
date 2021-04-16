import numpy as np
from numpy.random import rand
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


class Perceptron(object):

    def __init__(self, num_features, epoch=10, learning_rate=.01, initial_value=None):
        """
        creates a perceptron model with specifies values
        :param num_features: number of inputs
        :param epoch: number of epochs of training session
        :param learning_rate: learning rate alpha
        :param initial_value: initial weights
        :return: An object
        """
        if initial_value is None:
            initial_value = []
        self.epoch = epoch
        self.learning_rate = learning_rate
        if not initial_value:
            self.weights = np.ones(num_features + 1)  # the first row is bias
        else:
            self.weights = np.array(initial_value, dtype=float)

    def sigmoid(self, X):
        """
        activation function augmented by a threshold to facilitate the learning process
        :param X: numpy array input
        :return: result of sigmoid function
        """
        s = 1 / (1 + np.exp(-X))
        if s > .9:
            s = 1
        else:
            s = 0
        return s

    def predict(self, inputs):
        """
        :param inputs: numpy array input
        :return: tests the inputs by the trained weights
        """
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]

        activation = self.sigmoid(summation)
        return activation

    def train(self, training_inputs, labels):
        """
        trains the weights and biases based on input data and their corresponding label
        :param training_inputs: numpy array input
        :param labels: numpy array input's labels
        """
        for _ in range(self.epoch):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

    def mae(self, training_inputs, labels):
        """
        calculates the mean absolute error
        :param training_inputs: numpy array input
        :param labels: numpy array input labels
        :return: mean absolute error
        """
        y_pred = np.zeros(len(labels))
        for inputs, n in zip(training_inputs, range(len(labels))):
            y_pred[n] = self.predict(inputs)
        mae = mean_absolute_error(labels, y_pred)
        return mae


class MLP:
    def __init__(self, num_features=2, hidden_size=2, epoch=10, learning_rate=.01):
        """
        Builds a multilayer perceptron network
        :param num_features: Number of input features
        :param learning_rate: learning rate as a coefficient for updating weights
        :param hidden_size: Number of neurons in hidden layer
        :return: An object
        """
        self.w1 = np.ones((hidden_size, num_features))  # weights of input to hidden layer
        self.b1 = np.ones((hidden_size, 1))  # bias of hidden layer
        self.w2 = np.ones((1, hidden_size))  # weights of hidden layer to output
        self.b2 = np.ones((1, 1))  # bias of output layer

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.epoch = epoch

    def sigmoid(self, X):
        s = 1 / (1 + np.exp(-X))
        for t1, t2 in enumerate(s):
            if t2 < .1:
                s[t1] = 0
            elif t2 > .9:
                s[t1] = 1
        return s

    def sigmoid_der(self, x):
        return x * np.exp(-x) / np.power((1 + np.exp(-x)), 2)

    def forward(self, x):
        """
        Applies feed forward logic regarding predefined activation function
        :param x: numpy ndarray of input data
        :return: a tuple of (input of activation function of hidden layer,
                            output of activation function of hidden layer,
                            input of activation function of output layer,
                            output of activation function of output layer)
        """
        z1 = np.dot(self.w1, x) + self.b1  # output of first layer
        a1 = self.sigmoid(z1)  # output of activation function of first layer
        z2 = np.dot(self.w2, a1) + self.b2  # output of second layer
        a2 = self.sigmoid(z2)  # output of activation function of second layer
        return z1, a1, z2, a2

    def backpropagation(self, x, y, z1, a1, z2, a2):
        """
        Applies backpropagation algorithm on network
        :param x: numpy ndarray of input data
        :param y: numpy ndarray of input labels
        :param z1: input of activation function of hidden layer
        :param a1: output of activation function of hidden layer
        :param z2: input of activation function of output layer
        :param a2: output of activation function of output layer
        :return: A tuple of (delta_w1, delta_b1, delta_w2, delta_b2)
        """

        delta3 = a2 - y  # yhat - y
        dw2 = np.dot(delta3, a1.T)
        db2 = np.sum(delta3, axis=1, keepdims=True)
        delta2 = np.multiply(np.dot(self.w2.T, delta3), (1 - np.power(a1, 2)))
        # delta2 = np.multiply(np.dot(self.w2.T, delta3), self.sigmoid_der(a1))
        dw1 = np.dot(delta2, x.T)
        db1 = np.sum(delta2, axis=1, keepdims=True)
        return dw1, db1, dw2, db2

    def update(self, dw1, db1, dw2, db2):
        """
        Updates parameters using obtained gradient changes
        :param dw1: Amount of change in w1
        :param db1: Amount of change in b1
        :param dw2: Amount of change in w2
        :param db2: Amount of change in b2
        """
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2

    def train(self, training_inputs, labels):
        """
        Applies training procedure
        :param labels: numpy ndarray of input features
        :param training_inputs: numpy ndarray of input labels
        """
        for e in range(self.epoch):
            for x, y in zip(training_inputs, labels):
                x = x.reshape(2, 1)
                z1, a1, z2, a2 = self.forward(x)
                dw1, db1, dw2, db2 = self.backpropagation(x, y, z1, a1, z2, a2)
                self.update(dw1, db1, dw2, db2)

    def test(self, x):
        """
        Tests network using trained weights
        :param x: numpy ndarray of input features
        :return: numpy ndarray of predictions
        """
        _, _, _, preds = self.forward(x)
        return preds

    def mae(self, x, y):
        '''
        calculates the mean absolute error
        :param x: numpy ndarray of input features
        :param y: numpy ndarray of input labels
        :return: mean absolute error
        '''
        y_pred = np.zeros(len(y))
        for inputs, n in zip(x, range(len(y))):
            inputs = inputs.reshape(-1, 1)
            y_pred[n] = self.test(inputs)
        mae = mean_absolute_error(y, y_pred)
        return mae


def dataset(num_features=2, num_samples=200):
    """
    creates the input and labels dataset required for learning procedure of function_1
    :param num_features: number of feature inputs
    :param num_samples: number of sample inputs
    :return: numpy ndarray of input and label
    """
    _input = np.zeros((num_samples, num_features), dtype=float)
    _labels = np.zeros(num_samples)
    for samples in range(len(_input)):
        _input[samples] = (2 * rand(num_features)) - 1  # random number between -1,1
        if _input[samples][1] >= 0:
            _labels[samples] = 1
    return _input, _labels


def dataset_nand(num_features=2, num_samples=200):
    """
    creates the input and labels dataset required for learning procedure of function_1
    :param num_features: number of feature inputs
    :param num_samples: number of sample inputs
    :return: numpy ndarray of input and label
    """
    _input = np.zeros((num_samples, num_features), dtype=float)
    _labels = np.ones(num_samples)
    for samples in range(len(_input)):
        _input[samples] = (2 * rand(num_features)) - 1  # random number between -1,1
        if _input[samples][1] >= 0 and _input[samples][0] >= 0:
            _labels[samples] = 0
    return _input, _labels


def plot_mae(list_epoch, learning_rate=.01, initial_value=None, nand=False):
    """
    plots mean absolute error graph for the list of epochs provided
    :param list_epoch: list of epoch numbers
    :param learning_rate: learning rate coefficient alpha
    :param initial_value: initial wights value
    :param nand: if true plots the mae for function_2, default is function_1
    :return: plots the graph
    """
    if not nand:
        inputs, labels = dataset(2, 200)
    else:
        inputs, labels = dataset_nand(2, 200)
    mae = np.zeros(len(list_epoch))
    for n, i in zip(list_epoch, range(len(list_epoch))):
        perceptron = Perceptron(2, n, learning_rate, initial_value)
        perceptron.train(inputs, labels)
        mae[i] = perceptron.mae(inputs, labels)
        if n == list_epoch[-1]:
            print('final mean absolute error:', mae[-1])
        del perceptron.weights

    plt.plot(list_epoch, mae, color="orange", label="mean absolute error")
    plt.xlabel('epoch')
    plt.ylabel('error')
    txt = initial_value, 'learning_rate:%f' %learning_rate
    plt.title(txt)
    plt.legend()
    plt.show()


def plot_3d(epoch=10, nand=False):
    """
    plots the final result of the trained model on the input data
    :param epoch: learning epoch number
    :param nand: if true, performs the process for function_2, default is function_1
    :return: plots 3d result of predicted output
    """
    if not nand:
        inputs, labels = dataset(2, 200)
    else:
        inputs, labels = dataset_nand(2, 200)
    perceptron = Perceptron(2, epoch)
    perceptron.train(inputs, labels)
    x_1 = inputs[:, 0]
    x_2 = inputs[:, 1]

    def z_function(x, y):
        Z = np.zeros(len(x))
        for x, y, n in zip(x, y, range(len(x))):
            Z[n] = perceptron.predict([x, y])
        return Z

    z = z_function(x_1, x_2)
    ax = plt.axes(projection="3d")
    ax.scatter3D(x_1, x_2, labels, color="orange", label="label")
    ax.scatter3D(x_1, x_2, z, color="blue", label="predicted")
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()


def plot_weights(epoch=100):
    """
    plots the weight variation during the learning process
    :param epoch: learning epoch number
    :return: plots a graph
    """
    inputs, labels = dataset(2, 200)
    perceptron = Perceptron(2, epoch)
    parameters = np.zeros((epoch, len(perceptron.weights)))

    for i in range(perceptron.epoch):
        parameters[i] = perceptron.weights
        for in_put, label in zip(inputs, labels):
            prediction = perceptron.predict(in_put)
            perceptron.weights[1:] += perceptron.learning_rate * (label - prediction) * in_put
            perceptron.weights[0] += perceptron.learning_rate * (label - prediction)

    plt.plot(np.linspace(0, epoch, epoch), parameters[:, 0], color="orange", label="bias")
    plt.plot(np.linspace(0, epoch, epoch), parameters[:, 1], color="blue", label="x_1 weight")
    plt.plot(np.linspace(0, epoch, epoch), parameters[:, 2], color="red", label="x_2 weight")
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def plot_mae_mlp(list_epoch, learning_rate=.01, nand=False):
    """
    plots mean absolute error graph for the list of epochs provided for multilayer perceptron network
    :param list_epoch: list of epoch numbers
    :param learning_rate: learning rate coefficient alpha
    :param nand: if true plots the mae for function_2, default is function_1
    :return: plots the graph
    """
    if not nand:
        inputs, labels = dataset(2, 200)
    else:
        inputs, labels = dataset_nand(2, 200)

    mae = np.zeros(len(list_epoch))

    for n, i in zip(list_epoch, range(len(list_epoch))):
        mlp = MLP(2, 2, n, learning_rate)
        mlp.train(inputs, labels)
        mae[i] = mlp.mae(inputs, labels)
        if n == list_epoch[-1]:
            print('final mean absolute error:', mae[-1])

    plt.plot(list_epoch, mae, color="orange", label="mean absolute error")
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def plot_3d_mlp(epoch=100, nand=False):
    """
    plots the final result of the trained model on the input data
    :param epoch: learning epoch number
    :param nand: if true, performs the process for function_2, default is function_1
    :return: plots 3d result of predicted output
    """
    if not nand:
        inputs, labels = dataset(2, 200)
    else:
        inputs, labels = dataset_nand(2, 200)

    mlp = MLP(2, 2, epoch)
    mlp.train(inputs, labels)
    x_1 = inputs[:, 0]
    x_2 = inputs[:, 1]

    def z_function(x, y):
        Z = np.zeros(len(x))
        for x, y, n in zip(x, y, range(len(x))):
            x = x.reshape(1)
            y = y.reshape(1)
            Z[n] = mlp.test([x, y])
        return Z

    z = z_function(x_1, x_2)
    ax = plt.axes(projection="3d")
    ax.scatter3D(x_1, x_2, labels, color="orange", label="label")
    ax.scatter3D(x_1, x_2, z, color="blue", label="predicted")
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()


##### part_1 section_3
plot_mae([1, 2, 3, 5, 10, 15, 20, 30, 50, 100])
plot_3d(epoch=20)

##### part_1 section_4
plot_weights()

##### part_1 section_5
plot_mae([1, 2, 3, 5, 10, 15, 20, 30, 50, 100], initial_value=[.2, .2, .5])
plot_mae([1, 2, 3, 5, 10, 15, 20, 30, 50, 100], initial_value=[-1, 6, .5])
plot_mae([1, 2, 3, 5, 10, 15, 20, 30, 50, 100], initial_value=[-1, -2, -3])
plot_mae([1, 2, 3, 5, 10, 15, 20, 30, 50, 100], initial_value=[20, 15, 30])
plot_mae([1, 2, 3, 5, 10, 15, 20, 30, 50, 100], initial_value=[.002, .002, .005])

##### part_1 section_6
plot_mae([1,2,3,5,10,15,20,30,50], learning_rate=1)
plot_mae([1,2,3,5,10,15,20,30,50,100], learning_rate=.01)
plot_mae([1,2,3,5,10,15,20,30,50,100,500], learning_rate=.001)

##### part_1 section_7 nand function
plot_mae([1,5,10,15,20,30,50,100,500], nand=True)
plot_3d(epoch=200, nand=True)


##### part_2 multilayer network section_1
plot_mae_mlp([1,10,20,50,100,200,500,700])
plot_3d_mlp(epoch=700)

##### part_2 section_2 nand function
plot_mae_mlp([1,10,50,100,200,500,800], nand=True)
plot_3d_mlp(epoch=800, nand=True)
