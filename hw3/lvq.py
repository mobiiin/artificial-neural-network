import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def dataset():
    """reading Iris dataset and splitting to train and test sections"""

    iris_dataset = pd.read_csv('iris.data', sep=',', header=None)

    X = iris_dataset.iloc[:, 0:4].values
    y = iris_dataset.iloc[:, -1].values

    y = np.where(y == 'Iris-setosa', 0, y)
    y = np.where(y == 'Iris-versicolor', 1, y)
    y = np.where(y == 'Iris-virginica', 2, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    return X, y, X_train, X_test, y_train, y_test


class LVQ(object):
    """Learning Vector Quantization"""

    def __init__(self, _input, labels, epoch, learning_rate):
        """
        :param _input: numpy array input dataset
        :param labels: numpy array target labels
        """

        self._input = _input
        self.labels = labels
        self.unique_labels = list(set(labels))
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.prototypes = np.empty((len(self.unique_labels), _input.shape[1]))
        self.prototype_label = np.empty(len(self.unique_labels))

    def initialize_prototypes(self):
        """Initializes prototypes using class means
        we used class means for initial weight vectors"""

        for _class in self.unique_labels:
            class_data = self._input[self.labels == _class, :]

            self.prototypes[_class] = np.mean(class_data, axis=0)
            self.prototype_label[_class] = _class

        return self.prototypes, self.prototype_label

    def find_winner(self, vec):
        """
        finds the winner neuron based on the euclidean distance
        :param vec: numpy array input vector
        :return: neuron wight vector and class number
        """
        distances = list(self.distance(vec, p) for p in self.prototypes)
        min_dist_index = distances.index(min(distances))

        winner = self.prototypes[min_dist_index]
        winner_lbl = min_dist_index

        return winner, winner_lbl

    def update_weights(self, vec, winner, winner_lbl, sign, _iter):
        """
        Updates winner prototype vector
        :param vec: numpy array input vector
        :param winner: winner neuron weight vector
        :param winner_lbl: winner neuron label
        :param sign: used to update the weight vector,
        if the input vector belongs to the corresponding neuron/class
        the sign will be positive, and vise versa
        :param _iter: number of current epoch
        """
        self.prototypes[winner_lbl] += sign * self.L(_iter) * np.subtract(vec, winner)

    def L(self, _iter):
        return self.learning_rate / (1 + _iter / (self.epoch / 2))

    def distance(self, x, w):
        return np.linalg.norm(np.subtract(x, w))

    def train(self):
        """
        we periodically check to see whether the train score
        is high enough to terminate the training process
        """
        self.initialize_prototypes()
        for _iter in range(self.epoch):
            for inp, lbl in zip(self._input, self.labels):
                winner, winner_lbl = self.find_winner(inp)
                if winner_lbl == lbl:
                    sign = 1
                else:
                    sign = -1
                self.update_weights(inp, winner, winner_lbl, sign, _iter)
            if _iter % 10 == 0 and self.score(self.predict(self._input), self.labels) > .99:
                break

    def predict(self, _input):
        """

        :param _input:
        :return:
        """
        predictions = []
        for inp in _input:
            winner, winner_lbl = self.find_winner(inp)
            predictions.append(winner_lbl)
        return np.asarray(predictions)

    def score(self, predictions, labels):
        s = 0
        for p, l in zip(predictions, labels):
            if not p == l:
                s += 1
        return (1 - (s / len(labels))) * 100


def main():
    _, _, X_train, X_test, y_train, y_test = dataset()

    lvq = LVQ(X_train, y_train, epoch=100, learning_rate=.01)
    lvq.train()
    prediction = lvq.predict(X_test)
    print('final weights :\n', lvq.prototypes,
          '\n according to class order:\nIris-setosa\nIris-versicolor\nIris-virginica')
    print('LVQ score on test data :', lvq.score(prediction, y_test))


if __name__ == "__main__":
    main()
