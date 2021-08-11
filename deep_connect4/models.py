from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model

from connect4.costants import N_ROW, N_COL

class NeuralNetwork(Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = Flatten(input_shape=(N_ROW, N_COL))
        self.hidden1 = Dense(100, activation='relu')
        self.hidden2 = Dense(50, activation='relu')
        self.hidden3 = Dense(50, activation='relu')
        self.outlayer = Dense(N_COL)

    def call(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.outlayer(x)

class ConvolutionalNeuralNetwork(Model):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv = Conv2D(100, 4, input_shape=(N_ROW, N_COL))
        self.hidden1 = Dense(50, activation='relu')
        self.hidden2 = Dense(50, activation='relu')
        self.outlayer = Dense(N_COL)

    def call(self, x):
        x = self.conv(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.outlayer(x)