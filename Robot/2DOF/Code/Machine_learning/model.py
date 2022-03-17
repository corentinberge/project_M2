import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers

# Make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)

# q = positions, dq = vitesses, ddq = accélérations, tau = torques


class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.dense1 = layers.Dense(64, activation=tf.nn.relu)
        self.dense2 = layers.Dense(64, activation=tf.nn.relu)
        self.out = layers.Dense(2)

    def get_data(self, filename, separator, skip_rows, column_names):
        """
        Function that allows the user to get data from a csv of txt file

        :param filename: name of the file from which to get data
        :param separator: character that separates values
        :param skip_rows: how many rows to skip
        :param column_names: names of each column
        :return: DataFrame with all the data from the file
        """
        return pd.read_csv(filename,
                           sep=separator,
                           skiprows=skip_rows,
                           names=column_names)

    def separate_data(self, dataset, repartition):
        """
        Function that separates data between training and test

        :param dataset: dataset to separate
        :param repartition: how many data to keep in learning (in %)
        :return: a couple that contains 2 dataset : one for the training and one for the validation
        """
        train_dataset = dataset.sample(frac=repartition, random_state=0)
        return train_dataset, dataset.drop(train_dataset.index)

    def normalize_data(self):
        return tf.keras.layers.Normalization(axis=-1)

    def create_model(self, input_shape, learning_rate, show_summary=False):
        """
        Function that creates the model.
        It is composed by :
            - 1 input layer
            - 2 hidden layers
            - 1 output layer

        :param input_shape: shape of the input
        :param learning_rate: learning rate used by the optimizer
        :param show_summary: (False by default) if True, show the architecture of the model
        :return: the created model
        """
        normalizer = self.normalize_data()

        inputs = tf.keras.Input(shape=input_shape)

        x = normalizer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.out(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if show_summary:
            model.summary()

        model.compile(
            loss='mean_absolute_error',  # loss function to minimize
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"]  # metrics to monitor
        )

        return model

    def train_model(self, model, dataset, target, batch_size, epochs, plot_loss=False, plot_accuracy=False):
        """
        Function that trains the model with the following parameters

        :param model: model to train
        :param dataset: dataset used to train the model
        :param target: values that the model have to found
        :param batch_size: batch size
        :param epochs: number of epochs
        :param plot_loss: (False by default) if True, show the convergence of the loss
        :param plot_accuracy: (False by default) if True, show the evolution of the accuracy
        :return: the model trained
        """
        history = model.fit(dataset, target, batch_size=batch_size, epochs=epochs, validation_split=0.2)

        if plot_loss:
            plt.plot(history.epoch, history.history['loss'], label="Train")
            plt.plot(history.epoch, history.history['val_loss'], label="Val")
            plt.title("Model loss with batch size = {}".format(batch_size))
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        if plot_accuracy:
            plt.plot(history.history['accuracy'], label="Train")
            plt.plot(history.history['val_accuracy'], label="Val")
            plt.title("Model accuracy with batch size = {}".format(batch_size))
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()

        return history


if __name__ == '__main__':
    # filename = "../Identification/2dof_data_LC.txt"
    # filename = "../Identification/2dof_data_LC_V1.txt"
    filename = "../Identification/2dof_data_LC_V3_syncronized.txt"

    my_model = NeuralNetwork()
    raw_dataset = my_model.get_data(filename, '\t', 1, ["q1", "q2", "dq1", "dq2", "ddq1", "ddq2", "tau1", "tau2"])

    dataset = raw_dataset.copy()

    train_dataset, test_dataset = my_model.separate_data(dataset, 0.8)

    model = my_model.create_model((8, ), 0.001)

    train_que_tau = train_dataset.T[6:].T
    test_que_tau = test_dataset.T[6:].T

    batch_size = 64
    epochs = 100

    history = my_model.train_model(model, train_dataset, train_que_tau, batch_size, epochs, True, True)

    test_results = {'model': model.evaluate(test_dataset, test_que_tau)}

    print("\t ----- Test -----")
    print("Loss : ", test_results['model'][0])
    print("Accuracy : ", test_results['model'][1])

    test_predictions = model.predict(test_dataset).flatten()

    print()
