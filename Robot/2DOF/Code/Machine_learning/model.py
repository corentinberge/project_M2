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
        """
        :return: one layer of normalization
        """
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

    def get_target(self, train_dataset, test_dataset, nb_axes):
        """
        Function that gives targets for the train and for the test according to the number of joints used

        :param train_dataset:
        :param test_dataset:
        :param nb_axes: number of joints used
        :return: 2 dataset with only the target in it, one for training and one for test
        """
        index = nb_axes * 3     # *3 to have q, dq, ddq for each joint
        return train_dataset.T[index:].T, test_dataset.T[index:].T

    def evaluate_model(self, model, dataset, target, show_result=False):
        """
        Function that evaluates the model in order to see if the neural network is learning well

        :param model: model to evaluate
        :param dataset: dataset on which to evalaute the model
        :param target: target to aim
        :param show_result: (False by default) if True, show values of the loss and the accuracy
        :return: the model evaluated
        """
        result = {'model': model.evaluate(dataset, target)}

        if show_result:
            print("Loss : {} / Accuracy : {}".format(result['model'][0], result['model'][1]))

        return result


if __name__ == '__main__':
    # Initialisation of the parameters
    filename = "../Identification/2dof_data_LC.txt"
    # filename = "../Identification/2dof_data_LC_V1.txt"
    # filename = "../Identification/2dof_data_LC_V3_syncronized.txt"

    column_names = ["q1", "q2", "dq1", "dq2", "ddq1", "ddq2", "tau1", "tau2"]

    batch_size = 64
    epochs = 100

    nb_axes = 3

    # Creation of the object
    my_model = NeuralNetwork()

    """
    Functions calls
    """
    # Get data from file
    raw_dataset = my_model.get_data(filename, '\t', 1, column_names)

    dataset = raw_dataset.copy()

    # Separate data between train and test
    train_dataset, test_dataset = my_model.separate_data(dataset, repartition=0.8)

    # Create model
    model = my_model.create_model((8, ), learning_rate=0.001)

    # Get target from both train and test dataset
    train_que_tau, test_que_tau = my_model.get_target(train_dataset, test_dataset, nb_axes)

    # Train the model
    history = my_model.train_model(model, train_dataset, train_que_tau, batch_size, epochs, plot_loss=True, plot_accuracy=True)

    # Evaluate the model
    result = my_model.evaluate_model(model, test_dataset, test_que_tau, show_result=True)

    # Predictions
    test_predictions = model.predict(test_dataset).flatten()

    print()
