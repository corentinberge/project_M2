# TODO : comment récupérer les données en sortie ?

# q = positions, dq = vitesses, ddq = accélérations, tau = torques

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers

# Make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)


def get_data(filename, sep, skiprows, column_names):
    return pd.read_csv(filename,
                       sep=sep,
                       skiprows=skiprows,
                       names=column_names)


def separate_data(dataset, repartition):
    train_dataset = dataset.sample(frac=repartition, random_state=0)
    return train_dataset, dataset.drop(train_dataset.index)


def create_model(input_shape, learning_rate):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Dense(64, activation=tf.nn.relu)(inputs)
    x = layers.Dense(64, activation=tf.nn.relu)(x)
    outputs = layers.Dense(2)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(
        loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

    return model


def train_model(model, dataset, target, batch_size, epochs, plot_loss=False, plot_accuracy=False):
    history = model.fit(dataset, target, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    if plot_loss:
        plt.plot(history.epoch, history.history['loss'], label="Train")
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
    '''
    Obtention des données
    '''
    # filename = "data_test.txt"
    filename = "copy_2dof_data_LC_V1.txt"
    raw_dataset = pd.read_csv(filename,
                              sep='\t',
                              skiprows=1,
                              names=["q1", "q2", "dq1", "dq2", "ddq1", "ddq2", "tau1", "tau2"])

    dataset = raw_dataset.copy()
    dataset.tail()

    '''
    Nettoyer les données (inutile ?)
    '''
    dataset.isna().sum()
    dataset = dataset.dropna()

    '''
    Diviser les données en ensembles d'apprentissage et de test
    '''
    train_dataset = dataset.sample(frac=0.7, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    '''
    Inspectez les données
    '''
    # sns.pairplot(train_dataset[['tau1']], diag_kind='kde')

    '''
    Séparer les entités des étiquettes
    '''
    train_labels = train_dataset.copy()
    train_labels.pop('tau1')
    train_labels.pop('tau2')

    test_labels = test_dataset.copy()
    test_labels.pop('tau1')
    test_labels.pop('tau2')

    '''
    Normalisation
    '''
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    inputs = tf.keras.Input(shape=(8,))

    x = layers.Dense(64, activation=tf.nn.relu)(inputs)
    x = layers.Dense(64, activation=tf.nn.relu)(x)
    outputs = layers.Dense(2)(x)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(raw_dataset)

    normalizer(raw_dataset.iloc[:3])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(name='mean_absolute_error'),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=tf.keras.metrics.Accuracy(name='accuracy'),
    )

    model.compile(
        loss='mean_absolute_error',
        optimizer='adam',
        metrics=["accuracy"]
    )

    # tout_sans_tau = (np.array(train_dataset))[:, :6]
    # train_que_tau = (np.array(train_dataset))[:, 6:]
    # test_que_tau = (np.array(test_dataset))[:, 6:]
    train_que_tau = (train_dataset.T)[6:].T
    test_que_tau = (test_dataset.T)[6:].T

    history = model.fit(train_dataset, train_que_tau, batch_size=64, epochs=100, validation_split=0.2)

    plt.title("Training avec batch size = {}".format(64))
    plt.plot(history.epoch, history.history['loss'], label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    test_results = {'linear_model': model.evaluate(test_dataset, test_que_tau)}

    print("\t ----- Test -----")
    print("Loss : ", test_results['linear_model'][0])
    print("Accuracy : ", test_results['linear_model'][1])

    plt.title("Training avec batch_size={}".format(64))
    plt.plot(history.epoch, history.history['loss'], label="Train")
    # plt.plot(iters_sub, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    test_predictions = model.predict(test_dataset).flatten()

    print()
