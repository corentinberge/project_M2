import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import signal
from tensorflow.keras import layers

# Make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)


class NeuralNetwork(tf.keras.Model):
    def __init__(self, nb_axes):
        super(NeuralNetwork, self).__init__()
        self.nb_axes = nb_axes
        self.norm = tf.keras.layers.Normalization(axis=-1)
        self.dense1 = layers.Dense(64, activation=tf.nn.relu)
        self.dense2 = layers.Dense(64, activation=tf.nn.relu)
        self.out = layers.Dense(self.nb_axes)
        self.tech = 1 / 250

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
        inputs = tf.keras.Input(shape=input_shape)

        x = self.norm(inputs)
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

    def get_target(self, train_dataset, test_dataset, nb_axes):
        """
        Function that gives targets for the train and for the test according to the number of joints used

        :param train_dataset:
        :param test_dataset:
        :param nb_axes: number of joints used
        :return: 2 dataset with only the target in it, one for training and one for test
        """
        index = nb_axes * 3  # *3 to have q, dq, ddq for each joint
        return train_dataset.T[index:].T, test_dataset.T[index:].T

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

    def get_predictions_all_torques(self, model, dataset):
        """
        Function that makes predictions for the torques

        :param model: model from which to get predictions
        :param dataset: dataset to use
        :return: list of all the predicted values of all the torques
        """
        return model.predict(dataset)

    def get_predictions_one_torque(self, predictions, which_torque):
        """
        Function that gives the predictions for only one torque

        :param predictions: list of the predictions from which to get the predicted value for the torque
        :param which_torque: number of the torque to get
        :return: list of all the predicted values of the chosen torque
        """
        return predictions.T[which_torque - 1:which_torque].T

    def plot_torques_6dof(self, real_torques, predicted_torques, nb_axes=6):
        """
        Functions that plot the predicted torques and the real torques

        :param real_torques: DataFrame that contains the real torques
        :param predicted_torques: list that contains the predicted torques
        :param nb_axes: (6 by default) number of axes used
        :return: graphic with real and predicted torques
        """
        pred_tau1 = self.get_predictions_one_torque(predicted_torques, 1)
        pred_tau2 = self.get_predictions_one_torque(predicted_torques, 2)
        pred_tau3 = self.get_predictions_one_torque(predicted_torques, 3)
        pred_tau4 = self.get_predictions_one_torque(predicted_torques, 4)
        pred_tau5 = self.get_predictions_one_torque(predicted_torques, 5)
        pred_tau6 = self.get_predictions_one_torque(predicted_torques, 6)

        pred_t = []
        for i in range(len(pred_tau1.flatten())):
            pred_t.append(pred_tau1.flatten()[i])
        for i in range(len(pred_tau2.flatten())):
            pred_t.append(pred_tau2.flatten()[i])
        for i in range(len(pred_tau3.flatten())):
            pred_t.append(pred_tau3.flatten()[i])
        for i in range(len(pred_tau4.flatten())):
            pred_t.append(pred_tau4.flatten()[i])
        for i in range(len(pred_tau5.flatten())):
            pred_t.append(pred_tau5.flatten()[i])
        for i in range(len(pred_tau6.flatten())):
            pred_t.append(pred_tau6.flatten()[i])

        toto = real_torques.T[nb_axes * 3:].T

        real_tau1 = (toto.T[0:1].T).to_numpy()
        real_tau2 = (toto.T[1:2].T).to_numpy()
        real_tau3 = (toto.T[2:3].T).to_numpy()
        real_tau4 = (toto.T[3:4].T).to_numpy()
        real_tau5 = (toto.T[4:5].T).to_numpy()
        real_tau6 = (toto.T[5:6].T).to_numpy()

        real_t = []
        for i in range(len(real_tau1.flatten())):
            real_t.append(real_tau1.flatten()[i])
        for i in range(len(real_tau2.flatten())):
            real_t.append(real_tau2.flatten()[i])
        for i in range(len(real_tau3.flatten())):
            real_t.append(real_tau3.flatten()[i])
        for i in range(len(real_tau4.flatten())):
            real_t.append(real_tau4.flatten()[i])
        for i in range(len(real_tau5.flatten())):
            real_t.append(real_tau5.flatten()[i])
        for i in range(len(real_tau6.flatten())):
            real_t.append(real_tau6.flatten()[i])

        rt = [i for i in range(282)]

        plt.plot(real_t, 'r', label="Real torques")
        plt.plot(pred_t, 'b--', label="Predicted torques")

        # #plt.title("Model loss with batch size = {}".format(batch_size))
        plt.xlabel("Samples")
        plt.ylabel("Torques (N/m)")
        plt.legend()
        plt.show()

    # Use of indentification's functions
    def filter_butterworth(self, sampling_freq, f_coupure, sig):
        sfreq = sampling_freq
        f_p = f_coupure
        nyq = sfreq / 2

        sos = signal.iirfilter(5, f_p / nyq, btype='low', ftype='butter', output='sos')
        signal_filtrer = signal.sosfiltfilt(sos, sig)

        return signal_filtrer

    def param_from_txt(self, nb_joint):
        file_path = "5_sec.txt"

        tau_par_ordre = []
        with open(file_path, 'r') as f:

            tau1, tau2, tau3, tau4, tau5, tau6 = [], [], [], [], [], []
            q1, q2, q3, q4, q5, q6, q = [], [], [], [], [], [], []
            dq1, dq2, dq3, dq4, dq5, dq6, dq = [], [], [], [], [], [], []

            tau_simu_gazebo = []

            for line in f:
                data_split = line.strip().split('\t')

                q1.append(data_split[0])
                q2.append(data_split[1])
                q3.append(data_split[2])
                q4.append(data_split[3])
                q5.append(data_split[4])
                q6.append(data_split[5])

                dq1.append(data_split[6])
                dq2.append(data_split[7])
                dq3.append(data_split[8])
                dq4.append(data_split[9])
                dq5.append(data_split[10])
                dq6.append(data_split[11])

                tau1.append(data_split[12])
                tau2.append(data_split[13])
                tau3.append(data_split[14])
                tau4.append(data_split[15])
                tau5.append(data_split[16])
                tau6.append(data_split[17])

        q.append(q1)
        q.append(q2)
        q.append(q3)
        q.append(q4)
        q.append(q5)
        q.append(q6)
        q = np.array(q)
        q = np.double(q)

        dq.append(dq1)
        dq.append(dq2)
        dq.append(dq3)
        dq.append(dq4)
        dq.append(dq5)
        dq.append(dq6)
        dq = np.array(dq)
        dq = np.double(dq)

        tau_simu_gazebo = np.array(tau_simu_gazebo)
        tau_simu_gazebo = np.double(tau_simu_gazebo)
        tau4 = np.double(tau4)
        tau4 = abs(tau4)
        tau_par_ordre.append(tau1)
        tau_par_ordre.append(tau2)
        tau_par_ordre.append(tau3)
        tau_par_ordre.append(tau4)
        tau_par_ordre.append(tau5)
        tau_par_ordre.append(tau6)
        tau_par_ordre = np.array(tau_par_ordre)
        tau_par_ordre = np.double(tau_par_ordre)

        ddq = [[], [], [], [], [], []]
        dq_th = [[], [], [], [], [], []]

        for joint_index in range(nb_joint):

            for i in range(q[0].size - 1):
                j = i + 1
                dv = (q[joint_index][j] - q[joint_index][i]) / self.tech
                # print('da=\t','dv',(v[j]-v[i]),'/dt',(time[j]-time[i]),'=',da)
                dq_th[joint_index].append(dv)

            dq_th[joint_index].append(dv)

        dq_th = np.array(dq_th)

        for joint_index in range(nb_joint):

            for i in range(dq_th[0].size - 1):
                j = i + 1
                da = (dq_th[joint_index][j] - dq_th[joint_index][i]) / self.tech
                # print('da=\t','dv',(v[j]-v[i]),'/dt',(time[j]-time[i]),'=',da)
                ddq[joint_index].append(da)

            ddq[joint_index].append(0)

        ddq = np.array(ddq)

        tau_par_ordre = self.filter_butterworth(int(1 / self.tech), 5, tau_par_ordre)

        for i in range(6):
            q[i] = self.filter_butterworth(int(1 / self.tech), 5, q[i])
            dq_th[i] = self.filter_butterworth(int(1 / self.tech), 5, dq_th[i])
            ddq[i] = self.filter_butterworth(int(1 / self.tech), 5, ddq[i])

        self.generate_ddq(q, dq_th, ddq, tau_par_ordre)

        return q, dq_th, ddq, tau_par_ordre

    def generate_ddq(self, pos, vit, acc, tau):
        # this function take in input q dq ddq tau for all the joint
        # and write all the data in a file .txt
        with open("data_V2.txt", 'w+') as f:
            nbSamples = np.array(pos[0]).size
            q_pin = np.array(pos)
            dq_pin = np.array(vit)
            ddq_pin = np.array(acc)
            tau_pin = np.array(tau)
            print('shape of Q ', q_pin.shape)

            i = 0
            line = [str('q1'), '\t',
                    str('q2'), '\t',
                    str('q3'), '\t',
                    str('q4'), '\t',
                    str('q5'), '\t',
                    str('q6'), '\t',
                    str('dq1'), '\t',
                    str('dq2'), '\t',
                    str('dq3'), '\t',
                    str('dq4'), '\t',
                    str('dq5'), '\t',
                    str('dq6'), '\t',
                    str('ddq1'), '\t',
                    str('ddq2'), '\t',
                    str('ddq3'), '\t',
                    str('ddq4'), '\t',
                    str('ddq5'), '\t',
                    str('ddq6'), '\t',
                    str('tau1'), '\t',
                    str('tau2'), '\t',
                    str('tau3'), '\t',
                    str('tau4'), '\t',
                    str('tau5'), '\t',
                    str('tau6')]
            f.writelines(line)
            f.write('\n')

            for i in range(nbSamples):
                line = [str(q_pin[0][i]), '\t',
                        str(q_pin[1][i]), '\t',
                        str(q_pin[2][i]), '\t',
                        str(q_pin[3][i]), '\t',
                        str(q_pin[4][i]), '\t',
                        str(q_pin[5][i]), '\t',
                        str(dq_pin[0][i]), '\t',
                        str(dq_pin[1][i]), '\t',
                        str(dq_pin[2][i]), '\t',
                        str(dq_pin[3][i]), '\t',
                        str(dq_pin[4][i]), '\t',
                        str(dq_pin[5][i]), '\t',
                        str(ddq_pin[0][i]), '\t',
                        str(ddq_pin[1][i]), '\t',
                        str(ddq_pin[2][i]), '\t',
                        str(ddq_pin[3][i]), '\t',
                        str(ddq_pin[4][i]), '\t',
                        str(ddq_pin[5][i]), '\t',
                        str(tau_pin[0][i]), '\t',
                        str(tau_pin[1][i]), '\t',
                        str(tau_pin[2][i]), '\t',
                        str(tau_pin[3][i]), '\t',
                        str(tau_pin[4][i]), '\t',
                        str(tau_pin[5][i])]
                f.writelines(line)
                f.write('\n')


if __name__ == '__main__':
    nb_axes = 6

    # Creation of the object
    my_model = NeuralNetwork(nb_axes)

    if nb_axes == 2:
        filename = "../Identification/2dof_data_LC.txt"
        # filename = "../Identification/2dof_data_LC_V1.txt"
        # filename = "../Identification/2dof_data_LC_V3_syncronized.txt"
        column_names = ["q1", "q2", "dq1", "dq2", "ddq1", "ddq2", "tau1", "tau2"]

    if nb_axes == 6:
        q, dq_th, ddq, tau_par_ordre = my_model.param_from_txt(nb_axes)
        filename = "data_V2.txt"
        column_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6',
                        'dq1', 'dq2', 'dq3', 'dq4', 'dq5', 'dq6',
                        'ddq1', 'ddq2', 'ddq3', 'ddq4', 'ddq5', 'ddq6',
                        'tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6']

    rep_train_test = 0.8  # 80% train / 20% test
    input_shape = (len(column_names),)
    learning_rate = 0.001
    batch_size = 64
    epochs = 100

    """
    Functions calls
    """
    # Get data from file
    raw_dataset = my_model.get_data(filename, '\t', 1, column_names)

    dataset = raw_dataset.copy()

    # Separate data between train and test
    train_dataset, test_dataset = my_model.separate_data(dataset, repartition=rep_train_test)

    # Create model
    model = my_model.create_model(input_shape, learning_rate=learning_rate, show_summary=True)

    # Get target from both train and test dataset
    train_que_tau, test_que_tau = my_model.get_target(train_dataset, test_dataset, nb_axes)

    # Train the model
    history = my_model.train_model(model, train_dataset, train_que_tau, batch_size, epochs, plot_loss=True,
                                   plot_accuracy=True)

    # Evaluate the model
    result = my_model.evaluate_model(model, test_dataset, test_que_tau, show_result=True)

    # Get all predicted torques
    predictions = my_model.get_predictions_all_torques(model, dataset)

    # Get each predicted torques in a list
    tau1 = my_model.get_predictions_one_torque(predictions, 1)
    tau2 = my_model.get_predictions_one_torque(predictions, 2)

    # Plot torques
    my_model.plot_torques_6dof(dataset, predictions)

    print()
