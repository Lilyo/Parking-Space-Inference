import numpy as np
import math
def load_data(filename, seq_len, nb_classes_for_cnn, nb_classes_for_lstm , n_channels, shuffle):

    data = []
    with open(filename, 'r') as file_to_read:
        print('> Loading data... ')
        while True:

            lines = file_to_read.readline()
            if not lines:
                break

            sample = [float(i) for i in lines.split(',')]
            data.append(sample)

    if shuffle == True:
        np.random.shuffle(data)

    One_hot_label_for_cnn = one_hot_coding_label_for_cnn(data, nb_classes_for_cnn)
    One_hot_label_for_lstm = one_hot_coding_label_for_lstm(data, nb_classes_for_lstm)

    data_arr = np.array(data)
    X = np.zeros((len(data), seq_len, n_channels))  # [949 80 3]
    for i in range(n_channels):
        dat_ = data_arr[:, 3 + i * seq_len:3 + i * seq_len + seq_len]
        X[:, :, i] = dat_

    return X, One_hot_label_for_cnn, One_hot_label_for_lstm

def standardize_test(train, mean,var):
    X_train = (train - mean[None, :, :]) / var[None, :, :]
    return X_train

def standardize(train, test):
    X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
    X_test = (test - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]

    # when use testing data, we must standardize data by mean and variance of training , the following one line is not correct
    # X_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]
    return X_train, X_test, np.mean(train, axis=0), np.std(train, axis=0)

def cutting(data_array, One_hot_label_for_cnn_s, One_hot_label_for_lstm_s, training_percentage):

    row = math.floor(training_percentage * data_array.shape[0])
    x_train = data_array[:int(row)]
    x_test = data_array[int(row):]
    y_train_for_cnn = One_hot_label_for_cnn_s[:int(row)]
    y_test_for_cnn = One_hot_label_for_cnn_s[int(row):]

    y_train_for_lstm = One_hot_label_for_lstm_s[:int(row)]
    y_test_for_lstm = One_hot_label_for_lstm_s[int(row):]

    return x_train, x_test, y_train_for_cnn, y_test_for_cnn, y_train_for_lstm, y_test_for_lstm


def one_hot_init(nb_classes):
    print('> Initialize One hot...')
    targets = np.array([np.arange(nb_classes)]).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    print('> Initialize finished.')
    return one_hot_targets

def one_hot_coding_label_for_cnn(label,nb_classes):
    print('> One hot coding label...')
    # One hot init
    one_hot_targets = one_hot_init(nb_classes)

    label = np.array(label)

    one_hot_label = np.zeros(shape=(label.shape[0], nb_classes))
    for i in range(label.shape[0]):
        if label[i][1] == 0:
            one_hot_label[i] = one_hot_targets[0]
        elif label[i][1] == 1:
            one_hot_label[i] = one_hot_targets[1]
        elif label[i][1] == 2:
            one_hot_label[i] = one_hot_targets[2]
        elif label[i][1] == 3:
            one_hot_label[i] = one_hot_targets[3]
    print('> One hot coding label finished.')
    return one_hot_label

def one_hot_coding_label_for_lstm(label,nb_classes):
    print('> One hot coding label...')
    # One hot init
    one_hot_targets = one_hot_init(nb_classes)

    label = np.array(label)

    one_hot_label = np.zeros(shape=(label.shape[0], nb_classes))
    for i in range(label.shape[0]):
        if label[i][2] == 0:
            one_hot_label[i] = one_hot_targets[0]
        elif label[i][2] == 1:
            one_hot_label[i] = one_hot_targets[1]
        elif label[i][2] == 2:
            one_hot_label[i] = one_hot_targets[2]
        elif label[i][2] == 3:
            one_hot_label[i] = one_hot_targets[3]
    print('> One hot coding label finished.')
    return one_hot_label