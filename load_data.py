import tensorflow as tf
from tools import *
from next_batch import *
import os
import tools
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ============================================= Flag =============================================
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("source_filename", "2017-12-12-21_AllLabel.txt", "Filename of source input data")
tf.flags.DEFINE_string("target_filename", "2017-11-23-23-36-51_AllLabel.txt", "Filename of target input data")
tf.flags.DEFINE_string('mode', "train", "Mode: train/test")
# ================================================================================================
# Load data
n_channels = 3
nb_classes_for_cnn = 3
nb_classes_for_lstm = 2
seq_len = 80
iteration = 10001
batch_size = 64
learning_rate = 0.0001
lambda_c = 0.7
training_percentage = 0.9
keep_prob = 1. # 0.5
# ================================================================================================
X_s, One_hot_label_for_cnn_s, One_hot_label_for_lstm_s = load_data(FLAGS.source_filename, seq_len, nb_classes_for_cnn, nb_classes_for_lstm, n_channels, False)
x_s_train, x_s_test, y_s_train_for_cnn, y_s_test_for_cnn, y_s_train_for_lstm, y_s_test_for_lstm = cutting(X_s, One_hot_label_for_cnn_s, One_hot_label_for_lstm_s, training_percentage)
# x_s_train, x_s_test, x_s_train_mean, x_s_train_var = standardize(x_s_train, x_s_test)
train_s_dataset = Dataset(x_s_train, y_s_train_for_cnn, y_s_train_for_lstm, x_s_test, y_s_test_for_cnn, y_s_test_for_lstm, False)

# X_t, One_hot_label_for_cnn_t, One_hot_label_for_lstm_t = load_data(FLAGS.target_filename, seq_len, nb_classes_for_cnn, nb_classes_for_lstm, n_channels, False)
# x_t_train, x_t_test, y_t_train_for_cnn, y_t_test_for_cnn, y_t_train_for_lstm, y_t_test_for_lstm = cutting(X_t, One_hot_label_for_cnn_t, One_hot_label_for_lstm_t, training_percentage)
# x_t_train, x_t_test, x_t_train_mean, x_t_train_var = standardize(x_t_train, x_t_test)
# train_t_dataset = Dataset(x_t_train, y_t_train_for_cnn, y_t_train_for_lstm, x_t_test, y_t_test_for_cnn, y_t_test_for_lstm, False)
#
# X_st, One_hot_label_for_cnn_st, One_hot_label_for_lstm_st = np.concatenate((X_s, X_t), axis=0), np.concatenate((One_hot_label_for_cnn_s, One_hot_label_for_cnn_t), axis=0), np.concatenate((One_hot_label_for_lstm_s, One_hot_label_for_lstm_t), axis=0)
# x_st_train, x_st_test, y_st_train_for_cnn, y_st_test_for_cnn, y_st_train_for_lstm, y_st_test_for_lstm = cutting(X_st, One_hot_label_for_cnn_st, One_hot_label_for_lstm_st, training_percentage)
# x_st_train, x_st_test, x_st_train_mean, x_st_train_var = standardize(x_st_train, x_st_test)
# train_st_dataset = Dataset(x_st_train, y_st_train_for_cnn, y_st_train_for_lstm, x_st_test, y_st_test_for_cnn, y_st_test_for_lstm, True)
print("Data loaded")


for i in range(100000):
    batch_size=3
    X_mb, y_cnn_mb, y_lstm_mb = train_s_dataset.next_batch(batch_size)
    for j in range(batch_size):
        print('Iter ',i,'Shape ',X_mb.shape)
        print(X_mb[j][0])
        print(X_mb[j][80])
        print(X_mb[j][160])

    t_0=X_mb[:,0:80]
    t_1=X_mb[:,80:160]
    t_2=X_mb[:,160:]
    print(t_0.shape)
    print(t_1.shape)
    print(t_2.shape)
    latent=tf.concat((t_0,t_1,t_2),axis=1)
    with tf.Session() as sess:
        latent_ = np.array(sess.run(latent))
        print(latent_.shape)

    for j in range(batch_size):
        print('Iter ',i,'Shape ',X_mb.shape)
        print(X_mb[j][0])
        print(X_mb[j][80])
        print(X_mb[j][160])
        #
        # print(y_cnn_mb.shape)
        # print(y_cnn_mb[j])
        #
        # print(y_lstm_mb.shape)
        # print(y_lstm_mb[i])
    quit()