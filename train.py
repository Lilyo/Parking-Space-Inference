import tensorflow as tf
import argparse
from datetime import datetime
from tools import *
from next_batch import *
from help_function import *
from sklearn import metrics
from inference import *
import matplotlib.pyplot as plt
import os
# ============================================= Flag =============================================
parser= argparse.ArgumentParser()
parser.add_argument("--train_result", type=str, default="train_result/")
parser.add_argument("--date_time", type=str, default=datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')+"/")
parser.add_argument("--model", type=str, default="model/")
parser.add_argument("--performance", type=str, default="performance/")
parser.add_argument("--miscls_source", type=str, default="miscls_source/")
parser.add_argument("--miscls_target", type=str, default="miscls_target/")
parser.add_argument("--mislstm_source", type=str, default="mislstm_source/")
parser.add_argument("--mislstm_target", type=str, default="mislstm_target/")
parser.add_argument("--validation", type=str, default="validation/")
parser.add_argument("--source_filename", type=str, default="ID_0_C2_AllLabel.txt")
parser.add_argument("--target_filename", type=str, default="ID_0_Angle_0_Disturbance_AllLabel_v1.txt")
parser.add_argument("--mode", type=str, default="train")# Mode: train/retrain/test
parser.add_argument("--gpu", type=str, default="0")# GPU{0, 1, 2}

parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--gamma", type=float, default=3.0)
parser.add_argument("--lambda_c", type=float, default=1.0)
parser.add_argument("--lambda_r", type=float, default=0.7)
parser.add_argument("--lambda_g", type=float, default=0.7)
parser.add_argument("--lambda_d", type=float, default=0.7)
parser.add_argument("--lambda_lstm", type=float, default=1.0)
parser.add_argument("--training_percentage", type=float, default=0.9)
parser.add_argument("--keep_prob", type=float, default=1.0)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--iteration", type=int, default=20001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--seq_len", type=int, default=80)
args = parser.parse_args()

# ================================================================================================
# Load data
n_channels = 3
nb_classes_for_cnn = 3
nb_classes_for_lstm = 2

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

seq_len = args.seq_len
batch_size = args.batch_size
iteration = args.iteration
learning_rate = args.learning_rate
alpha = args.alpha
gamma = args.gamma
lambda_c = args.lambda_c
lambda_r = args.lambda_r
lambda_g = args.lambda_g
lambda_d = args.lambda_d
lambda_lstm = args.lambda_lstm
training_percentage = args.training_percentage
keep_prob = args.keep_prob
# ================================================================================================
if not os.path.exists(args.train_result):
    os.makedirs(args.train_result)
if not os.path.exists(args.train_result + args.date_time):
    os.makedirs(args.train_result + args.date_time)
    os.makedirs(args.train_result + args.date_time + args.model)
    os.makedirs(args.train_result + args.date_time + args.performance)
    os.makedirs(args.train_result + args.date_time + args.performance + args.miscls_source)
    os.makedirs(args.train_result + args.date_time + args.performance + args.miscls_target)
    os.makedirs(args.train_result + args.date_time + args.performance + args.mislstm_source)
    os.makedirs(args.train_result + args.date_time + args.performance + args.mislstm_target)
# ================================================================================================
X_s, One_hot_label_for_cnn_s, One_hot_label_for_lstm_s = load_data(args.source_filename, seq_len, nb_classes_for_cnn, nb_classes_for_lstm, n_channels, False)
x_s_train, x_s_test, y_s_train_for_cnn, y_s_test_for_cnn, y_s_train_for_lstm, y_s_test_for_lstm = cutting(X_s, One_hot_label_for_cnn_s, One_hot_label_for_lstm_s, training_percentage)
x_s_train, x_s_test, x_s_train_mean, x_s_train_var = standardize(x_s_train, x_s_test)
train_s_dataset = Dataset(x_s_train, y_s_train_for_cnn, y_s_train_for_lstm, x_s_test, y_s_test_for_cnn, y_s_test_for_lstm, False)

X_t, One_hot_label_for_cnn_t, One_hot_label_for_lstm_t = load_data(args.target_filename, seq_len, nb_classes_for_cnn, nb_classes_for_lstm, n_channels, False)
x_t_train, x_t_test, y_t_train_for_cnn, y_t_test_for_cnn, y_t_train_for_lstm, y_t_test_for_lstm = cutting(X_t, One_hot_label_for_cnn_t, One_hot_label_for_lstm_t, training_percentage)
x_t_train, x_t_test, x_t_train_mean, x_t_train_var = standardize(x_t_train, x_t_test)
train_t_dataset = Dataset(x_t_train, y_t_train_for_cnn, y_t_train_for_lstm, x_t_test, y_t_test_for_cnn, y_t_test_for_lstm, False)

X_st, One_hot_label_for_cnn_st, One_hot_label_for_lstm_st = np.concatenate((X_s, X_t), axis=0), np.concatenate((One_hot_label_for_cnn_s, One_hot_label_for_cnn_t), axis=0), np.concatenate((One_hot_label_for_lstm_s, One_hot_label_for_lstm_t), axis=0)
x_st_train, x_st_test, y_st_train_for_cnn, y_st_test_for_cnn, y_st_train_for_lstm, y_st_test_for_lstm = cutting(X_st, One_hot_label_for_cnn_st, One_hot_label_for_lstm_st, training_percentage)
x_st_train, x_st_test, x_st_train_mean, x_st_train_var = standardize(x_st_train, x_st_test)
train_st_dataset = Dataset(x_st_train, y_st_train_for_cnn, y_st_train_for_lstm, x_st_test, y_st_test_for_cnn, y_st_test_for_lstm, True)
print("Data loaded")
# ================================================================================================
net = inference(seq_len, nb_classes_for_cnn, nb_classes_for_lstm, n_channels, learning_rate, gamma, alpha, lambda_c, lambda_r, lambda_g, lambda_d, lambda_lstm, keep_prob)
g_optimizer = tf.train.AdamOptimizer(net.learning_rate_).minimize(net.loss, var_list = [net.var_n, net.var_g, net.var_c, net.var_r, net.var_l])
d_optimizer = tf.train.AdamOptimizer(net.learning_rate_).minimize(net.loss_d, var_list = [net.var_d])
# ================================================================================================

if args.mode == "train":

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Setting up Saver...")
        saver = tf.train.Saver()

        x_s_test_stack, y_s_test_for_cnn_stack, y_s_test_for_lstm_stack = train_s_dataset.test_stack()
        x_t_test_stack, y_t_test_for_cnn_stack, y_t_test_for_lstm_stack = train_t_dataset.test_stack()
        x_st_test_stack, y_st_test_for_cnn_stack, y_st_test_for_lstm_stack = train_st_dataset.test_stack()
        feed_test_onSource = {net.x1_t_0: x_s_test_stack[:,0:80], net.x1_t_1: x_s_test_stack[:,80:160], net.x1_t_2: x_s_test_stack[:,160:], net.x2_: x_st_test_stack[:,:80], net.x3_: x_s_test_stack[:,:80], net.labels_: y_s_test_for_cnn_stack, net.lstm_label_: y_s_test_for_lstm_stack}
        feed_test_onTarget = {net.x1_t_0: x_t_test_stack[:,0:80], net.x1_t_1: x_t_test_stack[:,80:160], net.x1_t_2: x_t_test_stack[:,160:], net.x2_: x_st_test_stack[:,:80], net.x3_: x_t_test_stack[:,:80], net.labels_: y_t_test_for_cnn_stack, net.lstm_label_: y_t_test_for_lstm_stack}
        for iter in range(iteration):
            X_s_mb, y_s_cnn_mb, y_s_lstm_mb = train_s_dataset.next_batch(batch_size)
            X_t_mb, y_t_cnn_mb, y_t_lstm_mb = train_t_dataset.next_batch(batch_size)
            X_st_mb, y_st_cnn_mb, y_st_lstm_mb = train_st_dataset.next_batch(batch_size)

            # Feed dictionary
            feed_train = {net.x1_t_0: X_s_mb[:,0:80], net.x1_t_1: X_s_mb[:,80:160], net.x1_t_2: X_s_mb[:,160:], net.x2_: X_st_mb[:,:80], net.x3_: X_t_mb[:,:80], net.labels_: y_s_cnn_mb, net.lstm_label_ : y_s_lstm_mb}
            g_optimizer.run(feed_dict=feed_train)
            d_optimizer.run(feed_dict=feed_train)

            if iter % 10 == 0:

                loss, loss_rec, loss_cla, loss_g, loss_d, loss_lstm, acc_cnn_multi, acc_lstm = sess.run([net.loss, net.loss_rec, net.loss_cla, net.loss_g, net.loss_d, net.loss_lstm, net.acc_cnn, net.acc_lstm], feed_dict=feed_train)
                print("Iter: {}/{}, Train loss: {:6f}, Rec loss: {:6f}, CNN loss: {:6f}, G loss: {:6f}, D loss: {:6f}, LSTM loss: {:6f}, CNN Accuracy: {:6f}, LSTM Accuracy: {:6f}".format(iter, iteration, loss, loss_rec, loss_cla, loss_g, loss_d, loss_lstm, np.mean(acc_cnn_multi), acc_lstm))


