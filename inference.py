import tensorflow as tf
from stn1d import _transform
class inference:

    def __init__(self, seq_len, n_classes_for_cnn, n_classes_for_lstm, n_channels, learning_rate, gamma, alpha, lambda_c, lambda_r, lambda_g, lambda_d, lambda_lstm, keep_prob):

        self.x1_t_0 = tf.placeholder(tf.float32, shape = [None, seq_len, 3])
        self.x1_t_1 = tf.placeholder(tf.float32, shape = [None, seq_len, 3])
        self.x1_t_2 = tf.placeholder(tf.float32, shape = [None, seq_len, 3])
        self.x2_ = tf.placeholder(tf.float32, shape = [None, seq_len, 3])
        self.x3_ = tf.placeholder(tf.float32, shape = [None, seq_len, 3])
        self.labels_ = tf.placeholder(tf.float32, [None, n_classes_for_cnn*3])
        self.lstm_label_ = tf.placeholder(tf.float32, [None, n_classes_for_lstm])

        self.gamma_ = gamma
        self.alpha_ = alpha
        self.lambda_c_ = lambda_c
        self.lambda_r_ = lambda_r
        self.lambda_g_ = lambda_g
        self.lambda_d_ = lambda_d
        self.lambda_lstm_ = lambda_lstm
        self.learning_rate_ = learning_rate
        self.seq_len_ = seq_len
        self.n_channels_ = n_channels
        self.n_classes_for_cnn_ = n_classes_for_cnn
        self.n_classes_for_lstm_ = n_classes_for_lstm
        self.keep_prob_ = keep_prob

        self.logits_, self.lstm_pred, self.stno2, self.x_reconstruct, self.d_fake, self.d_real = self.network()
        self.loss_rec = self.cost_reconstruction()
        self.loss_cla,self.score_split,self.label_split = self.cost_classification()
        self.loss_g = self.cost_generator()
        self.loss_d = self.cost_discriminator()
        self.loss_lstm = self.cost_lstm()
        self.loss = self.whole_loss()
        self.acc_cnn = self.accuracy_m(self.labels_, self.logits_)
        self.acc_lstm = self.accuracy(self.lstm_label_, self.lstm_pred)

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, mean=0., stddev=0.1)
        return tf.Variable(initial, name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name)

    def weight_variable_zeros(self, shape, name):
        initial = tf.zeros(shape)
        return tf.Variable(initial, name)

    def bias_variable_eyes(self, shape, name):
        initial = tf.reshape(tf.eye(shape),[-1])
        return tf.Variable(initial, name)

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def deconv2d(self, x, w, output_shape):
        return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool

    def max_unpool_1x2(self, x, shape):
        inference = tf.image.resize_nearest_neighbor(x, tf.stack([shape[1], shape[2] * 2]))
        return inference

    def stn1d_net(self, x_origin):
        print("============= stn1d_net =============")
        print("input data shape : %s" % x_origin.get_shape())

        W_n1 = self.weight_variable([1, 5, 3, 3], "W_n1")
        B_n1 = self.bias_variable([3], "B_n1")
        # h_n1 = max_out(tf.add(self.conv2d(x_origin, W_n1), B_n1),3)
        h_n1 = tf.nn.leaky_relu(tf.add(self.conv2d(x_origin, W_n1), B_n1))
        h_n_pool1 = self.max_pool_2x2(h_n1)
        print("h_n_pool1 shape : %s" % h_n_pool1.get_shape())

        W_n2 = self.weight_variable([1, 5, 3, 3], "W_n2")
        B_n2 = self.bias_variable([3], "B_n2")
        # h_n2 = max_out(tf.add(self.conv2d(h_n_pool1, W_n2), B_n2),3)
        h_n2 = tf.nn.leaky_relu(tf.add(self.conv2d(h_n_pool1, W_n2), B_n2))
        code_layer = self.max_pool_2x2(h_n2)
        print("code layer shape : %s" % code_layer.get_shape())

        code_origin = tf.reshape(code_layer, (-1, 20, 3))
        print("reshape code layer shape : %s" % code_origin.get_shape())

        x = tf.reshape(code_origin, [-1, 60])
        W_n3 = self.weight_variable([60, 30], "W_n3")
        B_n3 = self.bias_variable([30], "B_n3")

        W_n4 = self.weight_variable([30, 9], "W_n4")
        # B_n4 = self.bias_variable([9], "B_n4")

        # W_n4 = self.weight_variable_zeros([30, 9], "W_n4")
        B_n4 = self.bias_variable_eyes(3, "B_n4") #3^2 = 9

        # h_c1 = max_out((tf.add(tf.matmul(x, W_n3), B_n3)),30)
        h_c1 = tf.nn.leaky_relu((tf.add(tf.matmul(x, W_n3), B_n3)))
        print("h_n1 layer shape : %s" % h_c1.get_shape())
        # h_c2 = max_out(tf.add(tf.matmul(h_c1, W_n4), B_n4),9)
        h_c2 = tf.nn.leaky_relu(tf.add(tf.matmul(h_c1, W_n4), B_n4))
        print("h_c3 layer shape : %s" % h_c2.get_shape())
        theta = h_c2
        print("theta layer shape : %s" % theta.get_shape())

        out_size = (1, self.seq_len_)
        xyz_transformed, theta_r = _transform(theta, x_origin, out_size)

        print("xyz_transformed shape : %s" % xyz_transformed.get_shape())
        self.var_n = W_n1, B_n1, W_n2, B_n2, W_n3, B_n3, W_n4, B_n4

        self.theta_ = theta_r
        return xyz_transformed

    def share_net(self, x):

        x_origin = tf.reshape(x, [-1, 1, self.seq_len_, 3])
        print("input data shape : %s" % x_origin.get_shape())

        x_stn = self.stn1d_net(x_origin)

        W_s1 = self.weight_variable([1, 5, 3, 18], "W_s1")
        B_s1 = self.bias_variable([18], "B_s1")
        h_s1 = tf.nn.leaky_relu(tf.add(self.conv2d(x_stn, W_s1), B_s1))
        h_s_pool1 = self.max_pool_2x2(h_s1)
        print("h_s_pool1 shape : %s" % h_s_pool1.get_shape())

        W_s2 = self.weight_variable([1, 5, 18, 36], "W_s2")
        B_s2 = self.bias_variable([36], "B_s2")
        h_s2 = tf.nn.leaky_relu(tf.add(self.conv2d(h_s_pool1, W_s2), B_s2))
        code_layer = self.max_pool_2x2(h_s2)
        print("code layer shape : %s" % code_layer.get_shape())

        self.var_g = W_s1, B_s1, W_s2, B_s2

        return code_layer, x_stn

    def classifier_net(self, code_layer):
        code_origin = tf.reshape(code_layer, (-1, 60, 36))
        print("reshape code layer shape : %s" % code_origin.get_shape())

        x = tf.reshape(code_origin, [-1, 60*36])
        W_c1 = self.weight_variable([2160, 1080], "W_c1")
        B_c1 = self.bias_variable([1080], "B_c1")
        W_c2 = self.weight_variable([1080, 120], "W_c2")
        B_c2 = self.bias_variable([120], "B_c2")

        h_c1 = tf.nn.relu(tf.add(tf.matmul(x, W_c1), B_c1))
        print("h_c1 layer shape : %s" % h_c1.get_shape())

        h_c1_drop = tf.nn.dropout(h_c1, keep_prob=self.keep_prob_)

        h_c2 = tf.nn.relu(tf.add(tf.matmul(h_c1_drop, W_c2), B_c2))
        print("h_c2 layer shape : %s" % h_c2.get_shape())

        logits = tf.layers.dense(h_c2, 9)
        print("logits shape : %s" % logits.get_shape())

        self.var_c = W_c1, B_c1, W_c2, B_c2

        return logits

    def reconstruction_net(self, code_layer):

        W_r1 = self.weight_variable([1, 5, 18, 36], "W_r1")
        B_r1 = self.bias_variable([18], "B_r1")
        output_shape_r1 = tf.stack([tf.shape(self.x2_)[0], 1, 20, 18])
        h_r1 = tf.nn.leaky_relu(tf.add(self.deconv2d(code_layer, W_r1, output_shape_r1), B_r1))
        h_r_pool1 = self.max_unpool_1x2(h_r1, [-1, 1, 20, 18])
        print("h_r1 shape : %s" % h_r1.get_shape())

        W_r2 = self.weight_variable([1, 5, 3, 18], "W_r2")
        B_r2 = self.bias_variable([3], "B_r2")
        output_shape_r2 = tf.stack([tf.shape(self.x2_)[0], 1, 40, 3])
        h_r2 = tf.nn.leaky_relu(tf.add(self.deconv2d(h_r_pool1, W_r2, output_shape_r2), B_r2))
        print("h_r2 shape : %s" % h_r2.get_shape())
        x_reconstruct = self.max_unpool_1x2(h_r2, [-1, 1, 40, 3])
        print("reconstruct layer shape : %s" % x_reconstruct.get_shape())

        self.var_r = W_r1, B_r1, W_r2, B_r2
        return x_reconstruct

    def discriminator_net(self, x):
        x = tf.reshape(x, [-1, 20*36])
        W_d1 = self.weight_variable([720, 128], "W_d1")
        B_d1 = self.bias_variable([128], "B_d1")
        W_d2 = self.weight_variable([128, 1], "W_d2")
        B_d2 = self.bias_variable([1], "B_d2")

        h_d1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, W_d1), B_d1))
        print("h_d1 shape : %s" % h_d1.get_shape())
        h_d2 = tf.nn.sigmoid(tf.add(tf.matmul(h_d1, W_d2), B_d2))
        print("h_d2 shape : %s" % h_d2.get_shape())

        self.var_d = W_d1, B_d1, W_d2, B_d2
        return h_d2

    def lstm_net(self, x):
        x = tf.reshape(x, [-1, 3, 3])

        self.n_inputs = 3
        self.n_steps = 3
        self.n_hidden = 4

        W_l1 = self.weight_variable([self.n_inputs, self.n_hidden], "W_l1")
        B_l1 = self.bias_variable([self.n_hidden], "B_l1")
        W_l2 = self.weight_variable([self.n_hidden, self.n_classes_for_lstm_], "W_l2")
        B_l2 = self.bias_variable([self.n_classes_for_lstm_], "B_l2")

        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.n_inputs])

        # Linear activation
        x = tf.nn.leaky_relu(tf.matmul(x, W_l1) + B_l1)
        x = tf.split(x, self.n_steps, 0)

        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)

        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell_1, x, dtype=tf.float32)

        lstm_last_output = outputs[-1]

        self.var_l = W_l1, B_l1, W_l2, B_l2

        # Linear activation
        return tf.matmul(lstm_last_output, W_l2) + B_l2

    def network(self):
        with tf.variable_scope("share_net") as scope:
            self.o1_t_0, stno1_t_0 = self.share_net(self.x1_t_0)
            scope.reuse_variables()
            self.o1_t_1, stno1_t_1 = self.share_net(self.x1_t_1)
            scope.reuse_variables()
            self.o1_t_2, stno1_t_2 = self.share_net(self.x1_t_2)
            scope.reuse_variables()
            self.o2, stno2 = self.share_net(self.x2_)
            scope.reuse_variables()
            self.o3, stno3 = self.share_net(self.x3_)
        logits = self.classifier_net(tf.concat((self.o1_t_0, self.o1_t_1, self.o1_t_2), axis=2))

        x_reconstruct = self.reconstruction_net(self.o2)

        with tf.variable_scope("discriminator_net") as scope:
            d_real = self.discriminator_net(self.o1_t_0)
            scope.reuse_variables()
            d_fake = self.discriminator_net(self.o3)
        lstm_pred = self.lstm_net(logits)


        return logits, lstm_pred, stno2, x_reconstruct, d_fake, d_real

    def cost_classification(self):
        self.score = self.logits_
        score_split = tf.split(self.score,3,1)
        label_split = tf.split(self.labels_,3,1)
        total = 0.0
        for i in range(len(score_split)):
            total += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= score_split[i] ,labels= label_split[i]))
        return total/3,score_split,label_split

    def cost_reconstruction(self):
        x_origin = tf.reshape(self.x2_, [-1, 1, self.seq_len_, 3])
        loss = tf.reduce_mean(tf.pow(self.x_reconstruct - x_origin, 2))
        return loss

    def cost_generator(self):
        loss = -tf.reduce_mean(tf.log(self.d_fake))
        return loss

    def cost_discriminator(self):
        loss = -tf.reduce_mean(tf.log(self.d_real) + tf.log(1. - self.d_fake))
        loss = tf.multiply(loss, self.lambda_d_)
        return loss

    def cost_lstm(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.lstm_label_, logits=self.lstm_pred))
        return loss

    def whole_loss(self):
        cost = tf.multiply(self.loss_cla, self.lambda_c_)
        cost = tf.add(cost, tf.multiply(self.loss_rec, self.lambda_r_))
        cost = tf.add(cost, tf.multiply(self.loss_g, self.lambda_g_))
        cost = tf.add(cost, tf.multiply(self.loss_lstm, self.lambda_lstm_))
        return cost

    def accuracy(self, label, pred):
        # Accuracy
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def accuracy_m(self, label, pred):
        score_split = tf.split(pred, 3, 1)
        label_split = tf.split(label, 3, 1)

        correct_pred1 = tf.cast(tf.equal(tf.argmax(score_split[0], 1), tf.argmax(label_split[0], 1)), tf.float32)
        correct_pred2 = tf.cast(tf.equal(tf.argmax(score_split[1], 1), tf.argmax(label_split[1], 1)), tf.float32)
        correct_pred3 = tf.cast(tf.equal(tf.argmax(score_split[2], 1), tf.argmax(label_split[2], 1)), tf.float32)

        return correct_pred1, correct_pred2, correct_pred3