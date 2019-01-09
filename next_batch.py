import numpy as np

class Dataset:

    def __init__(self, data, label_for_cnn, label_for_lstm, x_s_test, y_s_test_for_cnn, y_s_test_for_lstm, shuffle):
        self._index_in_epoch = 2
        self._epochs_completed = 0
        self._data = data
        self._label_for_cnn = label_for_cnn
        self._label_for_lstm = label_for_lstm
        self._x_s_test = x_s_test
        self._y_s_test_for_cnn = y_s_test_for_cnn
        self._y_s_test_for_lstm = y_s_test_for_lstm
        self._shuffle = shuffle
        self._num_examples = data.shape[0]
        self._num_examples_test = x_s_test.shape[0]
        pass

    @property
    def data(self):
        return self._data
    def label_for_cnn(self):
        return self._label_for_cnn
    def label_for_lstm(self):
        return self._label_for_lstm

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            if self._shuffle == True:
                np.random.shuffle(idx)  # shuffle indexe
            self._data = self._data[idx]  # get list of `num` random samples
            self._label_for_cnn = self._label_for_cnn[idx]  # get list of `num` random samples
            self._label_for_lstm = self._label_for_lstm[idx]  # get list of `num` random samples

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples_1 = min(start-0 + batch_size, self._num_examples) - (start - 0)
            rest_num_examples_2 = min(start-1 + batch_size, self._num_examples) - (start - 1)
            rest_num_examples_3 = min(start-2 + batch_size, self._num_examples) - (start - 2)
            data_rest_part_1 = self._data[start-0:min(start-0 + batch_size, self._num_examples)]
            data_rest_part_2 = self._data[start-1:min(start-1 + batch_size, self._num_examples)]
            data_rest_part_3 = self._data[start-2:min(start-2 + batch_size, self._num_examples)]
            label_for_cnn_rest_part_1 = self._label_for_cnn[start-0:min(start-0 + batch_size, self._num_examples)]
            label_for_cnn_rest_part_2 = self._label_for_cnn[start-1:min(start-1 + batch_size, self._num_examples)]
            label_for_cnn_rest_part_3 = self._label_for_cnn[start-2:min(start-2 + batch_size, self._num_examples)]
            label_for_lstm_rest_part_1 = self._label_for_lstm[start-0:min(start-0 + batch_size, self._num_examples)]
            label_for_lstm_rest_part_2 = self._label_for_lstm[start-1:min(start-1 + batch_size, self._num_examples)]
            label_for_lstm_rest_part_3 = self._label_for_lstm[start-2:min(start-2 + batch_size, self._num_examples)]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            if self._shuffle == True:
                np.random.shuffle(idx0)  # shuffle indexes
            self._data = self._data[idx0]  # get list of `num` random samples
            self._label_for_cnn = self._label_for_cnn[idx0]  # get list of `num` random samples
            self._label_for_lstm = self._label_for_lstm[idx0]  # get list of `num` random samples

            # print('res',start,'-',self._num_examples,'->',rest_num_examples_1)
            # print('res',start-1,'-',min(start-1 + batch_size, self._num_examples),'->',rest_num_examples_2)
            # print('res',start-2,'-',min(start-2 + batch_size, self._num_examples),'->',rest_num_examples_3)

            start = 0
            self._index_in_epoch = 2 # Reset index start from 2
            self._index_in_epoch_1 = batch_size - rest_num_examples_1
            self._index_in_epoch_2 = batch_size - rest_num_examples_2
            self._index_in_epoch_3 = batch_size - rest_num_examples_3
            end_1 =  self._index_in_epoch_1
            end_2 =  self._index_in_epoch_2
            end_3 =  self._index_in_epoch_3

            # print('new',start,'-',end_1,'->',self._index_in_epoch)
            # print('new',start,'-',end_2,'->',self._index_in_epoch_2)
            # print('new',start,'-',end_3,'->',self._index_in_epoch_3)

            data_new_part_1 = np.array(self._data[start:end_1])
            data_new_part_2 = np.array(self._data[start:end_2])
            data_new_part_3 = np.array(self._data[start:end_3])
            label_for_cnn_new_part_1 = np.array(self._label_for_cnn[start:end_1])
            label_for_cnn_new_part_2 = np.array(self._label_for_cnn[start:end_2])
            label_for_cnn_new_part_3 = np.array(self._label_for_cnn[start:end_3])
            label_for_lstm_new_part_1 = np.array(self._label_for_lstm[start:end_1])
            label_for_lstm_new_part_2 = np.array(self._label_for_lstm[start:end_2])
            label_for_lstm_new_part_3 = np.array(self._label_for_lstm[start:end_3])

            c_data_stack_3 = np.concatenate((np.concatenate((np.concatenate((data_rest_part_1, data_new_part_1), axis=0), np.concatenate((data_rest_part_2, data_new_part_2), axis=0)), axis=1), np.concatenate((data_rest_part_3, data_new_part_3), axis=0)), axis=1).astype('f')
            c_label_for_cnn_stack_3 = np.concatenate((np.concatenate((np.concatenate((label_for_cnn_rest_part_1, label_for_cnn_new_part_1), axis=0), np.concatenate((label_for_cnn_rest_part_2, label_for_cnn_new_part_2), axis=0)), axis=1), np.concatenate((label_for_cnn_rest_part_3, label_for_cnn_new_part_3), axis=0)), axis=1).astype('f')
            # c_label_for_lstm_stack_3 = np.concatenate((np.concatenate((np.concatenate((label_for_lstm_rest_part_1, label_for_lstm_new_part_1), axis=0), np.concatenate((label_for_lstm_rest_part_2, label_for_lstm_new_part_2), axis=0)), axis=1), np.concatenate((label_for_lstm_rest_part_3, label_for_lstm_new_part_3), axis=0)), axis=1).astype('f')
            return c_data_stack_3, c_label_for_cnn_stack_3, np.concatenate((label_for_lstm_rest_part_1, label_for_lstm_new_part_1), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            _data_stack_3 = np.concatenate((np.concatenate((self._data[start:end], self._data[start - 1:end - 1]), axis=1),self._data[start - 2:end - 2]), axis=1).astype('f')
            _label_for_cnn_stack_3 = np.concatenate((np.concatenate((self._label_for_cnn[start:end], self._label_for_cnn[start - 1:end - 1]), axis=1),self._label_for_cnn[start - 2:end - 2]), axis=1).astype('f')
            # _label_for_lstm_stack_3 = np.concatenate((np.concatenate((self._label_for_lstm[start:end], self._label_for_lstm[start - 1:end - 1]), axis=1),self._label_for_lstm[start - 2:end - 2]), axis=1).astype('f')
            return _data_stack_3, _label_for_cnn_stack_3, self._label_for_lstm[start:end]

    def test_stack(self):
        start = 2
        end = self._num_examples_test
        _data_stack_3 = np.concatenate((np.concatenate((self._x_s_test[start:end], self._x_s_test[start - 1:end - 1]), axis=1),self._x_s_test[start - 2:end - 2]), axis=1).astype('f')
        _label_for_cnn_stack_3 = np.concatenate((np.concatenate((self._y_s_test_for_cnn[start:end], self._y_s_test_for_cnn[start - 1:end - 1]), axis=1),self._y_s_test_for_cnn[start - 2:end - 2]), axis=1).astype('f')
        return _data_stack_3, _label_for_cnn_stack_3, self._y_s_test_for_lstm[start:end]
# Reference
# https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data