# implemented by JunfengHu
# create time: 7/20/2019
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import copy
import utils
import os


class FormatDataPre(object):
    """
    Form prediction(test) data.
    """

    def __init__(self):
        pass

    def __call__(self, x_test, y_test):
        dec_in_test = x_test[-1:, :]
        x_test = x_test[:-1, :]
        return {'x_test': x_test, 'dec_in_test': dec_in_test, 'y_test': y_test}


class FormatData(object):
    """
    Form train/validation data.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, sample, train):

        total_frames = self.config.input_window_size + self.config.output_window_size

        video_frames = sample.shape[0]
        idx = np.random.randint(1, video_frames - total_frames) #在可选范围中随机挑选帧起始点

        data_seq = sample[idx:idx + total_frames, :]
        encoder_inputs = data_seq[:self.config.input_window_size - 1, :]
        # 最后一个弃掉了,这里代码还可以精简
        if train:
            decoder_inputs = data_seq[self.config.input_window_size - 1:
                                      self.config.input_window_size - 1 + self.config.output_window_size, :]
        else:
            decoder_inputs = data_seq[self.config.input_window_size - 1:self.config.input_window_size, :]
        decoder_outputs = data_seq[self.config.input_window_size:, :]
        return {'encoder_inputs': encoder_inputs, 'decoder_inputs': decoder_inputs, 'decoder_outputs': decoder_outputs}


class LieTsfm(object):
    """
    This class is redundant and could be integrated into dataset class. However, we didn't do that due to some historical events.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        rawdata = sample

        data = rawdata[:, :-1, :3].reshape(rawdata.shape[0], -1)
        return data


class HumanDataset(Dataset):

    def __init__(self, config, train=True):

        self.config = config
        self.train = train
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatData(config)
        if config.datatype == 'lie':
            if train:
                train_path = './data/h3.6m/Train/train_lie'
            else:
                train_path = './data/h3.6m/Test/test_lie'
        elif config.datatype == '':
            train_path = './data/h3.6m/Train/train_xyz'
        if train:
            subjects = ['S1', 'S6', 'S7', 'S8', 'S9', 'S11']
        else:
            subjects = ['S5']

        if config.filename == 'all':
            actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting',
                       'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']
        else:
            actions = [config.filename]

        set = []
        complete_train = []
        for id in subjects:
            for action in actions:
                for i in range(2):
                    if config.datatype == 'lie':
                        filename = '{0}/{1}_{2}_{3}_lie.mat'.format(train_path, id, action, i + 1)
                        rawdata = sio.loadmat(filename)['lie_parameters']
                        set.append(rawdata)
                    elif config.datatype == 'xyz':
                        filename = '{0}/{1}_{2}_{3}_xyz.mat'.format(train_path, id, action, i + 1)
                        rawdata = sio.loadmat(filename)['joint_xyz']
                        set.append(rawdata.reshape(rawdata.shape[0], -1))

                if len(complete_train) == 0:
                    complete_train = copy.deepcopy(set[-1])
                else:
                    complete_train = np.append(complete_train, set[-1], axis=0)

        if not train and config.data_mean is None:
            print('Load train dataset first!')

        if train and config.datatype == 'lie':
            data_mean, data_std, dim_to_ignore, dim_to_use = utils.normalization_stats(complete_train)
            config.data_mean = data_mean
            config.data_std = data_std
            config.dim_to_ignore = dim_to_ignore
            config.dim_to_use = dim_to_use

        set = utils.normalize_data(set, config.data_mean, config.data_std, config.dim_to_use)
        # [S_num, frame_for_S, 54]
        self.data = set

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        if self.config.datatype == 'lie':
            pass
        elif self.config.datatype == 'xyz':
            pass
        sample = self.formatdata(self.data[idx], False)
        return sample

class HumanPredictionDataset(object):

    def __init__(self, config):
        self.config = config
        if config.filename == 'all':
            self.actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases',
                            'sitting', 'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog',
                            'walkingtogether']
        else:
            self.actions = [config.filename]

        test_set = {}
        for subj in [5]:
            for action in self.actions:
                for subact in [1, 2]:
                    if config.datatype == 'lie':
                        filename = '{0}/S{1}_{2}_{3}_lie.mat'.format('./data/h3.6m/Test/test_lie', subj, action, subact)
                        test_set[(subj, action, subact)] = sio.loadmat(filename)['lie_parameters']

                    if config.datatype == 'xyz':
                        filename = '{0}/S{1}_{2}_{3}_xyz.mat'.format('./data/h3.6m/Test/test_xyz', subj, action, subact)
                        test_set[(subj, action, subact)] = sio.loadmat(filename)['joint_xyz']
                        test_set[(subj, action, subact)] = test_set[(subj, action, subact)].reshape(
                            test_set[(subj, action, subact)].shape[0], -1)
        try:
            config.data_mean
        except NameError:
            print('Load  train set first!')
        self.test_set = utils.normalize_data_dir(test_set, config.data_mean, config.data_std, config.dim_to_use)

    def get_data(self):
        x_test = {}
        y_test = {}
        dec_in_test = {}
        for action in self.actions:
            encoder_inputs, decoder_inputs, decoder_outputs = self.get_batch_srnn(self.config, self.test_set, action,
                                                                                  self.config.output_window_size)
            x_test[action] = encoder_inputs
            y_test[action] = decoder_outputs
            dec_in_test[action] = np.zeros([decoder_inputs.shape[0], 1, decoder_inputs.shape[2]])
            dec_in_test[action][:, 0, :] = decoder_inputs[:, 0, :]
        return [x_test, y_test, dec_in_test]

    def get_batch_srnn(self, config, data, action, target_seq_len):
        # Obtain SRNN test sequences using the specified random seeds

        frames = {}
        frames[action] = self.find_indices_srnn(data, action)

        batch_size = 8
        subject = 5
        source_seq_len = config.input_window_size

        seeds = [(action, (i % 2) + 1, frames[action][i]) for i in range(batch_size)]

        encoder_inputs = np.zeros((batch_size, source_seq_len - 1, config.input_size), dtype=float)
        decoder_inputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float)

        for i in range(batch_size):
            _, subsequence, idx = seeds[i]
            idx = idx + 50

            data_sel = data[(subject, action, subsequence)]

            data_sel = data_sel[(idx - source_seq_len):(idx + target_seq_len), :]

            encoder_inputs[i, :, :] = data_sel[0:source_seq_len - 1, :]  # x_test
            decoder_inputs[i, :, :] = data_sel[source_seq_len - 1:(source_seq_len + target_seq_len - 1),
                                      :]  # decoder_in_test
            decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]  # y_test

        return [encoder_inputs, decoder_inputs, decoder_outputs]

    def find_indices_srnn(self, data, action):

        """
        Obtain the same action indices as in SRNN using a fixed random seed
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py
        """

        SEED = 1234567890
        rng = np.random.RandomState(SEED)

        subject = 5
        subaction1 = 1
        subaction2 = 2

        T1 = data[(subject, action, subaction1)].shape[0]
        T2 = data[(subject, action, subaction2)].shape[0]
        prefix, suffix = 50, 100

        idx = []
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))

        return idx


class AMASSDataset(Dataset):
    """
     the directory structure of this dataset:
     train/
        -action/
            -subject/
     test/
        -action/
            -subject/
    """

    def __init__(self, config, train=True):

        self.config = config
        self.train = train
        self.formatdata = FormatData(config)
        if train:
            subjects = os.listdir('{0}/{1}/{2}'.format(config.data_root, 'train', config.filename))
        else:
            subjects = os.listdir('{0}/{1}/{2}'.format(config.data_root, 'test', config.filename))

        set = []
        complete_train = []
        for sub in subjects:
            if train:
                folderdir = '{0}/{1}/{2}/{3}'.format(config.data_root, 'train', config.filename, sub)
            else:
                folderdir = '{0}/{1}/{2}/{3}'.format(config.data_root, 'test', config.filename, sub)
            for file in os.listdir(folderdir):
                filedir = '{0}/{1}'.format(folderdir, file)
                rawdata = np.load(filedir)['poses'][:, :66]
                rawdata = self.frame_filter(rawdata)
                # 去除帧太少的序列
                if rawdata.shape[0] > 150:
                    set.append(rawdata)
            if len(complete_train) == 0:
                complete_train = copy.deepcopy(set[-1]) #每个subjects取最后一个动作序列计算均值方差
            else:
                complete_train = np.append(complete_train, set[-1], axis=0)
        if train:
            print('video num for training：', len(set))
        else:
            print('video num for test：', len(set))
        if not train and config.data_mean is None:
            print('Load train dataset first!')
        if train:
            data_mean, data_std, dim_to_ignore, dim_to_use = utils.normalization_stats(complete_train)
            config.data_mean = data_mean
            config.data_std = data_std
            config.dim_to_ignore = dim_to_ignore
            config.dim_to_use = dim_to_use

        set = utils.normalize_data(set, config.data_mean, config.data_std, config.dim_to_use)
        # [S_num, frame_for_S, 60]
        self.data = set
    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.formatdata(self.data[idx], False)
        return sample
    def frame_filter(self, rawdata):
        '''
        author: zhouhonghong
        过滤掉前后的静止画面
        :return:
        '''
        forward_frame = rawdata[0, :]
        remain_id = []
        for id in range(rawdata.shape[0] - 1):
            this_frame = rawdata[id + 1, :]
            if np.sum(np.abs(this_frame - forward_frame)) > 0.1: # 变化阈值，小于0.1视为静止
                remain_id.append(id + 1)
            forward_frame = this_frame
        start_id = remain_id[0]
        end_id = remain_id[-1]
        if abs(start_id-end_id) > 30:
            return rawdata[start_id:end_id-30, :]
        else:
            return rawdata[start_id, :]




class AMASSPredictionDataset(object):

    def __init__(self, config):
        self.config = config
        self.action = config.filename
        test_set = {}
        self.file_names = [[],] # 列表，顺序记录各subject文件夹下的数据文件名称
        subs = os.listdir('{0}/{1}/{2}'.format(config.data_root, 'test', self.action))
        for sub in subs:
            folderdir = '{0}/{1}/{2}/{3}'.format(config.data_root, 'test', self.action, sub)
            for filename in os.listdir(folderdir):
                filedir = '{0}/{1}'.format(folderdir, filename)
                test_set[(sub, filename)] = np.load(filedir)['poses'][:, :66]

        try:
            config.data_mean
        except NameError:
            print('Load  train set first!')

        self.test_set = utils.normalize_data_dir(test_set, config.data_mean, config.data_std, config.dim_to_use)

    def get_data(self):
        x_test = {}
        y_test = {}
        dec_in_test = {}
        encoder_inputs, decoder_inputs, decoder_outputs = self.get_batch_srnn(self.config, self.test_set,
                                                                                  self.config.output_window_size)
        x_test[self.action] = encoder_inputs
        y_test[self.action] = decoder_outputs
        dec_in_test[self.action] = np.zeros([decoder_inputs.shape[0], 1, decoder_inputs.shape[2]])
        dec_in_test[self.action][:, 0, :] = decoder_inputs[:, 0, :]
        return [x_test, y_test, dec_in_test]

    def get_batch_srnn(self, config, data, target_seq_len):
        # Obtain SRNN test sequences using the specified random seeds

        frames = {}
        frames[self.action] = self.find_indices_srnn(data)

        batch_size = 4 ##  不大于测试集视频数目
        source_seq_len = config.input_window_size

        seeds = [(frames[self.action][i]) for i in range(batch_size)]

        encoder_inputs = np.zeros((batch_size, source_seq_len - 1, config.input_size), dtype=float)
        decoder_inputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float)

        for i in range(batch_size):
            idx = seeds[i]
            idx = idx + source_seq_len

            data_sel = data[self.keys[i]]

            data_sel = data_sel[(idx - source_seq_len):(idx + target_seq_len), :]

            encoder_inputs[i, :, :] = data_sel[0:source_seq_len - 1, :]  # x_test
            decoder_inputs[i, :, :] = data_sel[source_seq_len - 1:(source_seq_len + target_seq_len - 1), :]  # decoder_in_test
            decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]  # y_test

        return [encoder_inputs, decoder_inputs, decoder_outputs]

    def find_indices_srnn(self, data):

        """
        Obtain the same action indices as in SRNN using a fixed random seed
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py
        """

        SEED = 1234567890
        rng = np.random.RandomState(SEED)
        prefix, suffix = 50,  100#  腾出前面的输入帧和后面的预测帧

        idx = []
        self.keys = []
        for key in data.keys():
            idx.append(rng.randint(0, data[key].shape[0] - prefix - suffix))
            self.keys.append(key)

        return idx
