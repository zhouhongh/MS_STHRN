from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import copy
import utils
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


class FormatData(object):
    """
    Form train/validation data.
    形成字典
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, sample, train):

        total_frames = self.config.input_window_size + self.config.output_window_size
        # CMU sample [375,60]
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
        self.test_set = utils.normalize_data_dic(test_set, config.data_mean, config.data_std, config.dim_to_use)

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
        prefix, suffix = 50, 50

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
    def __init__(self, config, train=True):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass

class AMASSPredictionDataset(Dataset):
    def __init__(self, config, train=True):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
    def get_data(self):
        pass