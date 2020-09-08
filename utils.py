import sys
import time
import numpy as np
import copy
import torch
import os


def normalization_stats(completeData):
    """
    Copied from https://github.com/una-dinosauria/human-motion-prediction
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return [data_mean, data_std, dimensions_to_ignore, dimensions_to_use]

def normalize_data(data, data_mean, data_std, dim_to_use):
    """
    Copied and modified from https://github.com/una-dinosauria/human-motion-prediction
    """
    data_out = []

    for idx in range(len(data)):
        data_out.append(np.divide((data[idx] - data_mean), data_std))
        data_out[-1] = data_out[-1][:, dim_to_use]

    return data_out

def normalize_data_dic(data, data_mean, data_std, dim_to_use):
    """
    Copied from https://github.com/una-dinosauria/human-motion-prediction
    """
    data_out = {}

    for key in data.keys():
        data_out[key] = np.divide((data[key] - data_mean), data_std)
        data_out[key] = data_out[key][:, dim_to_use]

    return data_out

def prepare_loss(data, length, dim_to_ignore):
    """
    recover ignore dimension in data to calculate lie loss
    :param data: prediction data
    :param length: length of one single human pose. 99 for h3.6m dataset
    :param dim_to_ignore: get from function normalization_stats
    :return: recovered data
    """
    origData = torch.zeros([data.shape[0], data.shape[1], length], device=data.device)
    dimensions_to_use = []
    for i in range(length):
        if i in dim_to_ignore:
            continue
        dimensions_to_use.append(i)

    origData[:, :,dimensions_to_use] = data
    return origData[:, :, 3:]

def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore):
    """
    Copied from https://github.com/una-dinosauria/human-motion-prediction
    """
    #  去标准化并加上为0的dim

    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    origData[:, dimensions_to_use] = normalizedData

    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData



def mean_euler_error(config, action, y_predict, y_test):
    n_batch = y_predict.shape[0]
    nframes = y_predict.shape[1]

    mean_errors = np.zeros([n_batch, nframes])
    for i in range(n_batch):
        for j in range(nframes):
            if config.dataset == 'Human':
                pred = unNormalizeData(y_predict[i], config.data_mean, config.data_std, config.dim_to_ignore)
                gt = unNormalizeData(y_test[i], config.data_mean, config.data_std, config.dim_to_ignore)
            else:
                pred = copy.deepcopy(y_predict[i])
                gt = copy.deepcopy(y_test[i])
        pred[:, 0:3] = 0
        gt[:, 0:3] = 0

        idx_to_use = np.where(np.std(gt, 0) > 1e-4)[0]
        euc_error = np.power(gt[:, idx_to_use] - pred[:, idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt(euc_error)
        mean_errors[i, :] = euc_error

    mme = np.mean(mean_errors, 0)

    print("\n" + action)
    toprint_idx = np.array([1, 3, 7, 9, 13, 15, 17, 24])
    idx = np.where(toprint_idx < len(mme))[0]
    toprint_list = ["& {:.3f} ".format(mme[toprint_idx[i]]) for i in idx]
    print("".join(toprint_list))

    mme_mean = np.mean(mme[toprint_idx[idx]])
    return mme, mme_mean

def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x))) #按照最近文件修改时间进行升序排序
        # print(dir_list)
        return dir_list


def create_directory(config):

    """
    crate Checkpoint directory path
    modified from https://github.com/BII-wushuang/Lie-Group-Motion-Prediction
    :param config:
    :return:
    """
    folder_dir = config.dataset + '/' + config.datatype + '_' + config.loss + 'loss_' + config.model
    if config.model == 'HMR':
        folder_dir += '_RecurrentSteps=' + str(config.encoder_recurrent_steps) + '_' + 'ContextWindow=' + str(
            config.context_window) + '_' + 'hiddenSize=' + str(config.hidden_size)
    if config.model == 'ST_HRN':
        folder_dir += '_RecurrentSteps=' + str(config.encoder_recurrent_steps) + '_hiddenSize=' + str(config.hidden_size) \
                      + '_decoder_name=' + str(config.decoder)

    folder_dir += '/' + config.filename + '/'
    folder_dir += 'inputWindow=' + str(config.input_window_size) + '_outputWindow=' + str(
        config.output_window_size) + '/'


    if config.model == 'HMR':
        checkpoint_dir = './checkpoint/HMR/' + folder_dir
        output_dir = './output/HMR/' + folder_dir
    elif config.model == 'ST_HRN':
        checkpoint_dir = './checkpoint/ST_HRN/' + folder_dir
        output_dir = './output/ST_HRN/' + folder_dir
    elif config.model == 'MS_STHRN':
        checkpoint_dir = './checkpoint/MS_STHRN/' + folder_dir
        output_dir = './output/MS_STHRN/' + folder_dir

    if not os.path.exists(checkpoint_dir):  # 如果路径不存在
        os.makedirs(checkpoint_dir)
    if not os.path.exists(output_dir):  # 如果路径不存在
        os.makedirs(output_dir)

    return [checkpoint_dir, output_dir]


class Progbar(object):
    """Progbar class copied from https://github.com/fchollet/keras/

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """
    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)