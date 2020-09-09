import numpy as np
import torch
import scipy.io as sio
import load_data as loader


class DatasetChooser(object):

    def __init__(self, config):
        self.config = config
        self.dataset = config.dataset

    def choose_dataset(self, train=True, prediction=False):

        if not prediction:
            if self.config.datatype == 'lie':
                if self.dataset == 'Human':
                    bone_length_path = None
                    data = loader.HumanDataset(self.config, train=train)
                    self.config.input_size = data[0]['encoder_inputs'].shape[1]
                elif self.dataset == 'AMASS':
                    bone_length_path = None
                    data = loader.AMASSDataset(self.config, train=train)
                    self.config.input_size = data[0]['encoder_inputs'].shape[1]

        else:
            if self.config.datatype == 'lie':
                if self.dataset == 'Human':
                    bone_length_path = None
                    data_loader = loader.HumanPredictionDataset(self.config)
                    data = data_loader.get_data()
                    self.config.input_size = data[0][list(data[0].keys())[0]].shape[2]
                elif self.dataset == 'AMASS':
                    bone_length_path = None
                    data_loader = loader.AMASSPredictionDataset(self.config)
                    data = data_loader.get_data()
                    self.config.input_size = data[0][list(data[0].keys())[0]].shape[2]


        if bone_length_path is not None:
            rawdata = sio.loadmat(bone_length_path)
            rawdata = rawdata[list(rawdata.keys())[3]]
            bone = self.cal_bone_length(rawdata)
        else:
            bone = self.config.bone_length

        return data, bone

    def __call__(self, train=True, prediction=False):
        return self.choose_dataset(train, prediction)

    def cal_bone_length(self, rawdata):

        njoints = rawdata.shape[1]
        bone = np.zeros([njoints, 3])
        if self.config.datatype == 'lie':
            for i in range(njoints):
                bone[i, 0] = round(rawdata[0, i, 3], 2)
            # delete zero in bone, n joints mean n-1 bones
            bone = bone[1:, :]

        elif self.config.datatype == 'xyz':
            for i in range(njoints):
                bone[i, 0] = round(np.linalg.norm(rawdata[0, i, :] - rawdata[0, i - 1, :]), 2)

        return bone