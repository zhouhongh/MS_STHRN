#############################################################################
# Input: 3D joint locations
# Plot out the animated motion
#############################################################################

import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def plot_animation(predict, labels, config, filename):
    if config.dataset == 'Human':
        predict_plot = plot_h36m(predict, labels, config, filename)
    elif config.dataset == 'AMASS':
        predict_plot = plot_amass(predict, labels, config)

    return predict_plot

class plot_h36m(object):

    def __init__(self, predict, labels, config, filename):
        self.joint_xyz = labels
        self.nframes = labels.shape[0]
        self.joint_xyz_f = predict
        # create
        self.folder_dir = '/' + config.filename + '/'
        self.folder_dir += 'inputWindow=' + str(config.input_window_size) + '_outputWindow=' + str(
            config.output_window_size) + '/'
        if config.model == 'HMR':
            self.folder_dir = './GIF/HMR/' + self.folder_dir
        elif config.model == 'ST_HRN':
            self.folder_dir = './GIF/ST_HRN/' + self.folder_dir
        elif config.model == 'MS_STHRN':
            self.folder_dir = './GIF/MS_STHRN/' + self.folder_dir
        if not os.path.exists(self.folder_dir):  # 如果路径不存在
            os.makedirs(self.folder_dir)

        matplotlib.rc('axes', edgecolor=(1.0, 1.0, 1.0, 0.0))
        # set up the axes
        xmin = -750
        xmax = 750
        ymin = -750
        ymax = 750
        zmin = -750
        zmax = 750

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.grid(False)
        self.ax.w_xaxis.set_pane_color((1.0, 1.0, 0.0, 0.0))
        self.ax.w_yaxis.set_pane_color((1.0, 1.0, 0.0, 0.0))
        #self.ax.w_zaxis.set_pane_color((1.0, 1.0, 0.0, 0.0))
        #self.ax.set_xlabel('x')
        #self.ax.set_ylabel('y')
        #self.ax.set_zlabel('z')
        self.ax.set_xticks([])
        self.ax.set_zticks([])
        self.ax.set_yticks([])

        self.chain = config.chain_config
        # self.chain = [np.array([0, 1, 2, 3, 4, 5]),
        #              np.array([0, 6, 7, 8, 9, 10]),
        #              np.array([0, 12, 13, 14, 15]),
        #              np.array([13, 17, 18, 19, 22, 19, 21]),
        #              np.array([13, 25, 26, 27, 30, 27, 29])]
        self.scats = []
        self.lns = []
        self.filename = filename

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)

        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])

        xdata_f = np.squeeze(self.joint_xyz_f[frame, :, 0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame, :, 1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame, :, 2])

        for i in range(len(self.chain)):
            self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='#f94e3e')) # red: prediction
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],], linewidth=2.0, color='#0780ea')) # blue: ground truth

    def plot(self):

        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=100, repeat=False)
        plt.title(self.filename, fontsize=16)
        ani.save(self.folder_dir + self.filename + '.gif', writer='pillow')

        plt.close()
        # plt.show()

class plot_amass(object):
    pass