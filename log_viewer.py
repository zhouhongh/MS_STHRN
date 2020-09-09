import utils
import numpy as np
# log_file = utils.get_file_list('./log/')
# log = np.load('./log/'+log_file[-1])
# trainloss_list = log['trainloss_list']
# validloss_list = log['validloss_list']
# Error_list = log['Error_list']
# print('train loss', trainloss_list)
# print('valid_loss', validloss_list)
# print('error', Error_list)

# test
path = '/mnt/DataDrive164/zhouhonghong/AMASS_selected/train/walk/KIT_9/walking_medium01_poses.npz'
data = np.load(path)

print(data.files)
print('fps', data['mocap_framerate'])
poses = data['poses']
# print('poses', poses)
trans = data['trans']
print(trans)
