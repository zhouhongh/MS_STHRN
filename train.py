from choose_dataset import DatasetChooser
from ST_HRN import ST_HRN
from HMR import HMR
from MS_STHRN import MS_STHRN
from torch.utils.data import DataLoader
import torch
import utils
import torch.optim as optim
import os
from utils import Progbar
from loss import loss as Loss
import numpy as np
import datetime
import scipy.io as sio
from plot_animation import plot_animation
import config
from argparse import ArgumentParser
from torchviz import make_dot
from tensorboardX import SummaryWriter
# python train.py --dataset Human --action walking
# os.environ["CUDA_VISIBLE_DEVICES"]='1'
def choose_net(config):

    if config.model == 'ST_HRN':
        net = ST_HRN(config)
    elif config.model == 'HMR':
        net = HMR(config)
    elif config.model == 'MS_STHRN':
        net = MS_STHRN(config)
    return net





def train(config, checkpoint_dir):
    print('Start Training the Model!')

    if config.longterm is True:
        config.output_window_size = 100

    # from choose_dataset import DatasetChooser
    choose = DatasetChooser(config)
    train_dataset, bone_length = choose(train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset, _ = choose(train=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    prediction_dataset, bone_length = choose(prediction=True)
    x_test, y_test, dec_in_test = prediction_dataset

    device = torch.device("cuda:"+str(config.device_ids[0]) if torch.cuda.is_available() else "cpu")
    print('Device {} will be used to save parameters'.format(device))
    torch.cuda.manual_seed_all(112858)

    net = choose_net(config)
    net.to(device)
    print('Total param number:' + str(sum(p.numel() for p in net.parameters())))
    print('Encoder param number:' + str(sum(p.numel() for p in net.encoder_cell.parameters())))
    print('Decoder param number:' + str(sum(p.numel() for p in net.decoder.parameters())))
    if torch.cuda.device_count() > 1:
        print("{} GPUs are usable!".format(str(torch.cuda.device_count())))
    net = torch.nn.DataParallel(net, device_ids=config.device_ids)

    if config.restore is True:
        dir = utils.get_file_list(checkpoint_dir)
        print('Load model from:' + checkpoint_dir + dir[-1])
        net.load_state_dict(torch.load(checkpoint_dir + dir[-1], map_location='cuda:0'))

    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    # Save model
    if not (os.path.exists(checkpoint_dir)):
        os.makedirs(checkpoint_dir)

    best_error = float('inf')
    best_error_list = None

    # log
    trainloss_list = []
    validloss_list = []
    Error_list = []

    time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('log file use this time as its name: '+time_now)
    if not (os.path.exists('./log')):
        os.makedirs('./log')

    # start training
    for epoch in range(config.max_epoch):
        print("At epoch:{}".format(str(epoch + 1)))
        prog = Progbar(target=config.training_size)
        prog_valid = Progbar(target=config.validation_size)

        # Train
        for it in range(config.training_size):
            for i, data in enumerate(train_loader, 0):
                encoder_inputs = data['encoder_inputs'].float().to(device)  ## 前t-1帧
                decoder_inputs = data['decoder_inputs'].float().to(device)  ## t-1到t-1+output_window_size帧
                decoder_outputs = data['decoder_outputs'].float().to(device)  ## t帧到以后的
                prediction = net(encoder_inputs, decoder_inputs, train=True)
                loss = Loss(prediction, decoder_outputs, bone_length, config)
                net.zero_grad()
                loss.backward()
                # g = make_dot(loss)
                # g.render('MSSTHRN_model', view=False)
                # [print(x.grad) for x in optimizer.param_groups[0]['params']]
                _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()

            prog.update(it + 1, [("Training Loss", loss.item())])
        trainloss_list.append(loss.item())

        # Valid
        with torch.no_grad():
            for it in range(config.validation_size):
                for j in range(3):
                    for i, data in enumerate(test_loader, 0):
                        if j == 0 and i == 0:
                            encoder_inputs = data['encoder_inputs'].float().to(device)
                            decoder_inputs = data['decoder_inputs'].float().to(device)
                            decoder_outputs = data['decoder_outputs'].float().to(device)
                        else:
                            encoder_inputs = torch.cat([data['encoder_inputs'].float().to(device), encoder_inputs],
                                                       dim=0)
                            decoder_inputs = torch.cat([data['decoder_inputs'].float().to(device), decoder_inputs],
                                                       dim=0)
                            decoder_outputs = torch.cat([data['decoder_outputs'].float().to(device), decoder_outputs],
                                                        dim=0)

                prediction = net(encoder_inputs, decoder_inputs, train=True)
                loss = Loss(prediction, decoder_outputs, bone_length, config)
                prog_valid.update(it + 1, [("Testing Loss", loss.item())])
            validloss_list.append(loss.item())

        # Test prediction
        actions = list(x_test.keys())
        y_predict = {}
        with torch.no_grad():
            for act in actions:
                x_test_ = torch.from_numpy(x_test[act]).float().to(device)
                dec_in_test_ = torch.from_numpy(dec_in_test[act]).float().to(device)
                pred = net(x_test_, dec_in_test_, train=False)
                pred = pred.cpu().numpy()
                y_predict[act] = pred

        error_actions = 0.0
        for act in actions:
            if config.datatype == 'lie':
                mean_error, _ = utils.mean_euler_error(config, act, y_predict[act],
                                                       y_test[act][:, :config.output_window_size, :])
                error = mean_error[[1, 3, 7, 9]]

            if config.datatype == 'smpl':
                mean_error, _ = utils.mean_euler_error(config, act, y_predict[act],
                                                       y_test[act][:, :config.output_window_size, :])
                error = mean_error[[1, 3, 7, 9]]
            # error_actions += error.mean()

            error_actions += mean_error.mean()

        error_actions /= len(actions)
        Error_list.append(error_actions)

        # save log
        np.savez('./log/' + config.model + '_' + time_now + '.npz', trainloss_list=trainloss_list, validloss_list=validloss_list,
                 Error_list=Error_list)

        if error_actions < best_error:
            print(error_actions)
            print(best_error)
            best_error_list = error
            best_error = error_actions
            torch.save(net.state_dict(), checkpoint_dir + 'Epoch_' + str(epoch + 1) + '.pth')
        print('Current best:' + str(round(best_error_list[0], 2))+ ' ' + str(round(best_error_list[1], 2)) +
                                    ' ' + str(round(best_error_list[2], 2)) + ' ' + str(round(best_error_list[3], 2)))



def prediction(config, checkpoint_dir, output_dir):

    print('Start testing the Model!')

    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir)
    print("Outputs saved to: " + output_dir)

    # 更改预测时输出的视频帧长
    config.output_window_size = 50

    choose = DatasetChooser(config)
    # This step is to get mean value, etc for unnorm
    _, _ = choose(train=True)

    if config.longterm is False:
        prediction_dataset, bone_length = choose(prediction=True)
        x_test, y_test, dec_in_test = prediction_dataset
        actions = list(x_test.keys())
    else:
        # get raw validation data because the test data isn't usable
        train_dataset, bone_length = choose(train=False)
        test_set = train_dataset.data
        x_test = {}
        y_test = {}
        dec_in_test = {}
        test_set = test_set[0]
        x_test[config.filename] = np.reshape(test_set[:config.input_window_size-1,:], [1, -1, config.input_size])
        y_test[config.filename] = np.reshape(test_set[config.input_window_size:, :], [1, -1, config.input_size])
        dec_in_test[config.filename] = np.reshape(test_set[config.input_window_size-1:-1, :], [1, -1, config.input_size])
        config.output_window_size = y_test[config.filename].shape[1]
        actions = [config.filename]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {} will be used to save parameters'.format(device))
    net = choose_net(config)
    net.to(device)
    print('Total param number:' + str(sum(p.numel() for p in net.parameters())))
    print('Encoder param number:' + str(sum(p.numel() for p in net.encoder_cell.parameters())))
    print('Decoder param number:' + str(sum(p.numel() for p in net.decoder.parameters())))

    net = torch.nn.DataParallel(net)
    dir = utils.get_file_list(checkpoint_dir)
    print('Load model from:' + checkpoint_dir + dir[-1]) # 使用最后保存的checkpoint
    ## 用GPU时记得改成 map_location='cuda:0',用cpu时是 map_location=torch.device('cpu')
    net.load_state_dict(torch.load(checkpoint_dir + dir[-1], map_location='cuda:0'))
    y_predict = {}
    with torch.no_grad():
        for act in actions:
            x_test_ = torch.from_numpy(x_test[act]).float().to(device)
            dec_in_test_ = torch.from_numpy(dec_in_test[act]).float().to(device)
            pred = net(x_test_, dec_in_test_, train=False)
            pred = pred.cpu().numpy()
            y_predict[act] = pred

    for act in actions:
        if config.datatype == 'lie':
            mean_error, _ = utils.mean_euler_error(config, act, y_predict[act], y_test[act])
            sio.savemat(output_dir + 'error_' + act + '.mat', dict([('error', mean_error)]))

            for i in range(y_predict[act].shape[0]):
                if config.dataset == 'Human':
                    y_p = utils.unNormalizeData(y_predict[act][i], config.data_mean, config.data_std, config.dim_to_ignore)
                    y_t = utils.unNormalizeData(y_test[act][i], config.data_mean, config.data_std, config.dim_to_ignore)
                #     expmap_all = utils.revert_coordinate_space(np.vstack((y_t, y_p)), np.eye(3), np.zeros(3))
                #     y_p = expmap_all[config.output_window_size:]
                #     y_t = expmap_all[:config.output_window_size]
                # else:
                #     y_p = y_predict[act][i]
                #     y_t = y_test[act][i]

                y_p_xyz = utils.fk(y_p, config, bone_length)
                y_t_xyz = utils.fk(y_t, config, bone_length)

                sio.savemat(output_dir + 'prediction_' + act + '_' + str(i) + '.mat', dict([('prediction', y_p)]))
                sio.savemat(output_dir + 'gt_' + act + '_' + str(i) + '.mat', dict([('gt', y_t)]))

                filename = act + '_' + str(i)
                if config.visualize:
                    # 红色是我的
                    pre_plot = plot_animation(y_p_xyz, y_t_xyz, config, filename)
                    pre_plot.plot()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--gpu", default=[1], help="GPU device ids")
    parser.add_argument("--training", default=True, dest="training", help="train or test")
    parser.add_argument("--action", type=str, default='all', dest="action", help="choose one action in the dataset:"
                                                                                 "h3.6m_actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting',"
                                                                                 "'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']"
                                                                                 "'all means all of the above")
    parser.add_argument("--dataset", type=str, required=True, dest="dataset",
                        help="choose dataset from 'Human' or 'AMASS'")
    parser.add_argument("--datatype", type=str, default='lie', dest="datatype", help="only lie is usable")
    parser.add_argument("--visualize", type=bool, default=False, dest="visualize",
                        help="visualize the prediction or not ")
    args = parser.parse_args()
    config = config.TrainConfig(args.dataset, args.datatype, args.action, args.gpu, args.training, args.visualize)
    checkpoint_dir, output_dir = utils.create_directory(config)
    if config.train_model is True:
        train(config, checkpoint_dir)
    else:
        prediction(config, checkpoint_dir, output_dir)
