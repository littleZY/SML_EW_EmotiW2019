#
#   Hello World server in Python
#   Binds REP socket to tcp://*:7331
#   Expects b"Hello" from client, replies with b"World"
#
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import time
import zmq
import md_config as cfg
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, CuDNNLSTM, Dense, TimeDistributed, GlobalAveragePooling1D, Activation, \
    BatchNormalization
import h5py
import copy

def define_model(hparams):
    current_n_lstms = hparams['NUM_LSTM_LAYERS']
    current_lstm_units = hparams['LSTM_UNITS']
    current_n_denses = hparams['NUM_DENSE_LAYERS']
    current_dense_units = hparams['DENSE_UNITS']
    current_dropout_rates = hparams['DROPOUT_RATES']
    current_time_step = hparams['TIME_STEP']
    current_input_units = hparams['INPUT_UNITS']
    current_densen_act = hparams['ACTIVATION_F']

    model = Sequential()
    if hparams['FC1'][1] > 0:
        model.add(TimeDistributed(Dense(hparams['FC1'][1], activation='relu'),
                                  input_shape=(current_time_step, hparams['FC1'][0])))

    model.add(LSTM(current_lstm_units[0], return_sequences=True, input_shape=(current_time_step, current_input_units),
                  stateful=False))
        # CuDNNLSTM(current_lstm_units[0], return_sequences=True, input_shape=(current_time_step, current_input_units),
        #           stateful=False))

    if current_n_lstms > 1:
        for idx in range(1, current_n_lstms):
            model.add(LSTM(current_lstm_units[idx], return_sequences=True))
            # model.add(CuDNNLSTM(current_lstm_units[idx], return_sequences=True))

    for idx in range(current_n_denses):
        model.add(TimeDistributed(Dense(current_dense_units[idx], activation='relu')))
        # model.add(TimeDistributed(Dropout(0.3)))

    model.add(TimeDistributed(Dense(1, activation=current_densen_act)))
    model.add(GlobalAveragePooling1D())

    return model

def load_weights_to_model(current_model, hparams, ft_type):
    """ Only apply to the LSTM model in this file, for other models, try to change :v"""
    f = h5py.File('./models/{}_{}_models_{}_{}_0_epochs{}_best_weight.h5'.format(hparams['model_path'], ft_type,
                                                                                 hparams['n_segments'],
                                                                                 hparams['alpha'],
                                                                                 hparams['EPOCHS']), 'r')
    print(list(f.keys()))

    # tmp2 = current_model.layers[6].get_weights()

    current_model.layers[0].set_weights([f['time_distributed_2']['time_distributed_2']['kernel:0'].value,
                                         f['time_distributed_2']['time_distributed_2']['bias:0'].value])
    current_model.layers[1].set_weights(
        [f['cu_dnnlstm']['cu_dnnlstm']['kernel:0'].value, f['cu_dnnlstm']['cu_dnnlstm']['recurrent_kernel:0'].value,
         f['cu_dnnlstm']['cu_dnnlstm']['bias:0'].value])
    current_model.layers[2].set_weights([f['cu_dnnlstm_1']['cu_dnnlstm_1']['kernel:0'].value,
                                         f['cu_dnnlstm_1']['cu_dnnlstm_1']['recurrent_kernel:0'].value,
                                         f['cu_dnnlstm_1']['cu_dnnlstm_1']['bias:0'].value])
    current_model.layers[3].set_weights([f['time_distributed']['time_distributed']['kernel:0'].value,
                                         f['time_distributed']['time_distributed']['bias:0'].value])
    current_model.layers[4].set_weights([f['time_distributed_1']['time_distributed_1']['kernel:0'].value,
                                         f['time_distributed_1']['time_distributed_1']['bias:0'].value])
    current_model.layers[5].set_weights([f['time_distributed_3']['time_distributed_3']['kernel:0'].value,
                                         f['time_distributed_3']['time_distributed_3']['bias:0'].value])

    f.close()
    return current_model

def get_gaze_features(raw_input):
    """
    Get gaze features from raw input
    :param raw_input:
    :return:
    """
    # Get statiscal feature from raw input
    gaze_direction = raw_input[:, 5:11]
    gaze_angle = raw_input[:, 11: 13]
    eye_landmark2D = raw_input[:, 13: 125]
    eye_landmark3D = raw_input[:, 125: 293]
    pose_direction = raw_input[:, 293: 299]
    face_landmark2D = raw_input[:, 299: 435]
    face_landmark3D = raw_input[:, 435: 679]
    au_reg = raw_input[:, 679: 695]
    au_cls = raw_input[:, 695: 713]

    gaze_direction_std = np.std(gaze_direction, axis=0)
    gaze_direction_mean = np.mean(gaze_direction, axis=0)

    gaze_angle_std = np.std(gaze_angle, axis=0)
    gaze_angle_mean = np.mean(gaze_angle, axis=0)

    eye_landmark2D_shape_0 = np.abs(eye_landmark2D[:, 56 + 9: 56 + 14] - eye_landmark2D[:, 56 + 19: 56 + 14: -1])
    eye_landmark2D_shape_1 = np.abs(eye_landmark2D[:, 56 + 37: 56 + 42] - eye_landmark2D[:, 56 + 47: 56 + 42: -1])
    eye_landmark2D_shape = np.hstack((eye_landmark2D_shape_0, eye_landmark2D_shape_1))
    eye_landmark2D_shape_cov = np.divide(np.std(eye_landmark2D_shape, axis=0),
                                         np.mean(eye_landmark2D_shape, axis=0))

    eye_distance = 0.5 * (eye_landmark3D[:, 56 * 2 + 8] + eye_landmark3D[:, 56 * 2 + 42])
    eye_distance_cov = np.std(eye_distance) / np.mean(eye_distance)
    eye_distance_ratio = np.min(eye_distance) / np.max(eye_distance)
    eye_distance_fea = np.array([eye_distance_cov, eye_distance_ratio])

    eye_location2D = []
    for idx in range(4):
        cur_mean = np.mean(eye_landmark2D[:, 28 * idx: 28 * (idx + 1)], axis=1)
        eye_location2D.append(cur_mean)

    eye_location2D = np.vstack(eye_location2D).T
    eye_location2D_mean = np.mean(eye_location2D, axis=0)
    eye_location2D_std = np.std(eye_location2D, axis=0)

    eye_location3D = []
    for idx in range(6):
        cur_mean = np.mean(eye_landmark3D[:, 28 * idx: 28 * (idx + 1)], axis=1)
        eye_location3D.append(cur_mean)
    eye_location3D = np.vstack(eye_location3D).T
    eye_location3D_mean = np.mean(eye_location3D, axis=0)
    eye_location3D_std = np.std(eye_location3D, axis=0)

    pose_direction_mean = np.mean(pose_direction, axis=0)
    pose_direction_std = np.std(pose_direction, axis=0)
    ret_features = np.hstack((gaze_direction_std, gaze_direction_mean, gaze_angle_mean, gaze_angle_std,
                              eye_landmark2D_shape_cov, eye_location2D_mean, eye_location2D_std,
                              eye_location3D_mean,
                              eye_location3D_std, eye_distance_fea, pose_direction_mean, pose_direction_std))

    return ret_features

def parse_df(df_path, n_segments=15, alpha=0.5, prev_frames=-1):
    try:
        df = pd.read_csv(df_path, header=0, sep=',').values
        face_id = df[:, 1]
        seq_length = df.shape[0]
        # print("Seq length: ", seq_length)
        if seq_length < 100:
            return None
        indexing = int((n_segments - 1) * (1 - alpha))
        k_value = seq_length // (1 + indexing)  # In some case, we will ignore some last frames

        ret = []
        index_st = 0
        for idx in range(n_segments):
            index_ed = k_value + int(k_value * (1 - alpha) * idx)
            index_features = get_gaze_features(df[index_st: index_ed, :])
            ret.append(index_features)
            index_st = index_ed - int((1 - alpha) * k_value)

        ret = np.vstack(ret)
    except:
        print('IO error')
        ret = None

    return ret

def get_model(model_index, n_segments=15, input_units=60):
    """
    Make prediction for data_npy
    :param data_npy:
    :return:
    """
    ld_cfg = cfg.md_cfg
    hparams = copy.deepcopy(ld_cfg[model_index])

    if 'VGG' in hparams['NAME']:
        ft_type = 'vgg2'
    elif 'OF' in hparams['NAME']:
        ft_type = 'of'
    else:
        ft_type = 'au'

    hparams['TIME_STEP'] = n_segments
    hparams['INPUT_UNITS'] = hparams['FC1'][1] if hparams['FC1'][1] > 0 else input_units
    hparams['optimizer'] = 'adam'
    hparams['ACTIVATION_F'] = 'tanh'
    hparams['CLSW'] = 1

    cur_model = define_model(hparams)
    cur_model.build()
    # load_weights_to_model(cur_model, hparams, ft_type)
    cur_model.load_weights(
            './models/{}_{}_models_{}_{}_0_epochs{}_best_weight.h5'.format(hparams['model_path'], ft_type,
                                                                           hparams['n_segments'], hparams['alpha'],
                                                                           hparams['EPOCHS']))

    return cur_model

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:7331")

    # Model index: 0, 1 for VGG_SE and 2, 3 for EyeGaze_HeadPose
    eye_gaze_v1 = get_model(model_index=2)
    eye_gaze_v2 = get_model(model_index=3)
    prev_frames = -1
    while True:
        #  Wait for next request from client
        message = socket.recv()
        # print("Received request: %s" % message)
        df_path = message.decode("utf-8")
        # print(df_path)
        eye_gaze_features = parse_df(df_path, n_segments=15, alpha=0.5, prev_frames=prev_frames)

        if eye_gaze_features is not None:
            # print(eye_gaze_features.shape)
            eye_gaze_features = eye_gaze_features[np.newaxis, :]
            # print(eye_gaze_features.shape)

            v1 = eye_gaze_v1.predict(eye_gaze_features)[0][0]
            v2 = eye_gaze_v2.predict(eye_gaze_features)[0][0]
            enga_score = 0.5*(v1 + v2)
            #  Do some 'work'
            # time.sleep(.300)
            send_str = "{:.5f}".format(enga_score)
            #  Send reply back to client
            socket.send(send_str.encode('ascii'))
        else:
            socket.send(b'NA')