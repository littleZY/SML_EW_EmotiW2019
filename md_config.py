import numpy as np

lr_config_001 = np.logspace(-2, -4, base=10, num=6)

md_cfg = []
vgg_cfg_001 = {'NAME': 'VGG_001', 'model_path': '07c4b02529e14762854114d3521bd8cf', 'n_segments':21, 'alpha':0.5, 'augment_size':0, 'EPOCHS':10000, 'NUM_LSTM_LAYERS':2, 'LSTM_UNITS':[64,128], 'NUM_DENSE_LAYERS':2, 'DENSE_UNITS':[64,128], 'DROPOUT_RATES':0, 'FC1':[0, 0], 'BATCH_SIZE':45, 'learning_rate':lr_config_001[2]}
vgg_cfg_002 = {'NAME': 'VGG_002', 'model_path': 'da05e56e0de54351bb42aff26657f5ed', 'n_segments':21, 'alpha':0.5, 'augment_size':0, 'EPOCHS':10000, 'NUM_LSTM_LAYERS':2, 'LSTM_UNITS':[64,128], 'NUM_DENSE_LAYERS':2, 'DENSE_UNITS':[48,64], 'DROPOUT_RATES':0, 'FC1':[128, 64], 'BATCH_SIZE':45, 'learning_rate':lr_config_001[1]}
of_cfg_001 = {'NAME': 'OF_001', 'model_path':'f841b83091a34af3a086a6d2acb88f19', 'n_segments':15, 'alpha':0.5, 'augment_size':0, 'EPOCHS':10000, 'NUM_LSTM_LAYERS': 2, 'LSTM_UNITS':[128, 128], 'NUM_DENSE_LAYERS':2, 'DENSE_UNITS':[128, 128], 'DROPOUT_RATES':0, 'FC1':[0, 0], 'BATCH_SIZE':32, 'learning_rate':lr_config_001[4]}
of_cfg_002 = {'NAME': 'OF_002', 'model_path':'d53865dd121f47b4af481c070ae2c62b', 'n_segments':15, 'alpha':0.5, 'augment_size':0, 'EPOCHS':10000, 'NUM_LSTM_LAYERS': 2, 'LSTM_UNITS':[128, 128], 'NUM_DENSE_LAYERS':2, 'DENSE_UNITS':[48, 128], 'DROPOUT_RATES':0, 'FC1':[60, 100], 'BATCH_SIZE':40, 'learning_rate':lr_config_001[5]}

md_cfg.append(vgg_cfg_001)
md_cfg.append(vgg_cfg_002)
md_cfg.append(of_cfg_001)
md_cfg.append(of_cfg_002)
