from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = False

opts['init_model_path'] = '../models/imagenet-vgg-m.mat'
opts['model_path'] = '../models/mdnet_vot-otb_new.pth'
opts['original_model_path'] = '../models/mdnet_vot-otb_original.pth'

opts['batch_frames'] = 8
opts['batch_pos'] = 32
opts['batch_neg'] = 96

opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]

opts['img_size'] = 107
opts['padding'] = 16

opts['lr'] = 0.0001
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
# opts['ft_layers'] = ['conv','fc']
opts['ft_layers'] = ['fe', 'cl', 'fusion']
opts['lr_mult'] = {'cl1_fc':10, 'cl2_fc':10, 'cl3_fc':10, 'fusion_fc':10}
opts['n_cycles'] = 50
