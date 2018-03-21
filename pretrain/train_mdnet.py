import os
import sys
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

from data_prov import *
from model import *
from options import *

img_home = '../dataset/'
data_path = 'data/vot-otb.pkl'

def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train_mdnet():
    
    ## Init dataset ##
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)

    K = len(data)
    dataset = [None]*K
    for k, (seqname, seq) in enumerate(data.iteritems()):
        img_list = seq['images']
        gt = seq['gt']
        img_dir = os.path.join(img_home, seqname)
        dataset[k] = RegionDataset(img_dir, img_list, gt, opts)

    ##------------------------------------

    ## Init model ##
    # opts['init_model_path'] = '../models/imagenet-vgg-m.mat'
    # VGG-M pretrained on ImageNet
    model = MDNet(opts['init_model_path'], K)
    # model = MDNet(None, K)
    if opts['use_gpu']:
        model = model.cuda()

    # opts['ft_layers'] = ['conv','fc']
    model.set_learnable_params(opts['ft_layers'])
        
    ## Init criterion and optimizer ##
    criterion = BinaryLoss()
    evaluator = Precision()

    # opts['lr'] = 0.0001
    optimizer = set_optimizer(model, opts['lr'])

    best_prec = 0.
    # opts['n_cycles'] = 50
    # TODO: cancel test
    for i in range(opts['n_cycles']):
    # for i in range(2):
        print "==== Start Cycle %d ====" % (i)
        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        for j,k in enumerate(k_list):
            tic = time.time()
            pos_regions, neg_regions = dataset[k].next()
            
            pos_regions = Variable(pos_regions)
            neg_regions = Variable(neg_regions)
        
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
        
            pos_score = model(pos_regions, k, out_layer='fusion')
            neg_score = model(neg_regions, k, out_layer='fusion')

            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            # TODO: unc for the clip grad effect in the new architecture
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()
            
            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print "Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                    (i, j, k, loss.data[0], prec[k], toc)

        cur_prec = prec.mean()
        print "Mean Precision: %.3f" % (cur_prec)
        if cur_prec > best_prec:
            best_prec = cur_prec

            if opts['use_gpu']:
                model = model.cpu()

            # self.shared_layers = [self.cnn_layers,
            #                       self.conv1_feat_extractor,
            #                       self.conv2_feat_extractor,
            #                       self.conv3_feat_extractor,
            #                       self.conv1_classifier,
            #                       self.conv2_classifier,
            #                       self.conv3_classifier,
            #                       self.fusion_classifier
            #                       ]

            states = {'cnn_layers': model.cnn_layers.state_dict(),
                      'conv1_feat_extractor': model.conv1_feat_extractor.state_dict(),
                      'conv2_feat_extractor': model.conv2_feat_extractor.state_dict(),
                      'conv3_feat_extractor': model.conv3_feat_extractor.state_dict(),
                      'conv1_classifier': model.conv1_classifier.state_dict(),
                      'conv2_classifier': model.conv2_classifier.state_dict(),
                      'conv3_classifier': model.conv3_classifier.state_dict(),
                      'fusion_classifier': model.fusion_classifier.state_dict()
                      }
            # states = {'shared_layers': model.layers.state_dict()}

            print "Save model to %s" % opts['model_path']
            torch.save(states, opts['model_path'])
            # print "Save model to %s" % opts['model_path']
            # torch.save(states, opts['model_path'])

            if opts['use_gpu']:
                model = model.cuda()
            # if opts['use_gpu']:
            #     model = model.cuda()


if __name__ == "__main__":
    train_mdnet()

