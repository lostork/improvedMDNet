import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

# add parameters to params dict
def append_params(params, module, prefix):
    # module : nn.Sequential(...)
    # prefix : 'conv1'
    for child in module.children():
        # child = conv2d(3,96,...)
        for k,p in child._parameters.iteritems():
            # k : 'weight', 'bias'.
            # p : Parameter.
            if p is None: continue
            
            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k
            
            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))


def classify_from_feat(feat, classifer, branches, k):

    # TODO:unc; is this change fusion feat
    # TODO: before_fc_feats is not stable
    before_fc_feats = feat.clone()  # TODO: may have problem
    out = feat
    for name, module in classifer.named_children():
        out = module(out)
        if name.endswith('conv'):
            out = out.view(out.size(0), -1)
            before_fc_feats = out

    out = branches[k](out)
    return out, before_fc_feats

# assure feat has been flattened
def classify_from_bf_fc_feat(feat, classifer, branches, k):
    out = feat
    for name, module in classifer.named_children():
        if not name.endswith('conv'):
            out = module(out)

    out = branches[k](out)
    return out

# conv1_out = conv1_feat
#
# for name, module in self.conv1_classifier.named_children():
#     conv1_out = module(conv1_out)
#     if name.endswith('conv'):
#         conv1_out = conv1_out.view(conv1_out.size(0), -1)
#
# conv1_out = self.cl1_branches[k](conv1_out)

class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.cnn_layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        # overlap pooling
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU()))
            # ,
            # # torch.nn.Linear(in_features, out_features, bias=True)
            #     ('fc4',   nn.Sequential(nn.Dropout(0.5),
            #                             nn.Linear(512 * 3 * 3, 512),
            #                             nn.ReLU())),
            #     ('fc5',   nn.Sequential(nn.Dropout(0.5),
            #                             nn.Linear(512, 512),
            #                             nn.ReLU()))
        ]))

        self.conv1_feat_extractor = nn.Sequential(OrderedDict([
            ('fe1_conv', nn.Sequential(nn.Conv2d(96, 96, kernel_size=7, stride=2),
                                       nn.MaxPool2d(kernel_size=3, stride=3)))   # TODO: output size check.
        ]))

        self.conv2_feat_extractor = nn.Sequential(OrderedDict([
            ('fe2_conv', nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1)))
        ]))

        # self.conv3_feat_extractor = nn.Sequential(OrderedDict([
        #     ('fe3_conv', nn.Sequential(nn.Conv2d(512, 16, kernel_size=1, stride=1))),
        #     ('fe3_deconv', nn.Sequential(nn.Upsample((51,51), mode='bilinear')))
        # ]))

        # self.conv1_classifier = nn.Sequential(OrderedDict([
        #     ('cl1_fc4',   nn.Sequential(nn.Dropout(0.5),
        #                             nn.Linear(3 * 3 * 96, 512),
        #                             nn.ReLU())),
        #     ('cl1_fc5',   nn.Sequential(nn.Dropout(0.5),
        #                             nn.Linear(512, 512),
        #                             nn.ReLU()))
        # ]))

        # self.conv2_classifier = nn.Sequential(OrderedDict([
        #     ('cl2_fc4', nn.Sequential(nn.Dropout(0.5),
        #                               nn.Linear(3 * 3 * 256, 512),
        #                               nn.ReLU())),
        #     ('cl2_fc5', nn.Sequential(nn.Dropout(0.5),
        #                               nn.Linear(512, 512),
        #                               nn.ReLU()))
        # ]))

        self.conv3_classifier = nn.Sequential(OrderedDict([
            ('cl3_fc4', nn.Sequential(nn.Dropout(0.5),
                                      nn.Linear(3 * 3 * 512, 512),
                                      nn.ReLU())),
            ('cl3_fc5', nn.Sequential(nn.Dropout(0.5),
                                      nn.Linear(512, 512),
                                      nn.ReLU()))
        ]))

        self.fusion_classifier = nn.Sequential(OrderedDict([
            ('fusion_conv', nn.Sequential(nn.Conv2d(96+256+512, 512, kernel_size=1, stride=1))),
            ('fusion_fc4', nn.Sequential(nn.Dropout(0.5),
                                      nn.Linear(3 * 3 * 512, 512),
                                      nn.ReLU())),
            ('fusion_fc5', nn.Sequential(nn.Dropout(0.5),
                                      nn.Linear(512, 512),
                                      nn.ReLU()))
        ]))

        self.shared_layers = [self.cnn_layers,
                              self.conv1_feat_extractor,
                              self.conv2_feat_extractor,
                              # self.conv3_feat_extractor,
                              # self.conv1_classifier,
                              # self.conv2_classifier,
                              self.conv3_classifier,
                              self.fusion_classifier
                              ]

        # self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
        #                                              nn.Linear(512, 2)) for _ in range(K)])

        # self.cl1_branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
        #                                                  nn.Linear(512, 2)) for _ in range(K)])
        #
        # self.cl2_branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
        #                                                  nn.Linear(512, 2)) for _ in range(K)])

        self.cl3_branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                         nn.Linear(512, 2)) for _ in range(K)])

        self.fusion_branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                            nn.Linear(512, 2)) for _ in range(K)])

        self.domain_specific_layers = [
            # self.cl1_branches,
            #                            self.cl2_branches,
                                       self.cl3_branches,
                                       self.fusion_branches
                                       ]

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                # load pretrained shared layer parameters
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unknown model format: %s" % (model_path))

        # add all (including shared layers and branch) parameter to self.params
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()

        for layers in self.shared_layers:
            for name, module in layers.named_children():
                append_params(self.params, module, name)

        # for name, module in self.cnn_layers.named_children():
        #     # name = 'conv1'
        #     # module = nn.Sequential
        #     append_params(self.params, module, name)


        # for name, module in self.conv1_feat_extractor.named_children():
        #     append_params(self.params, module, name)
        #
        # for name, module in self.conv2_feat_extractor.named_children():
        #     append_params(self.params, module, name)
        #
        # for name, module in self.conv3_feat_extractor.named_children():
        #     append_params(self.params, module, name)
        #

        # for k, module in enumerate(self.cl1_branches):
        #     append_params(self.params, module, 'cl1_fc6_%d' % (k))
        #
        # for k, module in enumerate(self.cl2_branches):
        #     append_params(self.params, module, 'cl2_fc6_%d' % (k))

        for k, module in enumerate(self.cl3_branches):
            append_params(self.params, module, 'cl3_fc6_%d' % (k))

        for k, module in enumerate(self.fusion_branches):
            append_params(self.params, module, 'fusion_fc6_%d' % (k))

        # for k, module in enumerate(self.branches):
        #     append_params(self.params, module, 'fc6_%d' % (k))

        # append_params(self.params, self.conv1_feat_extractor, 'fe_1')

    def set_learnable_params(self, layers):
        for k, p in self.params.iteritems():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
 
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.iteritems():
            if p.requires_grad:
                params[k] = p
        return params
    
    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        #
        # forward model from in_layer to out_layer

        conv1_feat = None
        conv2_feat = None
        conv3_feat = None
        fusion_feat = None
        conv1_scores = None
        conv2_scores = None
        conv3_scores = None
        fusion_scores = None

        conv1_bf_fc_feats = None
        conv2_bf_fc_feats = None
        conv3_bf_fc_feats = None

        if in_layer in ('conv1', 'before_fc') and (out_layer in ('fusion', 'fusion_feats_only', 'combined_feats', 'conv3')):
            if in_layer == 'conv1':
                for name, module in self.cnn_layers.named_children():
                    if name == 'conv1':
                        conv1_feat = module(x)
                    elif name == 'conv2':
                        conv2_feat = module(conv1_feat)
                    elif name == 'conv3':
                        conv3_feat = module(conv2_feat)


                conv1_feat = self.conv1_feat_extractor(conv1_feat)
                conv2_feat = self.conv2_feat_extractor(conv2_feat)
                # conv3_feat = self.conv3_feat_extractor(conv3_feat)

                #TODO:unc;
                fusion_feat = torch.cat((conv1_feat, conv2_feat, conv3_feat), 1)

                # if out_layer == 'fusion_feats_only':
                #     #             x = x.view(x.size(0),-1)
                #     for name, module in self.fusion_classifier.named_children():
                #         if name == 'fusion_conv':
                #             result = module(fusion_feat)
                #             result = result.view(result.size(0), -1)
                #             return result

                # TODO:unc; is this change fusion feat
                # conv1_out = conv1_feat
                #
                # for name, module in self.conv1_classifier.named_children():
                #     conv1_out = module(conv1_out)
                #     if name.endswith('conv'):
                #         conv1_out = conv1_out.view(conv1_out.size(0), -1)
                #
                # conv1_out = self.cl1_branches[k](conv1_out)
                fusion_scores, fusion_bf_fc_feats = classify_from_feat(fusion_feat, self.fusion_classifier, self.fusion_branches, k)
                if out_layer == 'fusion_feats_only':
                    return fusion_bf_fc_feats

                if out_layer == 'combined_feats':
                    # TODO: unc;
                    combined_feats = torch.cat((fusion_feat,
                                                fusion_bf_fc_feats.view(-1, 512, 3, 3)), 1)

                    # print (combined_feats[1,1,:] == conv2_bf_fc_feats[1,:])
                    # print combined_feats[1,1,:].size()
                    # print combined_feats[:,1,:].size()
                    # print (combined_feats[1, 3, :] == fusion_bf_fc_feats[1, :])

                    return combined_feats

                # conv1_scores, conv1_bf_fc_feats = classify_from_feat(conv1_feat.view(conv1_feat.size(0), -1), self.conv1_classifier, self.cl1_branches, k)
                # conv2_scores, conv2_bf_fc_feats = classify_from_feat(conv2_feat.view(conv2_feat.size(0), -1), self.conv2_classifier, self.cl2_branches, k)
                conv3_scores, conv3_bf_fc_feats = classify_from_feat(conv3_feat.view(conv3_feat.size(0), -1), self.conv3_classifier, self.cl3_branches, k)

                if out_layer == 'conv3':
                    return conv3_bf_fc_feats



            elif in_layer == 'before_fc':
                # conv1_bf_fc_feats = x[:,0,:]
                # conv2_bf_fc_feats = x[:,1,:]
                # conv3_bf_fc_feats = x[:,2,:]
                # fusion_bf_fc_feats = x[:,3,:]
                # conv1_bf_fc_feats = x[:,0:96,:,:].clone() # TODO: maybe more check.
                # conv1_bf_fc_feats = conv1_bf_fc_feats.view(conv1_bf_fc_feats.size(0), -1)
                # conv2_bf_fc_feats = x[:,96:352,:,:].clone()
                # conv2_bf_fc_feats = conv2_bf_fc_feats.view(conv2_bf_fc_feats.size(0), -1)
                conv3_bf_fc_feats = x[:,352:864,:,:].clone()
                conv3_bf_fc_feats = conv3_bf_fc_feats.view(conv3_bf_fc_feats.size(0), -1)
                fusion_bf_fc_feats = x[:,864:1376,:,:].clone()
                fusion_bf_fc_feats = fusion_bf_fc_feats.view(fusion_bf_fc_feats.size(0), -1)

                # conv1_scores = classify_from_bf_fc_feat(conv1_bf_fc_feats, self.conv1_classifier, self.cl1_branches, k)
                # conv2_scores = classify_from_bf_fc_feat(conv2_bf_fc_feats, self.conv2_classifier, self.cl2_branches, k)
                conv3_scores = classify_from_bf_fc_feat(conv3_bf_fc_feats, self.conv3_classifier, self.cl3_branches, k)
                fusion_scores = classify_from_bf_fc_feat(fusion_bf_fc_feats, self.fusion_classifier, self.fusion_branches, k)

            # TODO: better impl and more options
            # final_scores = (conv1_scores + conv2_scores + conv3_scores + fusion_scores) / 4
            final_scores = (conv3_scores + fusion_scores) / 2
            return final_scores


        #TODO:
        # run = False
        # for name, module in self.cnn_layers.named_children():
        #     if name == in_layer:
        #         run = True
        #     if run:
        #         x = module(x)
        #         if name == 'conv3':
        #             x = x.view(x.size(0),-1)
        #         if name == out_layer:
        #             return x
        #
        # x = self.branches[k](x)
        # if out_layer=='fc6':
        #     return x
        # elif out_layer=='fc6_softmax':
        #     return F.softmax(x)
    
    def load_model(self, model_path):
        # TODO: understand the details

        # states = {'cnn_layers': model.cnn_layers.state_dict(),
        #           'conv1_feat_extractor': model.conv1_feat_extractor.state_dict(),
        #           'conv2_feat_extractor': model.conv2_feat_extractor.state_dict(),
        #           'conv3_feat_extractor': model.conv3_feat_extractor.state_dict(),
        #           'conv1_classifier': model.conv1_classifier.state_dict(),
        #           'conv2_classifier': model.conv2_classifier.state_dict(),
        #           'conv3_classifier': model.conv3_classifier.state_dict(),
        #           'fusion_classifier': model.fusion_classifier.state_dict()
        #           }

        states = torch.load(model_path)


        # shared_layers = states['shared_layers']
        # self.cnn_layers.load_state_dict(shared_layers)
        # TODO: more check.
        self.cnn_layers.load_state_dict(states['cnn_layers'])
        self.conv1_feat_extractor.load_state_dict(states['conv1_feat_extractor'])
        self.conv2_feat_extractor.load_state_dict(states['conv2_feat_extractor'])
        # self.conv3_feat_extractor.load_state_dict(states['conv3_feat_extractor'])
        # self.conv1_classifier.load_state_dict(states['conv1_classifier'])
        # self.conv2_classifier.load_state_dict(states['conv2_classifier'])
        self.conv3_classifier.load_state_dict(states['conv3_classifier'])
        self.fusion_classifier.load_state_dict(states['fusion_classifier'])


    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]
        
        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.cnn_layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.cnn_layers[i][0].bias.data = torch.from_numpy(bias[:, 0])

    

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
 
    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:,1]
        neg_loss = -F.log_softmax(neg_score)[:,0]
        
        loss = pos_loss.sum() + neg_loss.sum()
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        
        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()
        
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):

        # take the topk scores and check if they are belong to pos_score,
        # #(belongs) / #(num of pos_score) = precision
        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)
        
        return prec.data[0]
