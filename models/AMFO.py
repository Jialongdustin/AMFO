import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.backbone import ast_models
from models.backbone import Resnet
from models.binary_net import BinaryNet
from config import settings

class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()
        if args.pretrained == True:
            self.backbone = Resnet.resnet18(pretrained=True)
        else:
            self.backbone = Resnet.resnet18(pretrained=False)
        self.node = 512
        if args.dataset == 'librispeech':
            self.session_len = settings.librispeech_SessLen
            self.m_n = -0.5 # nagative offset--according your data you can change a little
            self.m_p = 0.1 # positive offset
        self.fc1 = nn.Linear(512, self.node, bias=False)
        self.fc2 = nn.Linear(self.node, self.session_len[0], bias=False)
        self.fc2_n = nn.Linear(self.node, self.session_len[0], bias=False)
        self.binary_classifier = BinaryNet(2*args.base_num)
        self.relu = nn.ReLU(inplace=True)
        SessLen = len(self.session_len)
        for i in range(1, SessLen):
            exec('self.fc' + str(i + 2) + '= nn.Linear(self.node, self.session_len[i], bias=False)')
            exec('self.fc_n' + str(i + 2) + '= nn.Linear(self.node, self.session_len[i], bias=False)')
            
            exec('self.fc_aux' + str(i + 2) + '= nn.Linear(512, self.node, bias=False)')
           
        Alpha = torch.zeros(SessLen, self.node)
        Alpha[0] = Alpha[0] + 1  
        self.register_buffer('Alpha', Alpha)  
        Alpha_n = torch.zeros(SessLen, self.node)
        Alpha_n[0] = Alpha_n[0] + 1  
        self.register_buffer('Alpha_n', Alpha_n)
        self.r = nn.Parameter(torch.tensor(0.0))  
        self.gamma = args.gamma
        self.Gamma = [1] * SessLen 
        self.sess = 0
        self.args = args

    def get_feature(self, x):
        global y
        y, x = self.backbone(x)   
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

        return y, x
    
    def attention_forward(self, x):
        x = self.binary_classifier(x)
        
        return x

    def forward(self, x, sess=0, epoch=0, Mode='train', IOF='voice'):
        global y
        self.sess = sess
        if IOF == 'voice':
            if sess > 0:  
                with torch.no_grad():
                    y, x = self.get_feature(x)
            else:
                y, x = self.get_feature(x)
        out = self.fc1(x)
        if sess > 0:
            y = x.detach()
            out_n = self.fc1(y)
        else:
            if IOF == 'feature':
                y = x.detach()
                out_n = self.fc1(y)
            else:
                out_n = self.fc1(y)
        
        for i in range(sess + 1):
            if i == 0:
                output_n = F.linear(F.normalize(out_n, p=2, dim=-1), F.normalize(self.fc2_n.weight, p=2, dim=-1))
                output = F.linear(F.normalize(out, p=2, dim=-1), F.normalize(self.fc2.weight, p=2, dim=-1))
               
            else:
                fc = eval('self.fc' + str(i + 2))
                fc_n = eval('self.fc_n' + str(i + 2))
                fc_aux = eval('self.fc_aux' + str(i + 2))
                # normalize
                out_aux = F.linear(F.normalize(x.view(x.size(0), -1), p=2, dim=-1), F.normalize(fc_aux.weight, p=2, dim=-1)) 
                y_aux = F.linear(F.normalize(y.view(y.size(0), -1), p=2, dim=-1), F.normalize(fc_aux.weight, p=2, dim=-1))
                if i < sess:
                    out_aux = out_aux * self.Alpha[i]     
                    y_aux = y_aux * self.Alpha_n[i]
                else:
                    if Mode == 'train':
                        beta = 1.0 + max(epoch, 0)   
                        t = torch.mean(out_aux, dim=0)  
                        self.alpha = torch.sigmoid(beta * t)  
                        out_aux = out_aux * self.alpha  
                        t_n = torch.mean(y_aux, dim=0)
                        self.alpha_n = torch.sigmoid(beta * t_n)
                        y_aux = y_aux * self.alpha_n
                    elif Mode == 'train_old':
                        out_aux = out_aux * self.alpha
                        y_aux = y_aux * self.alpha_n
                    else:
                        out_aux = out_aux * self.Alpha[i]
                        y_aux = y_aux * self.Alpha_n[i]
                new_node = out * self.gamma + out_aux  
                new_node = self._l2norm(new_node, dim=1)
                new_node_n = out_n * self.gamma + y_aux 
                new_node_n = self._l2norm(new_node_n, dim=1)
                output = torch.cat([output, F.linear(F.normalize(new_node, p=2, dim=-1), F.normalize(fc.weight, p=2, dim=-1))], dim=1)
                output_n = torch.cat([output_n, F.linear(F.normalize(new_node_n, p=2, dim=-1), F.normalize(fc_n.weight, p=2, dim=-1))], dim=1)
                
        return out, out_n, output_n, output
    

    def finish_train(self):
        self.Alpha[self.sess] = self.alpha.detach()  
        self.Alpha_n[self.sess] = self.alpha_n.detach()
        # pass

    def _l2norm(self, inp, dim=1):
        """Normlize the inp tensor with l2-norm."""
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))   

    def get_loss(self, args, pred, pred_indomain, combain, scores, label, output_old=None, output_old_in=None, compression=True):
        angular_criterion_p = AngularPenaltySMLoss(loss_type='cosface', s=32.0, m=self.m_p)
        angular_criterion_n = AngularPenaltySMLoss(loss_type='cosface', s=32.0, m=self.m_n)
    
        loss_angular_n = angular_criterion_n(pred, label.long())
        loss_angular_p = angular_criterion_p(pred_indomain, label.long())

        loss_atten = args.weight_atten * F.cross_entropy(combain, label.long(), reduction='mean')
        label_binary_all = (label >= args.base_num).long()
        loss_binary = args.weight_binary * F.cross_entropy(scores, label_binary_all, reduction='mean')
        loss_margin = args.weight_margin * (loss_angular_p + loss_angular_n)
        loss_backbone = loss_atten + loss_margin + loss_binary
        loss_dis = 0
        R1 = 0
        R2 = 0
        if output_old is not None:
            loss_dis = self.distillation_loss(pred_indomain, output_old_in)

        if self.sess > 0 and compression:
            R1 = torch.sum(nn.ReLU()(torch.norm(self.alpha, p=1, dim=0) / self.node - self.r))
            R2 = torch.sum(nn.ReLU()(torch.norm(self.alpha_n, p=1, dim=0) / self.node - self.r))
            
        return loss_backbone + loss_dis + 0.1*(R1+R2), loss_margin, loss_dis, loss_atten, loss_binary 
    

    def distillation_loss(self, pred_N, pred_O, T=0.5):  
        if pred_N.shape[1] != pred_O.shape[1]:    
            pred_N = pred_N[:, :pred_O.shape[1]]   
        outputs = torch.log_softmax(pred_N / T, dim=1)  # compute the log of softmax values
        labels = torch.softmax(pred_O / T, dim=1)
        # loss = self.cal_kl_div(outputs, labels)
        # loss = loss * (T**2)
        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        loss = -torch.mean(outputs, dim=0, keepdim=False)

        return loss
    


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''

        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(wf, labels)
        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)

