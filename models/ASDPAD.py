import torch as t
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
import sys
from models.surrogate_networks import *
sys.path.append('../')
from loss.vgg_loss import VGG19Loss
from loss.hard_triplet_loss import HardTripletLoss

def l2_norm(input, axis=1):
    norm = t.norm(input, 2, axis, True)
    output = t.div(input, norm)
    return output
    
class CNN(nn.Module):
    def __init__(self, cnn_model):
        super(CNN, self).__init__()
        self.cnn = cnn_model
        
        if self.cnn == "ResNet50":
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True)
            )
        elif self.cnn == "DenseNet121":
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True)
            )
        elif self.cnn == "ConvNeXtTiny":
            self.model = models.convnext_tiny(pretrained=True)
            self.model.classifier[2] = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(inplace=True)
            )
        elif self.cnn == "ConvNeXtSmall":
            self.model = models.convnext_small(pretrained=True)
            self.model.classifier[2] = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(inplace=True)
            ) 
        self.fc = nn.Linear(256, 256)
        self.fc.weight.data.normal_(0, 0.005)
        self.fc.bias.data.fill_(0.1)
        self.fc1 = nn.Sequential(
            self.fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag):
        feature = self.model(input)
        feature = self.fc1(feature)
        if(norm_flag):
            feature_norm = t.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = t.div(feature, feature_norm)
        return feature
    
class SSDG_GRL(t.autograd.Function):
    def __init__(self, max_iter):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput
    
class SSDG_Discriminator(nn.Module):
    def __init__(self, max_iter):
        super(SSDG_Discriminator, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(256, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = SSDG_GRL(max_iter)

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer.forward(feature))
        return adversarial_out

    
class SSDG_Classifier(nn.Module):
    def __init__(self):
        super(SSDG_Classifier, self).__init__()
        self.classifier_layer = nn.Linear(256, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out

class DG(nn.Module):
    def __init__(self, cnn_model):
        super(DG, self).__init__()
        self.embedder = CNN(cnn_model)
        self.classifier = SSDG_Classifier()

    def forward(self, input, norm_flag):
        feature = self.embedder(input, norm_flag)
        classifier_out = self.classifier(feature, norm_flag)
        return classifier_out, feature

class ASDPAD():
    def __init__(self, opt, vis):
        self.opt = opt

        self.netG_A1 = surrogate_define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_A2 = surrogate_define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, 
                                          not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_A1 = self.netG_A1.cuda(opt.gpu_ids[0])
        self.netG_A2 = self.netG_A2.cuda(opt.gpu_ids[0])
        if opt.surrogate_model_path1 != "" and opt.surrogate_model_path2 != "":
            self.load_surrogate(opt.surrogate_model_path1, opt.surrogate_model_path2)
            
        self.SSDG = DG(opt.cnn_model).cuda(opt.gpu_ids[0])
        self.ad_net_genuine = SSDG_Discriminator(opt.max_iter).cuda(opt.gpu_ids[0])
        if (opt.load_cnn_path):
            net_ = torch.load(opt.load_cnn_path, map_location='cpu')
            import collections
            new_net_ = collections.OrderedDict()
            for key in net_.keys():
                new_key = key.replace("model","embedder.model")
                new_net_[new_key] = net_[key]           
            info = self.SSDG.load_state_dict(new_net_, strict=False)
        self.SSDG.train()
        self.ad_net_genuine.train()
        
        self.criterion_vgg19 = VGG19Loss().cuda(opt.gpu_ids[0])
        self.criterion_softmax = nn.CrossEntropyLoss().cuda(opt.gpu_ids[0])
        self.criterion_triplet = HardTripletLoss(margin=self.opt.margin, hardest=False).cuda(opt.gpu_ids[0])
        
        self.optimizerG1 = torch.optim.Adam(self.netG_A1.parameters(), lr=opt.surrogate_init_lr, betas=(opt.beta1, 0.999))
        self.optimizerG2 = torch.optim.Adam(self.netG_A2.parameters(), lr=opt.surrogate_init_lr, betas=(opt.beta1, 0.999))
        optimizer_dict = [
            {"params": filter(lambda p: p.requires_grad, self.SSDG.parameters()), "lr": opt.cnn_init_lr},
            {"params": filter(lambda p: p.requires_grad, self.ad_net_genuine.parameters()), "lr": opt.cnn_init_lr},
        ]
        self.optimizerSSDGAndAdgenuine = optim.Adam(optimizer_dict, lr=opt.cnn_init_lr)

    def update_optimizeG1_parameters(self, surrogate_loss):
        self.netG_A1.train()
        self.optimizerG1.zero_grad()
        surrogate_loss.backward()
        self.optimizerG1.step()
        
    def update_optimizeG2_parameters(self, surrogate_loss):
        self.netG_A2.train()
        self.optimizerG2.zero_grad()
        surrogate_loss.backward()
        self.optimizerG2.step()
        
    def update_optimizeSSDGAndAdgenuine_parameters(self, cls_loss, triplet, genuine_adloss):
        self.SSDG.train()
        self.ad_net_genuine.train()
        loss = cls_loss + self.opt.lambda_triplet * triplet + self.opt.lambda_adgenuine * genuine_adloss
        self.optimizerSSDGAndAdgenuine.zero_grad() 
        loss.backward()
        self.optimizerSSDGAndAdgenuine.step()
        
    def forward(self, genuine_A):
        self.netG_A1.eval()
        self.netG_A2.eval()
        recaptured_B1 = self.netG_A1(genuine_A)
        recaptured_B1_label = torch.zeros(len(genuine_A), dtype = torch.int).cuda(self.opt.gpu_ids[0])
        recaptured_B2 = self.netG_A2(genuine_A)
        recaptured_B2_label = torch.zeros(len(genuine_A), dtype = torch.int).cuda(self.opt.gpu_ids[0])
        return recaptured_B1, recaptured_B1_label, recaptured_B2, recaptured_B2_label

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
            
    def load_surrogate(self, load_path1, load_path2):
        net1 = self.netG_A1
        if isinstance(net1, torch.nn.DataParallel):
            net1 = net1.module
        state_dict1 = torch.load((load_path1), map_location="cpu")
        if hasattr(state_dict1, '_metadata'):
            del state_dict1._metadata
        for key in list(state_dict1.keys()):
            self.__patch_instance_norm_state_dict(state_dict1, net1, key.split('.'))
        info = net1.load_state_dict(state_dict1)
        
        net2 = self.netG_A2
        if isinstance(net2, torch.nn.DataParallel):
            net2 = net2.module
        state_dict2 = torch.load((load_path2), map_location="cpu")
        if hasattr(state_dict2, '_metadata'):
            del state_dict2._metadata
        for key in list(state_dict2.keys()):
            self.__patch_instance_norm_state_dict(state_dict2, net2, key.split('.'))
        info = net2.load_state_dict(state_dict2)  
    