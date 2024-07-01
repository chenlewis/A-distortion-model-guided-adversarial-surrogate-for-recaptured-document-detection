import torch
import torchvision
import numpy as np
import torch.nn as nn

class VGG19Loss(nn.Module):
    
    def __init__(self):
        super(VGG19Loss, self).__init__()
        
        self.network = torchvision.models.vgg19(pretrained=True).features
        
        self.criterion = nn.L1Loss()
        
        self.layer_name_mapping = {
            '2': "relu1_1",
            '7': "relu2_1",
            '12': "relu3_1",
            '21': "relu4_1",
            '30': "relu5_1"
        }
        
    def select_features(self, x):
        output = {}
        
        for name, module in self.network._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x

        return list(output.values())
    
    def forward(self, output, gt):
        perceptual_losses = []
        style_losses = []
        
        output_features = self.select_features(output)
        gt_features = self.select_features(gt)

        for iter, (output_feature, gt_feature) in enumerate(zip(output_features, gt_features)):
            # perceptual_losses
            perceptual_losses.append(self.criterion(output_feature, gt_feature))
            # style_losses
            output_feature_flatten = output_feature.flatten(start_dim=2)
            output_feature_flatten_transpose = torch.transpose(output_feature_flatten, dim0=1, dim1=2)
            gt_feature_flatten = gt_feature.flatten(start_dim=2)
            gt_feature_flatten_transpose = torch.transpose(gt_feature_flatten, dim0=1, dim1=2)
            style_losses.append(self.criterion(torch.bmm(output_feature_flatten, output_feature_flatten_transpose), torch.bmm(gt_feature_flatten, gt_feature_flatten_transpose)))
    
        return sum(perceptual_losses) + 500 * sum(style_losses)/ (3*224*224)
    

    