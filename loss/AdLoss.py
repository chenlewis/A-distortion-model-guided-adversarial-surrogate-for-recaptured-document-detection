import torch
import torch.nn as nn
# https://github.com/taylover-pei/SSDG-CVPR2020
def AdLoss_Limited(discriminator_out, criterion, shape_list):
    ad_label2_index = torch.LongTensor(shape_list[0], 1).fill_(0)
    ad_label2 = ad_label2_index.cuda()
    ad_label3_index = torch.LongTensor(shape_list[1], 1).fill_(1)
    ad_label3 = ad_label3_index.cuda()
    ad_label = torch.cat([ad_label2, ad_label3], dim=0).view(-1)
    genuine_adloss = criterion(discriminator_out, ad_label)
    return genuine_adloss