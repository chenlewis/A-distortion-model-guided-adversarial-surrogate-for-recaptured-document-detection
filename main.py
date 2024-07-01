import os
import time
import torch
from torch.utils.data import DataLoader
from data.dataset import CNNDataset
from util.util import *
from util.visualizer import Visualizer
from options.train_options import TrainOptions
from models.ASDPAD import ASDPAD
from collections import OrderedDict
from loss.AdLoss import AdLoss_Limited
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')

# https://github.com/taylover-pei/SSDG-CVPR2020
# https://github.com/junyanz/CycleGAN

if __name__ == '__main__':
    opt = TrainOptions().parse()
    visualizer = Visualizer(opt)
    net = ASDPAD(opt, visualizer)
    original_net = ASDPAD(opt, visualizer)
    
    # Ink
    src1_train_dataloader_genuine, src1_train_dataloader_recaptured, \
    src1_iter_per_epoch_genuine, src1_iter_per_epoch_recaptured, _, _ = get_domain_genuine_recaptured(opt, opt.src1_genuine_csv, opt.src1_recaptured_csv)
    # Laser
    src2_train_dataloader_genuine, src2_train_dataloader_recaptured, \
    src2_iter_per_epoch_genuine, src2_iter_per_epoch_recaptured, _, _ = get_domain_genuine_recaptured(opt, opt.src2_genuine_csv, opt.src2_recaptured_csv)
    # test data
    test_data = CNNDataset(opt.test_csv, train=False)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=opt.num_threads)
    
    epoch = 1
    total_iters = 0
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0
    
    for iter_num in range(opt.max_iter+1):
        if (iter_num % src1_iter_per_epoch_genuine == 0):
            src1_train_iter_genuine = iter(src1_train_dataloader_genuine)
        if (iter_num % src2_iter_per_epoch_genuine == 0):
            src2_train_iter_genuine = iter(src2_train_dataloader_genuine)
        if (iter_num % src1_iter_per_epoch_recaptured == 0):
            src1_train_iter_recaptured = iter(src1_train_dataloader_recaptured)
        if (iter_num % src2_iter_per_epoch_recaptured == 0):
            src2_train_iter_recaptured = iter(src2_train_dataloader_recaptured)
            
        iter_start_time = time.time()
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        # 
        if (iter_num != 0 and iter_num % opt.iter_per_epoch == 0):
            epoch = epoch + 1
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0
            visualizer.reset()
        
        # genuine
        src1_img_genuine, src1_label_genuine = src1_train_iter_genuine.next()
        src1_img_genuine = src1_img_genuine.cuda(opt.gpu_ids[0])
        src1_label_genuine = src1_label_genuine.cuda(opt.gpu_ids[0])
        input1_genuine_shape = src1_img_genuine.shape[0]
        
        src2_img_genuine, src2_label_genuine = src2_train_iter_genuine.next()
        src2_img_genuine = src2_img_genuine.cuda(opt.gpu_ids[0])
        src2_label_genuine = src2_label_genuine.cuda(opt.gpu_ids[0])
        input2_genuine_shape = src2_img_genuine.shape[0]

        # surrogate
        src1_img_surrogate_ink, src1_label_surrogate_ink, src1_img_surrogate_laser, src1_label_surrogate_laser = net.forward(src1_img_genuine)
        # recaptured
        src1_img_recaptured, src1_label_recaptured = src1_train_iter_recaptured.next()
        src1_img_recaptured = src1_img_recaptured.cuda(opt.gpu_ids[0])
        src1_label_recaptured = src1_label_recaptured.cuda(opt.gpu_ids[0])
        
        # surrogate
        src2_img_surrogate_ink, src2_label_surrogate_ink, src2_img_surrogate_laser, src2_label_surrogate_laser = net.forward(src2_img_genuine)
        # recaptured
        src2_img_recaptured, src2_label_recaptured = src2_train_iter_recaptured.next()
        src2_img_recaptured = src2_img_recaptured.cuda(opt.gpu_ids[0])
        src2_label_recaptured = src2_label_recaptured.cuda(opt.gpu_ids[0])
        
        input_data = torch.cat([src1_img_genuine, src1_img_recaptured, src1_img_surrogate_ink, src2_img_surrogate_ink,
                                src2_img_genuine, src2_img_recaptured, src1_img_surrogate_laser, src2_img_surrogate_laser], dim=0)        
        source_label = torch.cat([src1_label_genuine, src1_label_recaptured, src1_label_surrogate_ink, src2_label_surrogate_ink,
                                  src2_label_genuine, src2_label_recaptured, src1_label_surrogate_laser, src2_label_surrogate_laser], dim=0)
        input1_recaptured_shape = src1_img_recaptured.shape[0] + src1_img_surrogate_ink.shape[0] + src2_img_surrogate_ink.shape[0]
        input2_recaptured_shape = src2_img_recaptured.shape[0] + src1_img_surrogate_laser.shape[0] + src2_img_surrogate_laser.shape[0]      

        classifier_label_out, feature = net.SSDG(input_data, opt.norm_flag)
        
        input1_shape = input1_genuine_shape + input1_recaptured_shape
        input2_shape = input2_genuine_shape + input2_recaptured_shape
        feature_genuine_1 = feature.narrow(0, 0, input1_genuine_shape)
        feature_genuine_2 = feature.narrow(0, input1_shape, input2_genuine_shape)
        feature_genuine = torch.cat([feature_genuine_1, feature_genuine_2], dim=0)
        discriminator_out_genuine = net.ad_net_genuine(feature_genuine)
        
        genuine_domain_label_1 = torch.LongTensor(input1_genuine_shape, 1).fill_(0).cuda(opt.gpu_ids[0])
        genuine_domain_label_2 = torch.LongTensor(input2_genuine_shape, 1).fill_(0).cuda(opt.gpu_ids[0])
        recaptured_domain_label_1 = torch.LongTensor(input1_recaptured_shape, 1).fill_(1).cuda(opt.gpu_ids[0])
        recaptured_domain_label_2 = torch.LongTensor(input2_recaptured_shape, 1).fill_(2).cuda(opt.gpu_ids[0])
        source_domain_label = torch.cat([genuine_domain_label_1, recaptured_domain_label_1,
                                         genuine_domain_label_2, recaptured_domain_label_2], dim=0).view(-1).cuda(opt.gpu_ids[0])
        triplet = net.criterion_triplet(feature, source_domain_label)

        genuine_shape_list = []
        genuine_shape_list.append(input1_genuine_shape)
        genuine_shape_list.append(input2_genuine_shape)
        genuine_adloss = AdLoss_Limited(discriminator_out_genuine, net.criterion_softmax, genuine_shape_list)
        cls_loss = net.criterion_softmax(classifier_label_out.narrow(0, 0, input_data.size(0)), source_label)
        
        net.update_optimizeSSDGAndAdgenuine_parameters(cls_loss, triplet, genuine_adloss)
        
        if iter_num % opt.surrogateUpdateFreq == 0:
            for k in range(opt.iter_num):
                src1_surrogate_data1, src1_surrogate_label1, src1_surrogate_data2, src1_surrogate_label2 = get_surrogate_data_label(net, src1_img_genuine)
                src1_original_surrogate_data1, _, src1_original_surrogate_data2, _ = get_surrogate_data_label(original_net, src1_img_genuine)

                src2_surrogate_data1, src2_surrogate_label1, src2_surrogate_data2, src2_surrogate_label2 = get_surrogate_data_label(net, src2_img_genuine)
                src2_original_surrogate_data1, _, src2_original_surrogate_data2, _ = get_surrogate_data_label(original_net, src2_img_genuine)

                src1_adv_cnn_loss1, src1_surrogate1_vgg = get_advLoss_surrogateVgg(opt, net, src1_img_genuine, src1_label_genuine, src1_original_surrogate_data1, src1_surrogate_data1, src1_surrogate_label1)
                src2_adv_cnn_loss1, src2_surrogate1_vgg = get_advLoss_surrogateVgg(opt, net, src2_img_genuine, src2_label_genuine, src2_original_surrogate_data1, src2_surrogate_data1, src2_surrogate_label1)
                surrogate1_loss = abs(opt.ink_1*src1_surrogate1_vgg - opt.ink_2*src1_adv_cnn_loss1 + opt.ink_1*src2_surrogate1_vgg - opt.ink_2*src2_adv_cnn_loss1)
                net.update_optimizeG1_parameters(surrogate1_loss)

                src1_adv_cnn_loss2, src1_surrogate2_vgg = get_advLoss_surrogateVgg(opt, net, src1_img_genuine, src1_label_genuine, src1_original_surrogate_data2, src1_surrogate_data2, src1_surrogate_label2)
                src2_adv_cnn_loss2, src2_surrogate2_vgg = get_advLoss_surrogateVgg(opt, net, src2_img_genuine, src2_label_genuine, src2_original_surrogate_data2, src2_surrogate_data2, src2_surrogate_label2)
                surrogate2_loss = abs(opt.laser_1*src1_surrogate2_vgg - opt.laser_2*src1_adv_cnn_loss2 + opt.laser_1*src2_surrogate2_vgg - opt.laser_2*src2_adv_cnn_loss2)
                net.update_optimizeG2_parameters(surrogate2_loss)
                
        if iter_num % opt.print_freq == 0:
            errors_ret = OrderedDict()
            errors_ret["surrogate1_loss"] = surrogate1_loss.detach().cpu().numpy()
            errors_ret["surrogate2_loss"] = surrogate2_loss.detach().cpu().numpy()
            errors_ret["cls_loss"] = cls_loss.detach().cpu().numpy()
            errors_ret["triplet"] = triplet.detach().cpu().numpy()
            errors_ret["genuine_adloss"] = genuine_adloss.detach().cpu().numpy()

            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.noConsole_print_current_losses(epoch, epoch_iter, errors_ret, t_comp, t_data)
            
        iter_data_time = time.time()
        if (iter_num != 0 and (iter_num+1) % opt.iter_per_epoch == 0):
            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                torch.save(net.netG_A1.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, "{}_net_G_A.pth".format(epoch)))
                torch.save(net.netG_A2.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, "{}_net_G_B.pth".format(epoch)))
                torch.save(net.SSDG.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, opt.cnn_model + "-epoch{}.pth".format(epoch))) 

                result_csv_name = os.path.join(opt.checkpoints_dir, opt.name, opt.cnn_model + "-epoch{}_test_result.csv".format(epoch))
                test_auc, test_eer, _ = test(opt, net.SSDG, test_dataloader, result_csv_name)
                visualizer.print_message("AUC:{}".format(test_auc))
                visualizer.print_message("EER:{}".format(test_eer))
                result = [(epoch, test_auc, test_eer)]
                write_csv(result, os.path.join(opt.checkpoints_dir, opt.name, "test.csv"),
                          mode = 'a', row_list = ['epoch', 'test_auc', 'test_eer'])

            visualizer.print_message('End of epoch %d \t Time Taken: %d sec' % (epoch, time.time() - epoch_start_time))