import os
import csv
import numpy as np
import math
import torch
from PIL import Image
from models.ASDPAD import *
from data.dataset import CNNDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def write_csv(results, file_name, mode = 'w', row_list = ['id', 'score', 'label']):
    import csv
    with open(file_name, mode, newline='') as f:
        writer = csv.writer(f)
        if (row_list != ""):
            writer.writerow(row_list)
        writer.writerows(results)
        
def get_threshold(probs, grid_density):
    Min, Max = min(probs), max(probs)
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(0.0 + i * 1.0 / float(grid_density))
    thresholds.append(1.1)
    return thresholds

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def get_EER_states(probs, labels, grid_density = 10000):
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if(FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif(FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list
    
def get_trainAndValDataset_ratio(original_dataset, ratio=0.9, random_seed=802):
    dataset_size = len(original_dataset)
    train_dataset, val_dataset = random_split(
        dataset=original_dataset,
        lengths=[int(ratio*dataset_size), dataset_size - int(ratio*dataset_size)],
        generator=torch.Generator().manual_seed(random_seed)
    )      
    return train_dataset, val_dataset

def get_iter_len(src_train_dataloader):
    src_train_iter = iter(src_train_dataloader)
    src_iter_per_epoch = len(src_train_iter)
    return src_iter_per_epoch

def get_domain_genuine_recaptured(opt, src_genuine_csv, src_recaptured_csv):
    src_dataset_genuine = CNNDataset(src_genuine_csv, train=True)
    src_train_dataset_genuine, src_val_dataset_genuine = get_trainAndValDataset_ratio(src_dataset_genuine, ratio=0.9, random_seed=802)
    src_train_dataloader_genuine = DataLoader(src_train_dataset_genuine, batch_size=opt.batch_size, shuffle=True)
    src_iter_per_epoch_genuine = get_iter_len(src_train_dataloader_genuine)
    
    src_dataset_recaptured = CNNDataset(src_recaptured_csv, train=True)
    src_train_dataset_recaptured, src_val_dataset_recaptured = get_trainAndValDataset_ratio(src_dataset_recaptured, ratio=0.9, random_seed=802)
    src_train_dataloader_recaptured = DataLoader(src_train_dataset_recaptured, batch_size=opt.batch_size, shuffle=True)
    src_iter_per_epoch_recaptured = get_iter_len(src_train_dataloader_recaptured)
    
    return src_train_dataloader_genuine, src_train_dataloader_recaptured, src_iter_per_epoch_genuine, src_iter_per_epoch_recaptured, src_val_dataset_genuine, src_val_dataset_recaptured

def get_surrogate_data_label(net, genuine_data):
    surrogate_data1, surrogate_label1, surrogate_data2, surrogate_label2 = net.forward(genuine_data)
    surrogate_data1 = surrogate_data1.cuda()
    surrogate_label1 = surrogate_label1.cuda()
    surrogate_data2 = surrogate_data2.cuda()
    surrogate_label2 = surrogate_label2.cuda()
    return surrogate_data1, surrogate_label1, surrogate_data2, surrogate_label2

def get_advLoss_surrogateVgg(opt, net, genuine_data, genuine_data_label, stimulatedData, surrogateData, surrogateLabel):
    trainInput = np.empty((0,3,224,224)).astype(np.float32)
    trainInput = torch.from_numpy(trainInput).cuda()
    target = np.empty((0)).astype(np.float32)
    target = torch.from_numpy(target).cuda()
                    
    trainInput = torch.cat((trainInput, genuine_data, surrogateData), dim=0)
    target = torch.cat((target, genuine_data_label, surrogateLabel), dim=0)
    
    advScore, _ = net.SSDG(trainInput, opt.norm_flag)
    
    advLoss = net.criterion_softmax(advScore, target.long())
    surrogateVgg = net.criterion_vgg19(stimulatedData, surrogateData)
    return advLoss, surrogateVgg


def test(opt, model, dataloader, csv_name):
    model.eval()
    results = []
    confusion_matrix = meter.ConfusionMeter(2)
    
    with torch.no_grad():
        for ii, (val_input, img_name, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            val_input = val_input.cuda()
            label = label.tolist()
            score, _ = model(val_input, opt.norm_flag)
            output = F.softmax(score, dim=1)[:, 1].detach().tolist()
            batch_results = [(img_name_, output_, label_) for img_name_, output_, label_ in zip(img_name, output, label)]
            results += batch_results
        try:
            write_csv(results, csv_name)
        except:
            write_csv(results, opt.description + "_temp.csv")    
        auc, eer, eer_thr = get_test_results(csv_name)
        model.train()
        return auc, eer, eer_thr
    
def get_auc_thrs(scores, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    return AUC, fpr, tpr, thresholds

def get_eer_thr(scores, labels):  
    scores_np = np.array(scores)
    labels_np = np.array(labels)
    EER, eer_thr, _, _ = get_EER_states(scores_np, labels_np)
    return EER, eer_thr    

def get_test_results(csv_name):
    results, scores, labels = get_single_img_scores_and_labels(csv_name)
    eer, eer_thr = get_eer_thr(scores, labels)
    auc, _, _, _ = get_auc_thrs(scores, labels)
    return auc, eer, eer_thr

def get_single_img_scores_and_labels(csv_name):
    detectionPatches_scores = []
    detectionImages={}
    with open(csv_name, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            detectionPatch = row[0]
            score = float(row[1])
            label = int(row[2])
            detectionImage = detectionPatch.rsplit("_", 1)[0]
            if detectionImage not in detectionImages.keys():
                detectionImages[detectionImage] = {"score": score, "count": 1, "label": label}
            else:
                detectionImages[detectionImage]["score"] += score
                detectionImages[detectionImage]["count"] += 1
    results = []
    scores = []
    labels = []

    for detectionImage in detectionImages.keys():
        results.append(detectionImage)
        scores.append(detectionImages[detectionImage]["score"] / detectionImages[detectionImage]["count"])
        labels.append(detectionImages[detectionImage]["label"])
    return results, scores, labels