
import mixing_params
import torch
import numpy as np
np.random.seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)
torch.autograd.set_detect_anomaly(True)
from torch.autograd import Variable
import models
import variational
import kernels
import priors
import math
import means
import distributions
from module import Module
import utils
import likelihoods
from torch.optim import SGD
import mlls
import time
import os
from sklearn import metrics
import pandas as pd
import sys


video_frames = pd.read_csv("video_frames.csv")
video_frame_map = {}
videos_ = video_frames['video'].tolist()
frames_ = video_frames['frames'].tolist()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Being used:", device)

def gettemporalannotation(file):
    video_gt_map = {}
    for row in file:
        [video_name, anomaly_type, startanof1, endanof1, startanof2, endanof2, _] = row.replace('\n', '').split('  ')
        video_gt_map[video_name] = [int(startanof1), int(endanof1), int(startanof2), int(endanof2)]
    return video_gt_map 

temporal_annotation_file = open("Temporal_Anomaly_Annotation.txt")
video_gt_map = gettemporalannotation(temporal_annotation_file)

def getframeauc(model, weights, X_test_abnormal, X_test_normal, video_names_abnormal, video_names_normal):
    no_segments = X_test_abnormal.shape[1]
    test_feat_abnormal = torch.from_numpy(X_test_abnormal)
    test_feat_normal = torch.from_numpy(X_test_normal)
    test_feat_abnormal = Variable(test_feat_abnormal).to(device)
    test_feat_normal = Variable(test_feat_normal).to(device)
    functional_dist_abnormal = model(test_feat_abnormal)
    predictions_abnormal = functional_dist_abnormal.mean@weights
    predictions_abnormal = 1/(1+torch.exp(-predictions_abnormal)) 
    functional_dist_normal = model(test_feat_normal)
    predictions_normal = functional_dist_normal.mean@weights
    
    predictions_normal = 1/(1+torch.exp(-predictions_normal))
    predictions_normal = predictions_normal.data.cpu().numpy().flatten()
    predictions_abnormal = predictions_abnormal.data.cpu().numpy().flatten()
    predictions_normal = predictions_normal.reshape(-1, no_segments)
    predictions_abnormal = predictions_abnormal.reshape(-1, no_segments)
    GT, Pred = [], []
    anomaly_types = ['abnormal', 'normal']
    for anomaly_type in anomaly_types:
        if anomaly_type=='abnormal':
            video_names = video_names_abnormal
        else:
            video_names = video_names_normal
        for i, video_multimodal in enumerate(video_names):
            if anomaly_type == 'abnormal':
                prediction_multimodal = predictions_abnormal[i]
            else:
                prediction_multimodal  = predictions_normal[i]
            
            for j, video in enumerate(video_multimodal):
                prediction = prediction_multimodal[j*32:(j+1)*32]
                
                no_clips = 189
                thirty2_shots = np.round(np.linspace(0, no_clips-1, 33))
                p_c = 0
                clip_pred_score = np.zeros(no_clips)
                for ishots in range(0, len(thirty2_shots)-1):
                    ss = int(thirty2_shots[ishots])
                    ee = int(thirty2_shots[ishots+1])
                    if ee<ss or ee==ss:
                        clip_pred_score[ss] = prediction[p_c]
                    else:
                        clip_pred_score[ss:ee] = prediction[p_c]
                    p_c+=1
                number_frames = frames_[videos_.index(video)]
                if anomaly_type=="normal":
                    val = np.zeros(number_frames)
                    GT.extend(val.tolist())
                else:
                    [start1, end1, start2, end2] = video_gt_map[video+".mp4"]
                    val = np.zeros(number_frames)
                    val[start1-1:end1] = 1
                    val[start2-1:end2] = 1
                    GT.extend(val.tolist())

                p_c = 0
                frame_pred = np.zeros(number_frames)
                oneeightynine_shots = np.round(np.linspace(0, number_frames-1, 190))
                for ishots in range(0, len(oneeightynine_shots)-1):
                    ss = int(oneeightynine_shots[ishots])
                    ee = int(oneeightynine_shots[ishots+1])
                    if ee<ss or ee==ss:
                        frame_pred[ss] = clip_pred_score[p_c]
                    else:
                        frame_pred[ss:ee] = clip_pred_score[p_c]
                    p_c+=1
           
                Pred.extend(frame_pred.tolist())
    GT = np.array(GT)
    Pred = np.array(Pred)
    fpr, tpr, thresholds = metrics.roc_curve (GT, Pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc  




class DNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 32)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.6)
        self.fc2 = torch.nn.Linear(32, 16)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class GaussianProcessLayer(models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds = (-100., 100.), grid_size = 100):
        variational_distribution = variational.CholeskyVariationalDistribution(num_inducing_points = grid_size, batch_shape = torch.Size([num_dim]))
        variational_strategy = variational.MultitaskVariationalStrategy(variational.GridInterpolationVariationalStrategy(self, grid_size = grid_size, grid_bounds = [grid_bounds], variational_distribution = variational_distribution, ), num_tasks = num_dim,)
        super().__init__(variational_strategy)
        
        self.covar_module = kernels.ScaleKernel(kernels.RBFKernel(lengthscale_priors = priors.SmoothedBoxPrior(math.exp(-1), math.exp(1), sigma = 0.1, transform = torch.exp)))
        self.mean_module = means.ConstantMean()
        self.grid_bounds = grid_bounds
    
    def forward(self, x):
       
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return distributions.MultivariateNormal(mean, covar)
    
class DKLModel(Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds = (-100., 100.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim = num_dim, grid_bounds = grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim
    def forward(self, x):
        features = self.feature_extractor(x)
        features = utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # please double check what is happening here
        features = features.transpose(-1, -2).unsqueeze(-1)
        
        res = self.gp_layer(features)
        return res
    
  
def evaluate(model,  weights, aucs, best_auc, X_test_abnormal, X_test_normal, video_names_abnormal_test, video_names_normal_test, no_segments, act1, act2, act3, rep, eps=0.01):
    auc = getframeauc(model, weights, X_test_abnormal, X_test_normal, video_names_abnormal_test, video_names_normal_test)
    aucs.append(auc)
    if auc>best_auc:
        best_auc = auc             
        torch.save({'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),}, os.path.join("trained_models/UCF_Crime_Multimodal/model_"+act1+"_"+act2+"_"+act3+"_"+str(rep)+"_"+str(eps)+".pth.tar"))
        torch.save(weights, "trained_models/UCF_Crime_Multimodal/likelihood_weight_"+act1+"_"+act2+"_"+act3+"_"+str(rep)+"_"+str(eps)+".pt")
    return aucs, best_auc


if __name__=="__main__":
    [_, rep, eps] = sys.argv    
    rep = int(rep)
    eps = float(eps)
        
    start_time = time.time()
    pairs = [['Shoplifting', 'RoadAccidents', 'Stealing']]
    selected_pair = pairs[0]
    act1, act2, act3 = selected_pair[0], selected_pair[1], selected_pair[2]
    X_train_abnormal, X_train_normal = np.load("Dataset/UCF_Crime_Multimodal/Multimodal_Pairs/X_train_abnormal_multimodal_"+act1+"_"+act2+"_"+act3+".npy"), \
    np.load("Dataset/UCF_Crime_Multimodal/Multimodal_Pairs/X_train_normal_multimodal_"+act1+"_"+act2+"_"+act3+".npy")
    X_test_abnormal, X_test_normal = np.load("Dataset/UCF_Crime_Multimodal/Multimodal_Pairs/X_test_abnormal_multimodal_"+act1+"_"+act2+"_"+act3+".npy"), np.load("Dataset/UCF_Crime_Multimodal/Multimodal_Pairs/X_test_normal_multimodal_"+act1+"_"+act2+"_"+act3+".npy")
    video_names_abnormal_test, video_names_normal_test = np.load("Dataset/UCF_Crime_Multimodal/Multimodal_Pairs/video_names_test_abnormal_multimodal_"+act1+"_"+act2+"_"+act3+".npy", allow_pickle = True), np.load("Dataset/UCF_Crime_Multimodal/Multimodal_Pairs/video_names_test_normal_multimodal_"+act1+"_"+act2+"_"+act3+".npy", allow_pickle = True)   
    video_names_abnormal_train, video_names_normal_train = np.load("Dataset/UCF_Crime_Multimodal/Multimodal_Pairs/video_names_train_abnormal_multimodal_"+act1+"_"+act2+"_"+act3+".npy", allow_pickle = True), np.load("Dataset/UCF_Crime_Multimodal/Multimodal_Pairs/video_names_train_normal_multimodal_"+act1+"_"+act2+"_"+act3+".npy", allow_pickle = True)   
    new_vid_abnormal_train, new_vid_normal_train = [], []
    for videos in video_names_abnormal_train:
        new_vid_abnormal_train.append('_'.join(videos))
    video_names_abnormal_train = np.array(new_vid_abnormal_train)
    
    for videos in video_names_normal_train:
        new_vid_normal_train.append('_'.join(videos))
    video_names_normal_train = np.array(new_vid_normal_train)
    X_test = np.concatenate([X_test_abnormal, X_test_normal])
    #X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])
    video_test = np.concatenate([video_names_abnormal_test, video_names_normal_test])
    no_segments = X_train_abnormal.shape[1]
    input_dim = X_train_abnormal.shape[2]
    
    X_train = np.concatenate([X_train_abnormal, X_train_normal])
    video_train = np.concatenate([video_names_abnormal_train, video_names_normal_train])
    
    batch_size = 5
    
    no_videos = len(video_train)
    max_iterations = 10000
    
    mixing_parameters = mixing_params.MixingParameters(no_videos, no_segments, video_names_abnormal_train, video_names_normal_train, eps = eps)
    dnn = DNN(input_dim = input_dim)
    pretrained = torch.load("pretrained_ucfcrime_multimodal/model_"+act1+"_"+act2+"_"+act3+"_top1.pth.tar")
    model_dict = dnn.state_dict()
    pretrained_dict = {k: v for k, v in pretrained['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    dnn.load_state_dict(model_dict)
    
    num_features = dnn.fc2.out_features
    model = DKLModel(feature_extractor = dnn, num_dim = num_features)
    likelihood = likelihoods.mil_likelihood.MILLikelihood(num_features = model.num_dim, num_classes = 1)
    
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    lr = 0.001
    optimizer = SGD([
            {'params': model.feature_extractor.parameters(), 'weight_decay': 0.001, 'lr': 0.001},
            {'params': model.gp_layer.hyperparameters(), 'lr': lr*100},
            {'params': model.gp_layer.variational_parameters(), 'lr': lr*100},
            {'params': likelihood.parameters(), 'lr': lr*100},
            
            ], lr = lr, momentum = 0.9, nesterov = True, weight_decay = 0)
    
    mll = mlls.VariationalELBO(likelihood, model.gp_layer, num_data = int((len(X_train_abnormal)+len(X_train_normal))*no_segments))
    
    losses = []
    kl_z_loss = []
    kl_u_loss = []
    exp_log_prob_loss = []
    best_auc = 0
    aucs = []
    idx = list(range(len(X_train)))
    
    for i in range(max_iterations):
        print("Working on an iteration", i)
        model.train()
        likelihood.train()
        np.random.shuffle(idx)
        train_feat = X_train[idx[:batch_size]]
        train_videos = video_train[idx[:batch_size]]
        r_dist = mixing_parameters.get_r(train_videos)
        pi_dist = mixing_parameters.get_pi(train_videos)
        
        seg_bag_idx = np.repeat(train_videos, no_segments)
        seg_bag_idx = seg_bag_idx.flatten()
        bag_labels_gt = np.ones((len(train_videos), no_segments))
        for j, video in enumerate(train_videos):
            if video in video_names_normal_train:
                temp = np.zeros(no_segments)
                temp[:] = -1
                bag_labels_gt[j] = temp
        bag_labels_gt = bag_labels_gt.flatten()
        bag_labels_gt = torch.from_numpy(bag_labels_gt)
        bag_labels_gt = bag_labels_gt.reshape(int(len(bag_labels_gt)/no_segments), no_segments)
        
        if torch.cuda.is_available():
            r_dist = Variable(r_dist).cuda()
            pi_dist = Variable(pi_dist).cuda()
            bag_labels_gt = Variable(bag_labels_gt).cuda()
            
        train_feat = train_feat.reshape(train_feat.shape[0]*train_feat.shape[1], train_feat.shape[2])
        train_video_names = train_videos#np.concatenate([train_video_abnormal, train_video_normal])
        train_feat = torch.from_numpy(train_feat)
        if torch.cuda.is_available():
            train_feat = Variable(train_feat, requires_grad = True).cuda()
        optimizer.zero_grad()
        [loss, exp_loss, kl_u, kl_z] = mll(model(train_feat), r_dist, pi_dist, seg_bag_idx, bag_labels_gt, mixing_parameters, train_videos)
        loss = -loss
        kl_z_loss.append(kl_z.data.cpu())
        kl_u_loss.append(kl_u.data.cpu())
        exp_log_prob_loss.append(exp_loss.data.cpu())
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu())
        
        if i%10==0:
            print("Performing Evaluation on the iteration", i)
            model.eval()
            likelihood.eval()
            [aucs, best_auc] = evaluate(model,  likelihood.mixing_weights.t(), aucs, best_auc, X_test_abnormal, X_test_normal, video_names_abnormal_test, video_names_normal_test, no_segments, act1, act2, act3, rep, eps=eps)
            np.save("logs/UCF_Crime_Multimodal/aucs_test_"+str(act1)+"_"+str(act2)+"_"+str(act3)+"_"+str(rep)+"_"+str(eps)+".npy", aucs)
            np.save("logs/UCF_Crime_Multimodal/losses_"+str(act1)+"_"+str(act2)+"_"+str(act3)+"_"+str(rep)+"_"+str(eps)+".npy", losses)
            np.save("logs/UCF_Crime_Multimodal/kl_loss_z_"+str(act1)+"_"+str(act2)+"_"+str(act3)+"_"+str(rep)+"_"+str(eps)+".npy", kl_z_loss)
            np.save("logs/UCF_Crime_Multimodal/kl_loss_u_"+str(act1)+"_"+str(act2)+"_"+str(act3)+"_"+str(rep)+"_"+str(eps)+".npy", kl_u_loss)
            np.save("logs/UCF_Crime_Multimodal/exp_prob_loss_"+str(act1)+"_"+str(act2)+"_"+str(act3)+"_"+str(rep)+"_"+str(eps)+".npy", exp_log_prob_loss)
            print("Working on an iteration")
            print("Best AUC Testing Score", best_auc)
    end_time = time.time()
    np.save("logs/UCF_Crime_Multimodal/time_"+str(act1)+"_"+str(act2)+"_"+str(act3)+"_"+str(rep)+"_"+str(eps)+".npy", [end_time-start_time])
        
        
    
    
    
    
    
    
    
