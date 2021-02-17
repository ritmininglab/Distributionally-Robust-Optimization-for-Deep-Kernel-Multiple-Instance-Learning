
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
from sklearn import metrics
import os
import sys
import pandas as pd 
   
def gettemporalannotation(file):
    video_gt_map = {}
    for row in file:
        [video_name, anomaly_type, startanof1, endanof1, startanof2, endanof2, _] = row.replace('\n', '').split('  ')
        video_gt_map[video_name] = [int(startanof1), int(endanof1), int(startanof2), int(endanof2)]
    return video_gt_map 

temporal_annotation_file = open("Temporal_Anomaly_Annotation.txt")
video_gt_map = gettemporalannotation(temporal_annotation_file)
video_frames = pd.read_csv("video_frames.csv")
videos_ = video_frames['video'].tolist()
frames_ = video_frames['frames'].tolist()



class DNN(torch.nn.Module):
    def __init__(self, input_dim,hidden_size):
        super(DNN, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim,hidden_size,5,batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, 16)
        self.norm = torch.nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        
        x,_ = self.lstm(x)
        x = self.norm(x)
        x = self.fc1(x)
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
        features = features.reshape(features.shape[0]*features.shape[1], features.shape[2])
        features = utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # please double check what is happening here
        features = features.transpose(-1, -2).unsqueeze(-1)
        
        res = self.gp_layer(features)
        return res
    
 
def getframeauc(predictions_abnormal, predictions_normal, video_names_abnormal, video_names_normal):
    

    GT, Pred = [], []
    anomaly_types = ['abnormal', 'normal']
    for anomaly_type in anomaly_types:
        if anomaly_type=='abnormal':
            video_names = video_names_abnormal
        else:
            video_names = video_names_normal
        for i, video in enumerate(video_names):
            if anomaly_type=="abnormal":
                prediction = predictions_abnormal[i]
            else:
                prediction = predictions_normal[i]

            number_frames = frames_[videos_.index(video)]
            if anomaly_type=="normal":
                val = np.zeros(number_frames)
                GT.extend(val.tolist()[:(number_frames//189)*189])
            else:
                [start1, end1, start2, end2] = video_gt_map[video+".mp4"]
                val = np.zeros(number_frames)
                val[start1-1:end1] = 1
                val[start2-1:end2] = 1
                GT.extend(val.tolist()[:(number_frames//189)*189])
            
            total_shots = number_frames//189
            frame_pred = np.repeat(prediction, total_shots)
           
            Pred.extend(frame_pred.tolist())
    GT = np.array(GT)
    Pred = np.array(Pred)
    fpr, tpr, thresholds = metrics.roc_curve (GT, Pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc  
            
def evaluate(model,  weights, best_auc, aucs, X_test_abnormal, X_test_normal, video_names_test_abnormal, video_names_test_normal, eps, rep_no, no_segments = 189):
    test_feat_abnormal = torch.from_numpy(X_test_abnormal)
    test_feat_normal = torch.from_numpy(X_test_normal)
    test_feat_abnormal = Variable(test_feat_abnormal).to(device)
    test_feat_normal = Variable(test_feat_normal).to(device)
    functional_dist_abnormal = model(test_feat_abnormal)
    predictions_abnormal = functional_dist_abnormal.mean @ weights
    predictions_abnormal = 1/(1+torch.exp(-predictions_abnormal))
    functional_dist_normal = model(test_feat_normal)
    predictions_normal = functional_dist_normal.mean @weights
    predictions_normal = 1/(1+torch.exp(-predictions_normal))
    predictions_normal = predictions_normal.data.cpu().numpy().flatten()
    predictions_abnormal = predictions_abnormal.data.cpu().numpy().flatten()
    predictions_normal = predictions_normal.reshape(-1, no_segments)
    predictions_abnormal = predictions_abnormal.reshape(-1, no_segments)
    auc = getframeauc(predictions_abnormal, predictions_normal, video_names_test_abnormal, video_names_test_normal)
    aucs.append(auc)
    if auc>best_auc:
        best_auc = auc             
        torch.save({'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),}, os.path.join("trained_models/UCF_Crime/model_"+str(rep_no)+"_"+str(eps)+".pth.tar"))
        torch.save(weights, "trained_models/UCF_Crime/likelihood_weight_"+str(rep_no)+"_"+str(eps)+".pt")
    return aucs, best_auc
        
        

if __name__=="__main__":
    [_, rep_no, eps] = sys.argv
    rep_no = int(rep_no)
    eps = float(eps)

    X_train_abnormal, X_train_normal = np.load("Dataset/UCF_Crime/X_train_abnormal.npy", allow_pickle = True), np.load("Dataset/UCF_Crime/X_train_normal.npy")
    X_test_abnormal, X_test_normal = np.load("Dataset/UCF_Crime/X_test_abnormal.npy"), np.load("Dataset/UCF_Crime/X_test_normal.npy")
    video_names_test_abnormal, video_names_test_normal = np.load("Dataset/UCF_Crime/video_names_test_abnormal.npy"), np.load("Dataset/UCF_Crime/video_names_test_normal.npy")
    video_names_train_abnormal, video_names_train_normal = np.load("Dataset/UCF_Crime/video_names_train_abnormal.npy", allow_pickle = True), np.load("Dataset/UCF_Crime/video_names_train_normal.npy", allow_pickle = True) 
    batch_size = 5
    
    no_videos = len(X_train_abnormal)+len(X_train_normal)
    no_segments = X_train_abnormal.shape[1]
    max_iterations = 10000
    input_dim = X_train_abnormal.shape[2]
    abnormal_idx = list(range(len(X_train_abnormal)))
    normal_idx = list(range(len(X_train_normal)))
    
    mixing_parameters = mixing_params.MixingParameters(no_videos, no_segments, video_names_train_abnormal, video_names_train_normal, eps = eps)
    
    hidden_size = 189
    dnn = DNN(input_dim = input_dim, hidden_size = hidden_size)
    pretrained = torch.load("pretrained_ucfcrime/model_1_1.pth", map_location = lambda storage, loc: storage)
    model_dict = dnn.state_dict()
    pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    dnn.load_state_dict(model_dict)
    num_features = dnn.fc1.out_features
    model = DKLModel(feature_extractor = dnn, num_dim = num_features)
    likelihood = likelihoods.mil_likelihood.MILLikelihood(num_features = model.num_dim, num_classes = 1)
    
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood  = likelihood.cuda()
    lr = 0.001
    
    optimizer = SGD([
            {'params': model.feature_extractor.parameters(), 'weight_decay': 0.001, 'lr': 0.0001},
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
    X_train = np.concatenate([X_train_abnormal, X_train_normal])
    videos = np.concatenate([video_names_train_abnormal, video_names_train_normal])
    idx = list(range(len(X_train)))
    for i in range(max_iterations):
        print("Working on an iteration", i)
        model.train()
        likelihood.train()
        np.random.shuffle(idx)
        train_feat = X_train[idx[:batch_size]]
        train_videos = videos[idx[:batch_size]]
        
        r_dist = mixing_parameters.get_r(train_videos)
        pi_dist = mixing_parameters.get_pi(train_videos)
        
            
        seg_bag_idx = np.repeat(train_videos, no_segments)
        seg_bag_idx = seg_bag_idx.flatten()
        bag_labels_gt = np.ones((len(train_videos), no_segments))
        for j, video in enumerate(train_videos):
            if video in video_names_train_normal:
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
            
        
        
        
        train_video_names = train_videos
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
            [aucs, best_auc] = evaluate(model,  likelihood.mixing_weights.t(), best_auc, aucs, X_test_abnormal, X_test_normal, video_names_test_abnormal, video_names_test_normal, eps, rep_no, no_segments = no_segments)
            np.save("logs/UCF_Crime/aucs_"+str(rep_no)+"_"+str(eps)+".npy", aucs)
            np.save("logs/UCF_Crime/losses_"+str(rep_no)+"_"+str(eps)+".npy", losses)
            np.save("logs/UCF_Crime/kl_loss_z_"+str(rep_no)+"_"+str(eps)+".npy", kl_z_loss)
            np.save("logs/UCF_Crime/kl_loss_u_"+str(rep_no)+"_"+str(eps)+".npy", kl_u_loss)
            np.save("logs/UCF_Crime/exp_prob_loss_"+str(rep_no)+"_"+str(eps)+".npy", exp_log_prob_loss)
           
            
            
            
            
        
        
        
        
        
        
    
    
    
    
    
    
