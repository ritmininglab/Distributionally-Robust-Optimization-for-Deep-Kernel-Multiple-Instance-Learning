
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
from find_frame_auc import getframeauc
import os
import sys

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
    
 
def evaluate(model, weights, best_auc_val, best_auc_test, aucs_val, aucs_test, split,\
             X_test_abnormal, X_test_normal, X_val_abnormal, X_val_normal, video_names_abnormal_test, video_names_normal_test, video_names_abnormal_val, video_names_normal_val, rep, eps = 0.1):
        
        [auc_val, _, _] = getframeauc(model, weights, X_val_abnormal, X_val_normal, video_names_abnormal_val, video_names_normal_val)
        [auc_test, _, _] = getframeauc(model, weights, X_test_abnormal, X_test_normal, video_names_abnormal_test, video_names_normal_test)
        aucs_val.append(auc_val)
        aucs_test.append(auc_test)
        
        if best_auc_val<auc_val:
            best_auc_val = auc_val
            best_auc_test = auc_test
            torch.save({'state_dict': model.state_dict(), 
                        'opt_dict': optimizer.state_dict(),}, os.path.join("trained_models/SanghaiTech_Outlier/model_"+str(rep)+"_"+str(split)+"_"+str(eps)+".pth.tar"))
            torch.save(weights, "trained_models/SanghaiTech_Outlier/likelihood_weight_"+str(rep)+"_"+str(split)+"_"+str(eps)+".pt")
        return [aucs_val, best_auc_val, aucs_test, best_auc_test]
            
        
        
        

if __name__=="__main__":
    

    [_, split, rep, eps] = sys.argv
    split = int(split)
    eps = float(eps)
    rep = int(rep)
    batch_size = 5
    
    X_train_abnormal, X_train_normal = np.load("Dataset/SanghaiTech/X_train_abnormal_outlier.npy"), np.load("Dataset/SanghaiTech/X_train_normal.npy")
    video_names_abnormal_train, video_names_normal_train = np.load("Dataset/SanghaiTech/video_train_abnormal.npy", allow_pickle = True), np.load("Dataset/SanghaiTech/video_train_normal.npy", allow_pickle = True)
    
    X_val_abnormal, X_val_normal = np.load("Dataset/SanghaiTech/X_val_abnormal_"+str(split)+".npy"), np.load("Dataset/SanghaiTech/X_val_normal_"+str(split)+".npy")
    X_test_abnormal, X_test_normal = np.load("Dataset/SanghaiTech/X_test_abnormal_"+str(split)+".npy"), np.load("Dataset/SanghaiTech/X_test_normal_"+str(split)+".npy")
    
    video_names_abnormal_test, video_names_normal_test = np.load("Dataset/SanghaiTech/video_test_abnormal_"+str(split)+".npy", allow_pickle = True), np.load("Dataset/SanghaiTech/video_test_normal_"+str(split)+".npy", allow_pickle = True)
    video_names_abnormal_val, video_names_normal_val = np.load("Dataset/SanghaiTech/video_val_abnormal_"+str(split)+".npy", allow_pickle = True), np.load("Dataset/SanghaiTech/video_val_normal_"+str(split)+".npy", allow_pickle = True)
    

    no_videos = len(X_train_abnormal)+len(X_train_normal)
    no_segments = X_train_abnormal.shape[1]
    max_iterations = 10000
    input_dim = X_train_abnormal.shape[2]
    abnormal_idx = list(range(len(X_train_abnormal)))
    normal_idx = list(range(len(X_train_normal)))
    
    mixing_parameters = mixing_params.MixingParameters(no_videos, no_segments, video_names_abnormal_train, video_names_normal_train, eps = eps)
    
    dnn = DNN(input_dim = input_dim)
    pretrained = torch.load("pretrained_sanghaitech_outlier/model_top3_"+str(split)+".pth.tar", map_location = lambda storage, loc: storage)
    model_dict = dnn.state_dict()
    pretrained_dict = {k: v for k, v in pretrained['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    dnn.load_state_dict(model_dict)
    
    num_features = dnn.fc2.out_features
    model = DKLModel(feature_extractor = dnn, num_dim = num_features)
    likelihood = likelihoods.mil_likelihood.MILLikelihood(num_features = model.num_dim, num_classes = 1)
    
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood  = likelihood.cuda()
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
    best_auc_eval = 0
    best_auc_test = 0
    aucs_eval = []
    aucs_test = []
    X_train = np.concatenate([X_train_abnormal, X_train_normal])
    videos = np.concatenate([video_names_abnormal_train, video_names_normal_train])
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
            [aucs_eval, best_auc_eval, aucs_test, best_auc_test] = evaluate(model,  likelihood.mixing_weights.t(), best_auc_eval, best_auc_test, aucs_eval, aucs_test, split, \
            X_test_abnormal, X_test_normal, X_val_abnormal, X_val_normal, video_names_abnormal_test, video_names_normal_test, video_names_abnormal_val, video_names_normal_val, rep = rep, eps=eps)
            np.save("SanghaiTech_Outlier/logs/aucs_eval_"+str(rep)+"_"+str(split)+"_"+str(eps)+".npy", aucs_eval)
            np.save("SanghaiTech_Outlier/logs/aucs_test_"+str(rep)+"_"+str(split)+"_"+str(eps)+".npy", aucs_test)
            np.save("SanghaiTech_Outlier/logs/losses_"+str(rep)+"_"+str(split)+"_"+str(eps)+".npy", losses)
            np.save("SanghaiTech_Outlier/logs/kl_loss_z_"+str(rep)+"_"+str(split)+"_"+str(eps)+".npy", kl_z_loss)
            np.save("SanghaiTech_Outlier/logs/kl_loss_u_"+str(rep)+"_"+str(split)+"_"+str(eps)+".npy", kl_u_loss)
            np.save("SanghaiTech_Outlier/logs/exp_prob_loss_"+str(rep)+"_"+str(split)+"_"+str(eps)+".npy", exp_log_prob_loss)
           
       
            
            
            
        
        
        
        
        
        
    
    
    
    
    
    
