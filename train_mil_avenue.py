
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


def getframeauc(predictions_abnormal, predictions_normal, video_names_test_abnormal, video_names_test_normal, no_segments):
    GT, Pred = [], []
    clip_size = 16
    video_names = np.concatenate([video_names_test_abnormal, video_names_test_normal])
    predictions = np.concatenate([predictions_abnormal, predictions_normal])
    video_gt = {}
    video_pred = {}

    for i, video in enumerate(video_names):
        video_gt[video] = []
        video_pred[video] = []
        prediction = predictions[i]

        if "train" in video:
            no_clips = np.load(os.path.join("Dataset/Avenue", "train", video+".npy")).shape[0]
        elif "test" in video:
            no_clips = np.load(os.path.join("Dataset/Avenue", "test", video+".npy")).shape[0]
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
        if "train" in video:
            idx = video.split("_")[1]
            val = np.load(os.path.join("Dataset/Avenue", "gts/train/"+idx+".npy"))
            no_frames = len(val)
            val = np.zeros(no_frames)
        elif "test" in video:
            idx = video.split("_")[1]
            val = np.load(os.path.join("Dataset/Avenue", "gts/test/"+idx+".npy"))
            no_frames = len(val)
        video_gt[video].extend(val.tolist())
        GT.extend(val.tolist())
        frame_pred = np.zeros(no_frames)
        for j in range(no_clips):
            start_frame = j*clip_size
            if (j+1)*clip_size>no_frames:
                end_frame = no_frames
            else:
                end_frame = (j+1)*clip_size
            frame_pred[start_frame: end_frame] = clip_pred_score[j]
        video_pred[video].extend(frame_pred)
        Pred.extend(frame_pred.tolist())
    GT = np.array(GT)
    Pred = np.array(Pred)
    fpr, tpr, thresholds = metrics.roc_curve(GT, Pred, pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc
    
    
def evaluate(model,  weights, best_auc, aucs, cv_no, X_test_abnormal, X_test_normal, video_names_abnormal_test, video_names_normal_test, eps, rep_no, no_segments):
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
    auc = getframeauc(predictions_abnormal, predictions_normal, video_names_abnormal_test, video_names_normal_test, no_segments)
    aucs.append(auc)
    if auc>best_auc:
        best_auc = auc             
        torch.save({'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),}, os.path.join("trained_models/Avenue/model_"+str(cv_no)+"_"+str(rep_no)+"_"+str(eps)+".pth.tar"))
        torch.save(weights, "trained_models/Avenue/likelihood_weight_"+str(cv_no)+"_"+str(rep_no)+"_"+str(eps)+".pt")
    return aucs, best_auc

    return aucs, best_auc  

if __name__=="__main__":
   
    [_, cv_no, rep_no, eps] = sys.argv
    cv_no =int(cv_no)
    rep_no = int(rep_no)
    eps = float(eps)
    cv_no = int(cv_no)
    batch_size = 5

    X_train_abnormal, X_train_normal = np.load("Dataset/Avenue/X_train_abnormal_"+str(cv_no)+".npy"), np.load("Dataset/Avenue/X_train_normal_"+str(cv_no)+".npy")
    X_test_abnormal, X_test_normal = np.load("Dataset/Avenue/X_test_abnormal_"+str(cv_no)+".npy"), np.load("Dataset/Avenue/X_test_normal_"+str(cv_no)+".npy")
    
    video_names_abnormal_train, video_names_normal_train = np.load("Dataset/Avenue/video_train_abnormal_"+str(cv_no)+".npy", allow_pickle = True), np.load("Dataset/Avenue/video_train_normal_"+str(cv_no)+".npy", allow_pickle = True)
    video_names_abnormal_test, video_names_normal_test = np.load("Dataset/Avenue/video_test_abnormal_"+str(cv_no)+".npy", allow_pickle = True), np.load("Dataset/Avenue/video_test_normal_"+str(cv_no)+".npy", allow_pickle = True)

    
    no_videos = len(X_train_abnormal)+len(X_train_normal)
    no_segments = X_train_abnormal.shape[1]
    max_iterations = 10000
    input_dim = X_train_abnormal.shape[2]
   
    mixing_parameters = mixing_params.MixingParameters(no_videos, no_segments, video_names_abnormal_train, video_names_normal_train)
    
    dnn = DNN(input_dim = input_dim)
    pretrained = torch.load("pretrained_avenue/model_top1_"+str(cv_no)+".pth.tar", map_location = lambda storage, loc: storage)
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

    best_auc = 0
    aucs = []
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
        train_video_names = train_videos
        train_feat = torch.from_numpy(train_feat)
        if torch.cuda.is_available():
            train_feat = Variable(train_feat, requires_grad = True).cuda()
        optimizer.zero_grad()
        [loss, exp_loss, kl_u, kl_z] = mll(model(train_feat), r_dist, pi_dist, seg_bag_idx, bag_labels_gt, mixing_parameters, train_videos)
        loss = -loss
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu())
        
        if i%10==0:
            print("Performing Evaluation on the iteration", i)
            model.eval()
            likelihood.eval()
            [aucs, best_auc] = evaluate(model,  likelihood.mixing_weights.t(), best_auc, aucs, cv_no, X_test_abnormal, X_test_normal, video_names_abnormal_test, video_names_normal_test, eps, rep_no, no_segments = no_segments)
            np.save("logs/Avenue/aucs_"+str(cv_no)+"_"+str(rep_no)+"_"+str(eps)+".npy", aucs)
            np.save("logs/Avenue/losses_"+str(cv_no)+"_"+str(rep_no)+"_"+str(eps)+".npy", losses)
            
        
            
            
            
            
        
        
        
        
        
        
    
    
    
    
    
    
