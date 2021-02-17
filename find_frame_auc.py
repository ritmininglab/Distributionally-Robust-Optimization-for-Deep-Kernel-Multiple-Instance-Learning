
import numpy as np
import os
from sklearn import metrics
import torch
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir_path = "SanghaiTech/Dataset/Split/"
fc7_features_path = os.path.join(dir_path, 'fc7-features')
annotation_path = os.path.join(dir_path, 'annotations')
root_dir = "SanghaiTech/Dataset/Videos"
annotated_videos = os.listdir(os.path.join(root_dir, 'testing', 'fc7-features'))
unannotated_videos = os.listdir(os.path.join(root_dir, 'training', 'preprocessed/'))

def getframeauc(model,  weights, X_test_abnormal, X_test_normal, video_names_abnormal, video_names_normal, type ="testing"):

    X_test_abnormal = X_test_abnormal.reshape(X_test_abnormal.shape[0]*X_test_abnormal.shape[1], X_test_abnormal.shape[2])
    X_test_normal = X_test_normal.reshape(X_test_normal.shape[0]*X_test_normal.shape[1], X_test_normal.shape[2])
    X_test_abnormal = torch.from_numpy(X_test_abnormal)
    X_test_normal = torch.from_numpy(X_test_normal)
    X_test_abnormal = Variable(X_test_abnormal).cuda()
    X_test_normal = Variable(X_test_normal).cuda()
    
    
    functional_dist_abnormal = model(X_test_abnormal)
    predictions_abnormal = functional_dist_abnormal.mean @ weights
    predictions_abnormal = 1/(1+torch.exp(-predictions_abnormal))
    
    functional_dist_normal = model(X_test_normal)
    predictions_normal = functional_dist_normal.mean @ weights
    predictions_normal = 1/(1+torch.exp(-predictions_normal))
    predictions_abnormal = predictions_abnormal.data.cpu().flatten()
    predictions_normal = predictions_normal.data.cpu().flatten()
    predictions_abnormal = predictions_abnormal.reshape(int(X_test_abnormal.shape[0]/32),32)
    predictions_normal = predictions_normal.reshape(int(X_test_normal.shape[0]/32),32)
    GT, Pred = [], []
    clip_size = 16
    video_names = np.concatenate([video_names_abnormal, video_names_normal])
    predictions = np.concatenate([predictions_abnormal, predictions_normal])
    for i, video in enumerate(video_names):
        
        prediction = predictions[i]
        if type=="training":
            no_clips = len(sorted(os.listdir(fc7_features_path+"/training/"+video)))
        else:
            no_clips = len(sorted(os.listdir(fc7_features_path+"/testing/"+video)))
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
        
        if video in annotated_videos:
            
            val = np.load(os.path.join(root_dir, 'testing', 'test_frame_mask', video+".npy"))
            number_frames = len(val)
            GT.extend(val.tolist())
        elif video in unannotated_videos:
           
            number_frames = len(os.listdir(os.path.join(root_dir, 'training', 'preprocessed', video)))
            val = np.zeros(number_frames)
            GT.extend(val.tolist())
        else:
            print("Unusual")
            print(video)
        frame_pred = np.zeros(number_frames)
        for j in range(no_clips):
            start_frame = j*clip_size
            if (j+1)*clip_size>number_frames:
                end_frame = number_frames
            else:
                end_frame = (j+1)*clip_size
            frame_pred[start_frame: end_frame] =clip_pred_score[j]
        Pred.extend(frame_pred.tolist())
    fpr, tpr, thresholds = metrics.roc_curve (GT, Pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return [roc_auc, GT, Pred]
