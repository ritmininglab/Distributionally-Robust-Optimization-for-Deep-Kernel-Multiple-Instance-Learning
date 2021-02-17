
import cvxpy as cp
import torch
import numpy as np


def solver(r_b, eps = 0.1):
    pi_bn = cp.Variable(len(r_b))
    uniform = np.ones_like(r_b)/len(r_b)
    constraints = []
    constraints.append(pi_bn>=0)
    constraints.append(cp.sum(pi_bn) == 1)
    constraints.append(cp.kl_div(pi_bn, uniform)<=eps)
    obj = cp.Minimize(-r_b@cp.log(pi_bn))
    prob = cp.Problem(obj, constraints)
    prob.solve(cp.ECOS)
    return pi_bn.value.tolist()
    
    
    

class MixingParameters(object):
    def __init__(self, num_videos, num_segments, video_names_abnormal, video_names_normal, eps=0.1):
        self.num_videos = num_videos
        self.num_segments = num_segments
        self.r = torch.ones(num_videos, num_segments)/self.num_segments
        self.pi = torch.ones(num_videos, num_segments)/self.num_segments
        self.video_names_abnormal = video_names_abnormal
        self.video_names_normal = video_names_normal
        self.all_videos = np.concatenate([self.video_names_abnormal, self.video_names_normal])
        self.eps = eps
    def update_r(self, expected_log_likelihood, batch_video_names):
        log_prob = expected_log_likelihood.detach().cpu()
        for i, video in enumerate(batch_video_names):
            rho = self.pi[np.where(self.all_videos==video)[0][0]]*torch.exp(log_prob[i])
            self.r[np.where(self.all_videos==video)[0][0]] = rho/torch.sum(rho)
        
    def update_pi(self, batch_video_names):
        for i, video in enumerate(batch_video_names):
            selected_r = self.r[np.where(self.all_videos==video)[0][0]]
            r_b = selected_r.detach().cpu().numpy()
            try:
                pi_b = solver(r_b, eps = self.eps)
                pi_b = np.array(pi_b)
                if len(pi_b[pi_b<0])!=0:
                    continue
                self.pi[np.where(self.all_videos==video)[0][0]] = torch.from_numpy(pi_b)
            
            except AttributeError:
                
                print("Attribute Error")
                continue
                
            except Exception as inst:
                print("Exception", inst)
                
                continue
            

            
            
    def get_r(self, videos):
        
        r_dist = torch.zeros(len(videos), self.num_segments)
        for i, video in enumerate(videos):
            r_dist[i] = self.r[np.where(self.all_videos==video)[0][0]]
        return r_dist
    
    def get_pi(self, videos):
        
        pi_dist = torch.zeros(len(videos), self.num_segments)
        for i, video in enumerate(videos):
            pi_dist[i] = self.pi[np.where(self.all_videos==video)[0][0]]
        return pi_dist
    
    
    
        
        
        
        
        
        
        
