
import warnings
import torch
from distributions import Distribution, MultitaskMultivariateNormal, base_distributions
from .likelihood import Likelihood
from torch.autograd import Variable
log_sigmoid = torch.nn.LogSigmoid()
class MILLikelihood(Likelihood):
    """
    Implements the MIL likelihood for the MIL training
    
    """
    
    def __init__(self, num_features = None, num_classes = None, mixing_weights = True, mixing_weights_prior = None, **kwargs):
        super().__init__()
        
        if num_classes is None:
            raise ValueError("num_classes is required")
        if num_classes!=1:
            raise ValueError("Only one class per instance is required")
            
        self.num_classes = num_classes
        if mixing_weights:
            self.num_features = num_features
            if num_features is None:
                raise ValueError("num_features is required with mixing weights")
            self.register_parameter(name = "mixing_weights", parameter = torch.nn.Parameter(torch.randn(num_classes, num_features).div_(num_features)))
            if mixing_weights_prior is not None:
                self.register_prior("mixing_weights_prior", "mixing_weights")
        else:
            self.num_features = num_classes
            self.mixing_weights = None
    
    def get_bag_instances(self, mixed_fs, seg_bag_idx, bags, no_likelihood_points, no_segs_per_bag, num_features):
        
        
        bag_functional_values = torch.zeros_like(mixed_fs)
        
        bag_functional_values = bag_functional_values.reshape(no_likelihood_points, len(bags), no_segs_per_bag)
        for i, bag in enumerate(bags):
            idxs = (seg_bag_idx==bag)
            bag_functional_values[:, i, :] = mixed_fs[:, idxs]
       
        return bag_functional_values
        
        
    
    def forward(self, function_samples, z_samples, seg_bag_idx, bag_labels_gt, video_names, *params, **kwargs):
        num_data,  num_features = function_samples.shape[-2:]
        
         # Catch legacy mode
        if num_data == self.num_features:
            warning.warn("The input to MIL likelihood should be MultitaskMultivariatwNormal (num_data x num_tasks.\
                                                                                                Batch MultivariateNormal inputs (num_tasks x num_data) will be depricated.", DeprecationWarning,)
            function_samples = function_samples.transpose(-1, -2)
            
            num_data, num_features = function_samples.shape[-2:]
        if num_features != self.num_features:
            raise RuntimeError("There should be %d features"% self.num_features)
        if self.mixing_weights is not None:
            mixed_fs = function_samples @ self.mixing_weights.t()
        else:
            mixed_fs = function_samples
        
        
        
        mixed_fs = mixed_fs.reshape(-1, num_data)
        num_bags, num_segs_per_bag = z_samples.shape[-2:]
        bags = video_names
        num_likelihood_points = function_samples.shape[0]
        bag_function_samples = self.get_bag_instances(mixed_fs, seg_bag_idx, bags, num_likelihood_points, num_segs_per_bag, num_features)
        intermediate = torch.log(1/(1+torch.exp(-bag_labels_gt*bag_function_samples)))
        res = torch.sum(z_samples*intermediate, 2)
        
       

        return [res, intermediate]
    
    def __call__(self, function, *params, **kwargs):
        if isinstance(function, Distribution) and not isinstance(function, MultitaskMultivariateNormal):
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                DeprecationWarning,
            )
            function = MultitaskMultivariateNormal.from_batch_mvn(function)
        return super().__call__(function, *params, **kwargs)
  

        
        
        
