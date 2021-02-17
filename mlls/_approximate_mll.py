
import torch
from .marginal_log_likelihood import MarginalLogLikelihood
from abc import ABC, abstractmethod
from distributions import base_distributions
import numpy as np

class _ApproximateMarginalLogLikelihood(MarginalLogLikelihood, ABC):
    
    r"""
    An approximate marginal log likelihood (typically a bound) for approximate GP models.
    We expect that :attr:`model` is a :obj:`gpytorch.models.ApproximateGP`.
    Args:
        :attr:`likelihood` (:obj:`gpytorch.likelihoods.Likelihood`):
            The likelihood for the model
        :attr:`model` (:obj:`gpytorch.models.ApproximateGP`):
            The approximate GP model
        :attr:`num_data` (int):
            The total number of training data points (necessary for SGD)
        :attr:`beta` (float - default 1.):
            A multiplicative factor for the KL divergence term.
            Setting it to 1 (default) recovers true variational inference
            (as derived in `Scalable Variational Gaussian Process Classification`_).
            Setting it to anything less than 1 reduces the regularization effect of the model
            (similarly to what was proposed in `the beta-VAE paper`_).
        :attr:`combine_terms` (bool):
            Whether or not to sum the expected NLL with the KL terms (default True)
    """
    
    def __init__(self, likelihood, model, num_data,  beta = 1.0, combine_terms = True):
        super().__init__(likelihood, model)
        self.combine_terms = combine_terms
        self.num_data = num_data
        self.beta = beta
        
        
    @abstractmethod
    def _log_likelihood_term(self, variational_dist_f, r_dist, pi_dist, seg_bag_idx, bag_labels_gt, train_videos,  **kwargs):
        raise NotImplementedError
        
    def forward(self, approximate_dist_f, r_dist, pi_dist, seg_bag_idx, bag_labels_gt, mixing_params, train_videos, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.
        Args:
            :attr:`approximate_dist_f` (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
            :attr:`**kwargs`:
                Additional arguments passed to the likelihood's `expected_log_prob` function.
        """
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        [log_likelihood, exp_log_prob] = self._log_likelihood_term(approximate_dist_f, r_dist, pi_dist, seg_bag_idx, bag_labels_gt, train_videos, **kwargs)

        mixing_params.update_r(exp_log_prob, train_videos)
        exp_log_likelihood = log_likelihood.sum(-1).div(len(train_videos))
        mixing_params.update_pi(train_videos)
        kl_divergence_u = self.model.variational_strategy.kl_divergence().div(self.num_data / self.beta)
        kl_divergence_z = torch.zeros(r_dist.shape[0])
        for i in range(r_dist.shape[0]):
            Q = torch.distributions.Categorical(probs = r_dist[i])
            P = torch.distributions.Categorical(probs = pi_dist[i])
            div = torch.distributions.kl_divergence(Q, P)
            if abs(div)<9e-7:
                div = 0
            kl_divergence_z[i] = div

        kl_divergence_z = torch.mean(kl_divergence_z)
        # Add any additional registered loss terms
        added_loss = torch.zeros_like(exp_log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True
        
        # Log prior term
        log_prior = torch.zeros_like(exp_log_likelihood)
        
        for _, prior, closure, _ in self.named_priors():
          
            log_prior.add_(prior.log_prob(closure()).sum().div(self.num_data))
        
       
        if self.combine_terms:
           
            
            return exp_log_likelihood - kl_divergence_u - kl_divergence_z + log_prior - added_loss, exp_log_likelihood, kl_divergence_u, kl_divergence_z
        else:
            if had_added_losses:
                return exp_log_likelihood, kl_divergence_u, kl_divergence_z, log_prior.div(self.num_data), added_loss
            else:
                return exp_log_likelihood, kl_divergence_u, kl_divergence_z, log_prior.div(self.num_data)
        
        
    
    
    
