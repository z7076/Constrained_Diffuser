import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)
import math
from qpth.qp import QPFunction, QPSolvers

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1,):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns = torch.tensor(returns, requires_grad=True)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.grad_p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        if self.predict_epsilon:
            # Cause we condition on obs at t=0
            noise[:, 0, self.action_dim:] = 0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        if self.model.calc_energy:
            assert self.predict_epsilon
            x_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns.requires_grad = True
            noise.requires_grad = True

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, hidden_dim=256,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1, ar_inv=False, train_only_inv=False, constrained_mode=0):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        if self.ar_inv:
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)

        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )

        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)

        self.penalty = 1e-2
        self.use_equality = False
        self.is_cons = True
        
        algorithm = ['primal_dual', 'augmented_lagrangian', 'projected_gradient']
        self.constrained_mode = constrained_mode
        if self.constrained_mode == 0:
            self.algorithm = algorithm[0]
        elif self.constrained_mode == 1:
            self.algorithm = algorithm[1]
        elif self.constrained_mode == 2:
            self.algorithm = algorithm[2]
        elif self.constrained_mode == 3:
            self.algorithm = algorithm[0]
            self.is_cons = False


        def g_x1(state, t=None):
            # state = state.requires_grad_(True)
            proj = True
            if proj:
                if t is not None:
                    g_x = torch.clamp((0.95 - state[:,t,8]), max=0)
                else:
                    g_x = torch.clamp((0.95 - state[:,:,8]), max=0)
                return g_x
            else:
                if t is not None:
                    g_x = 0.95 - state[:,t,8]
                else:
                    g_x = 0.95 - state[:,:,8]
            return g_x
        
        
        g_x_funcs = [g_x1]
        self.g_x_funcs = g_x_funcs
        # dual variables for constraints
        
        self.alpha = 0.85

        num_batch = 10

        num_constraints = len(self.g_x_funcs)
        print('num_constraints', num_constraints)
        self.safe = torch.zeros(num_constraints, num_batch)
        self.dual_vars = torch.zeros((num_constraints, num_batch, self.horizon), dtype=torch.float32, device='cuda:0')*(5 / self.n_timesteps)
        if self.algorithm == 'augmented_lagrangian':
            self.slack_variables = torch.zeros((num_constraints, num_batch, self.horizon), dtype=torch.float32, device='cuda:0')

        self.safe = 0
        # for name, param in self.named_parameters():
        #     print(name, param)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if self.is_cons:
            # ['primal_dual', 'projected_gradient', 'augmented_lagrangian']
            cbf = True
            if self.algorithm == 'primal_dual':
                grad, vio = self.calc_grad(x_t)

                if cbf:

                    shift_dual = self.dual_vars[:,:,:-1]
                    padding = torch.zeros((self.dual_vars.shape[0],self.dual_vars.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_dual = torch.cat([padding, shift_dual], dim=2)

                    posterior_mean = posterior_mean + torch.sum(padding_dual.unsqueeze(-1) * grad, dim=0) - (1-self.alpha) * torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0)
                    
                else:
                    posterior_mean = posterior_mean + torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) * (-1 if self.use_equality else 1)
                self.dual_update(x_t, cbf)
            elif self.algorithm == 'augmented_lagrangian':
                grad, vio = self.calc_grad(x_t)
                if cbf:
                    shift_dual = self.dual_vars[:,:,:-1]
                    padding = torch.zeros((self.dual_vars.shape[0],self.dual_vars.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_dual = torch.cat([padding, shift_dual], dim=2)
                    shift_vio = vio[:,:,:-1]
                    padding_vio = torch.zeros((vio.shape[0],vio.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_vio_back = torch.cat([padding_vio, shift_vio], dim=2)
                    shift_vio = vio[:,:,1:]
                    padding_vio = torch.zeros((vio.shape[0],vio.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_vio_forw = torch.cat([shift_vio, padding_vio], dim=2)
                    shift_slack = self.slack_variables[:,:,:-1]
                    padding_slack = torch.zeros((self.slack_variables.shape[0],self.slack_variables.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_slack = torch.cat([padding_slack, shift_slack], dim=2)

                    posterior_mean = posterior_mean + (- torch.sum(padding_dual.unsqueeze(-1) * grad, dim=0) + (1-self.alpha) * torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) - self.penalty * torch.sum((vio - (1 - self.alpha)*padding_vio_back - padding_slack).unsqueeze(-1) * grad , dim=0) + self.penalty * (1 - self.alpha) * torch.sum((padding_vio_forw - (1 - self.alpha)*vio - self.slack_variables).unsqueeze(-1) * grad , dim=0))
                else:
                    posterior_mean = posterior_mean - torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) - self.penalty * torch.sum((vio - self.slack_variables).unsqueeze(-1) * grad , dim=0)
                self.dual_update_aug(x_t, cbf)

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance_langevin(self, x, cond, t, returns=None):
        if self.returns_condition:

            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x.shape)
        # print('sqrt_alpha', sqrt_alpha)
        alphas_cumprod = extract(self.alphas_cumprod, t, x.shape)
        score = (sqrt_alpha * x_recon - x) / (1 - alphas_cumprod)
        # print('score', score)

        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x.shape)
        beta = extract(self.betas, t, x.shape)
        model_mean = x + posterior_log_variance_clipped.exp() / 2 * score
        x_t = x
        posterior_mean = model_mean
        if self.is_cons:
            # ['primal_dual', 'projected_gradient', 'augmented_lagrangian']
            cbf = False
            if self.algorithm == 'primal_dual':
                grad, vio = self.calc_grad(x_t)

                if cbf:
                    shift_dual = self.dual_vars[:,:,:-1]
                    padding = torch.zeros((self.dual_vars.shape[0],self.dual_vars.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_dual = torch.cat([padding, shift_dual], dim=2)

                    posterior_mean = posterior_mean + torch.sum(padding_dual.unsqueeze(-1) * grad, dim=0) - (1-self.alpha) * torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0)
                    
                else:
                    posterior_mean = posterior_mean + torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) * (-1 if self.use_equality else 1)
                self.dual_update(x_t, cbf)
            elif self.algorithm == 'augmented_lagrangian':
                grad, vio = self.calc_grad(x_t)
                if cbf:
                    shift_dual = self.dual_vars[:,:,:-1]
                    padding = torch.zeros((self.dual_vars.shape[0],self.dual_vars.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_dual = torch.cat([padding, shift_dual], dim=2)
                    shift_vio = vio[:,:,:-1]
                    padding_vio = torch.zeros((vio.shape[0],vio.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_vio_back = torch.cat([padding_vio, shift_vio], dim=2)
                    shift_vio = vio[:,:,1:]
                    padding_vio = torch.zeros((vio.shape[0],vio.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_vio_forw = torch.cat([shift_vio, padding_vio], dim=2)
                    shift_slack = self.slack_variables[:,:,:-1]
                    padding_slack = torch.zeros((self.slack_variables.shape[0],self.slack_variables.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_slack = torch.cat([padding_slack, shift_slack], dim=2)

                    posterior_mean = posterior_mean + (- torch.sum(padding_dual.unsqueeze(-1) * grad, dim=0) + (1-self.alpha) * torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) - self.penalty * torch.sum((vio - (1 - self.alpha)*padding_vio_back - padding_slack).unsqueeze(-1) * grad , dim=0) + self.penalty * (1 - self.alpha) * torch.sum((padding_vio_forw - (1 - self.alpha)*vio - self.slack_variables).unsqueeze(-1) * grad , dim=0))
                else:
                    posterior_mean = posterior_mean - torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) - self.penalty * torch.sum((vio - self.slack_variables).unsqueeze(-1) * grad , dim=0)
                self.dual_update_aug(x_t, cbf)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:

            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def calc_grad(self, x):

            with torch.enable_grad():
                state = x.clone().detach().requires_grad_(True)
                
                # Accumulate g(x) for all functions
                grads = []
                vios = []
                for g_x_func in self.g_x_funcs:
                    # Apply the user-defined g(x) function
                    g_x = g_x_func(state)
                    # print(g_x)    

                    grad = torch.autograd.grad(g_x.sum(), state)[0]

                    vio = g_x

                    grads.append(grad)
                    vios.append(vio)
                self.safe = torch.vstack(vios)
                return torch.stack(grads, dim=0), torch.stack(vios, dim=0)
        
    def dual_update(self, x, cbf, learning_rate=1e-1):

        if cbf == False:
            for i, g_x_func in enumerate(self.g_x_funcs):
                # Apply the user-defined g(x) function
                g_x = g_x_func(x)
                
                # Update dual variables
                if self.use_equality:
                    
                    self.dual_vars[i] = self.dual_vars[i] + self.penalty * g_x.squeeze(-1)
                else:
                    self.dual_vars[i] = torch.clamp(
                        self.dual_vars[i] - learning_rate * g_x.squeeze(-1), 
                        min=0
                    )
        else:
            for i, g_x_func in enumerate(self.g_x_funcs):
                for t in range(self.horizon-1):
                    # print()
                    self.dual_vars[i,:, t] = torch.clamp(
                            self.dual_vars[i,:, t] - learning_rate * (g_x_func(x,t+1) - (1 - self.alpha)*g_x_func(x, t)).squeeze(-1), 
                            min=0
                        )

    def dual_update_aug(self, x, cbf):

        if cbf == False:
            for i, g_x_func in enumerate(self.g_x_funcs):
                # Apply the user-defined g(x) function
                g_x = g_x_func(x)
                
                # Update dual variables
                if self.use_equality:
                    self.dual_vars[i] = self.dual_vars[i] + self.penalty * (g_x.squeeze(-1) - self.slack_variables[i])
                else:

                    self.slack_variables[i] = torch.clamp(self.dual_vars[i] / self.penalty + g_x.squeeze(-1), min=0)
                    self.dual_vars[i] = self.dual_vars[i] + self.penalty * (g_x.squeeze(-1) - self.slack_variables[i])
                    self.penalty *= 1.0002

        else:
            for i, g_x_func in enumerate(self.g_x_funcs):
            # Apply the user-defined g(x) function
                for t in range(self.horizon-1):

                    g_x = g_x_func(x)
                    
                    # Update dual variables
                    if self.use_equality:
                        self.dual_vars[i,:,t] = self.dual_vars[i,:,t] + self.penalty * (g_x.squeeze(-1) - self.slack_variables[i])
                    else:
                        self.slack_variables[i,:,t] = torch.clamp(self.dual_vars[i,:,t] / self.penalty + (g_x_func(x,t+1) - (1 - self.alpha)*g_x_func(x, t)).squeeze(-1), min=0)
                        self.dual_vars[i,:,t] = self.dual_vars[i,:,t] + self.penalty * ((g_x_func(x,t+1) - (1 - self.alpha)*g_x_func(x, t)).squeeze(-1) - self.slack_variables[i,:,t])
                        self.penalty *= 1.0001


    def _project_to_feasible_region(self, x):
        """
        Simple projection method to handle constraints
        
        Args:
        - x (torch.Tensor): Point to project
        - constraint_func (callable): Constraint function
        
        Returns:
        - torch.Tensor: Projected point
        """
        # Basic implementation - can be made more sophisticated
        # x[:, :, 8] = torch.clamp(x[:, :, 8], min=-0.9, max=0.9)

        cbf = True
        if cbf: # Check if cbf is enabled and there are constraint functions
            use_qp = False
            if use_qp:
                batch_size = x.shape[0]
                device = x.device
                dtype = x.dtype

                # Number of timesteps where x_t+1 is constrained by x_t
                # These are t+1 = 1, ..., self.horizon-1
                num_constrained_transitions = self.horizon - 1

                if num_constrained_transitions > 0:
                    # Values of x at t (from 0 to H-2) that influence the constraint on x_t+1
                    x[:, 0, 8] = torch.clamp(x[:, 0, 8], max=0.95)
                    x_current_dim8 = x[:, :-1, 8].clone()  # Shape: [batch_size, H-1]

                    # Nominal values of x at t+1 (from 1 to H-1) that we want to be close to
                    # These are the decision variables for the QPs
                    x_next_nominal_dim8 = x[:, 1:, 8].clone() # Shape: [batch_size, H-1]

                    num_qps = batch_size * num_constrained_transitions

                    # Reshape for batch QP: each element of x_next_nominal_dim8 is an independent QP
                    # These are the x_orig values for the objective function
                    x_orig_flat = x_next_nominal_dim8.reshape(num_qps, 1)

                    # Objective: min 0.5 * y^T Q y + p^T y, where y is the decision variable (x'_{t+1,8})
                    # To minimize ||y - x_orig_flat||^2, Q=2*I, p=-2*x_orig_flat
                    Q_qp = torch.ones(num_qps, 1, 1, dtype=dtype, device=device) * 2.0
                    p_qp = -2.0 * x_orig_flat

                    # Inequality constraint: G*y <= h_val
                    # y_decision <= 0.95 - (1 - self.alpha) * (0.95 - x_current_dim8)
                    # So, G = [[1]]
                    G_qp = torch.ones(num_qps, 1, 1, dtype=dtype, device=device)

                    # Calculate h_val for the constraint
                    h_val_for_xt = 0.95 - x_current_dim8 # This is h(x_t)
                    # rhs_of_cbf_ineq = (1 - self.alpha) * h_val_for_xt # This is (1-alpha)h(x_t)
                    # Constraint: h(x_t+1) >= rhs_of_cbf_ineq
                    # 0.95 - x'_{t+1,8} >= rhs_of_cbf_ineq
                    # -x'_{t+1,8} >= rhs_of_cbf_ineq - 0.95
                    # x'_{t+1,8} <= 0.95 - rhs_of_cbf_ineq
                    # h_constraint_values = 0.95 - rhs_of_cbf_ineq
                    h_constraint_values = 0.95 - (1 - self.alpha) * (0.95 - x_current_dim8)
                    h_qp = h_constraint_values.reshape(num_qps, 1)

                    # No equality constraints
                    A_qp = torch.empty(num_qps, 0, 1, dtype=dtype, device=device)
                    b_qp = torch.empty(num_qps, 0, dtype=dtype, device=device)
                    
                    # Solve the batch of QPs
                    qp_solver = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED) # Suppress solver output
                    solution_flat = qp_solver(Q_qp, p_qp, G_qp, h_qp, A_qp, b_qp)
                    
                    # Reshape solution back and update the original tensor for x[:, 1:, 8]
                    x[:, 1:, 8] = solution_flat.reshape(batch_size, num_constrained_transitions)
            else:
                    limit = 0.95
                    for i in range(len(self.g_x_funcs)):
                        for t in range(self.horizon-1):
                            if t == 0:
                                x[:, t, 8] = torch.clamp(x[:, t, 8], max=limit)
                            # Apply the user-defined g(x) function
                            else:
                                g_x = self.g_x_funcs[i](x)
                                # Update x based on the constraint violation
                                x[:, t+1, 8] = torch.clamp(x[:, t+1, 8], max=-(1-self.alpha)*g_x[:, t] + limit)
        else:
            # x[:, :, 8] = torch.clamp(x[:, :, 8], max=0.95)
            # Ensure qpth is imported. Typically done at the top of the file.
            
            
            x_target_slice = x[:, :, 8].clone() # The original values for the 8th feature
            batch_size_qp, horizon_qp = x_target_slice.shape
            num_qps = batch_size_qp * horizon_qp

            if num_qps > 0:
                # Reshape for batch QP: each element of x_target_slice is an independent QP
                x_orig_flat = x_target_slice.reshape(num_qps, 1)

                # Q: Quadratic term coefficient matrix (batched)
                # For min (y-x_orig)^2, QP form 0.5 y^T Q y + p^T y needs Q=2.
                # Q shape: (num_qps, num_vars, num_vars) -> (num_qps, 1, 1)
                Q_qp = torch.ones(num_qps, 1, 1, dtype=x.dtype, device=x.device) * 2.0

                # p: Linear term coefficient vector (batched)
                # p = -2*x_orig
                # p shape: (num_qps, num_vars) -> (num_qps, 1)
                p_qp = -2.0 * x_orig_flat

                # Gx <= h: Inequality constraints (y <= 0.95  =>  1*y <= 0.95)
                # G shape: (num_qps, num_constraints, num_vars) -> (num_qps, 1, 1)
                G_qp = torch.ones(num_qps, 1, 1, dtype=x.dtype, device=x.device)
                # h shape: (num_qps, num_constraints) -> (num_qps, 1)
                h_qp = torch.ones(num_qps, 1, dtype=x.dtype, device=x.device) * 0.95

                # Ax = b: Equality constraints (none in this case)
                # A shape: (num_qps, num_eq_constraints, num_vars) -> (num_qps, 0, 1)
                A_qp = torch.empty(num_qps, 0, 1, dtype=x.dtype, device=x.device)
                # b shape: (num_qps, num_eq_constraints) -> (num_qps, 0)
                b_qp = torch.empty(num_qps, 0, dtype=x.dtype, device=x.device)
                

                qp_solver = QPFunction(verbose=-1) # Suppress solver output
                solution_flat = qp_solver(Q_qp, p_qp, G_qp, h_qp, A_qp, b_qp)
                
                # Reshape solution back and update the original tensor
                x[:, :, 8] = solution_flat.reshape(batch_size_qp, horizon_qp)

        return x

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        use_ddpm = True
        if use_ddpm:
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        else:
            model_mean, _, model_log_variance = self.p_mean_variance_langevin(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        xp = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        if self.is_cons and self.algorithm == 'projected_gradient':
            xp = self._project_to_feasible_region(xp)

        for g_x in self.g_x_funcs:

            self.safe = torch.relu(-g_x(xp)).sum(dim=-1)
        return xp

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=False, return_diffusion=False):
        device = self.betas.device
        self.penalty = 1e-2

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, 0)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(-200, self.n_timesteps)):
            if i <= 0:
                i_ = 0
            elif i > self.n_timesteps:
                i_ = self.n_timesteps-1
            else:
                i_ = i
            timesteps = torch.full((batch_size,), i_, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, 0)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        # if dataset:
        #     self.dataset = dataset
        #     self.flush_gx()
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, 0)

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns=None):

        if self.train_only_inv:
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                loss = self.inv_model.calc_loss(x_comb_t, a_t)
                info = {'a0_loss':loss}
            else:
                pred_a_t = self.inv_model(x_comb_t)
                loss = F.mse_loss(pred_a_t, a_t)
                info = {'a0_loss': loss}
        else:
            batch_size = len(x)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns)
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
            else:
                pred_a_t = self.inv_model(x_comb_t)
                inv_loss = F.mse_loss(pred_a_t, a_t)

            loss = (1 / 2) * (diffuse_loss + inv_loss)

        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

class ARInvModel(nn.Module):
    def __init__(self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)])
        self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.action_embed_hid), nn.ReLU(),
                                                    nn.Linear(self.action_embed_hid, self.num_bins))])

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid), nn.ReLU(),
                              nn.Linear(self.action_embed_hid, self.num_bins)))

    def forward(self, comb_state, deterministic=False):
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
                                              self.low_act + (l_0 + 1) * self.bin_size).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1))
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
                                                  self.low_act + (l_i + 1) * self.bin_size).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                                     l_action[:, i])

        return loss/self.action_dim

class ActionGaussianDiffusion(nn.Module):
    # Assumes horizon=1
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1,):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns = torch.tensor(returns, requires_grad=True)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device
        self.penalty = 5e-3

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, action_start, state, t, returns=None):
        noise = torch.randn_like(action_start)
        action_noisy = self.q_sample(x_start=action_start, t=t, noise=noise)

        if self.model.calc_energy:
            assert self.predict_epsilon
            action_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns.requires_grad = True
            noise.requires_grad = True

        pred = self.model(action_noisy, state, t, returns)

        assert noise.shape == pred.shape

        if self.predict_epsilon:
            loss = F.mse_loss(pred, noise)
        else:
            loss = F.mse_loss(pred, action_start)

        return loss, {'a0_loss':loss}

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        assert x.shape[1] == 1 # Assumes horizon=1
        x = x[:,0,:]
        cond = x[:,self.action_dim:] # Observation
        x = x[:,:self.action_dim] # Action
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

class TransformerStateTransitionModel(nn.Module):
    def __init__(
        self, 
        observation_dim, 
        action_dim, 
        hidden_dim=256, 
        num_heads=4, 
        num_layers=3, 
        dropout=0.1,
        context_length=1,  # How many past state-action pairs to consider
        ensemble_size=1,
        probabilistic=False
    ):
        super(TransformerStateTransitionModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.context_length = context_length
        self.hidden_dim = hidden_dim
        self.ensemble_size = ensemble_size
        self.probabilistic = probabilistic
        
        # Input projections
        self.state_projection = nn.Linear(observation_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        
        # Positional encoding for temporal context
        self.register_buffer("position_encoding", self._create_positional_encoding(context_length, hidden_dim))
        
        # Transformer encoder layers
        encoder_layers = []
        for _ in range(ensemble_size):
            layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, 
                    nhead=num_heads,
                    dim_feedforward=hidden_dim*4,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=num_layers
            )
            encoder_layers.append(layer)
        self.transformer_encoders = nn.ModuleList(encoder_layers)
        
        # Output projection (probabilistic or deterministic)
        if probabilistic:
            self.mean_heads = nn.ModuleList([nn.Linear(hidden_dim, observation_dim) for _ in range(ensemble_size)])
            self.log_std_heads = nn.ModuleList([nn.Linear(hidden_dim, observation_dim) for _ in range(ensemble_size)])
        else:
            self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, observation_dim) for _ in range(ensemble_size)])
        
    def _create_positional_encoding(self, length, dim):
        pos_enc = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc
    
    def forward(self, state, action, return_distribution=False, model_idx=None):
        """
        Predict next state given current state and action
        
        Args:
            state (tensor): Current state [batch_size, observation_dim] 
                           or [batch_size, context_length, observation_dim] if using history
            action (tensor): Action [batch_size, action_dim]
                           or [batch_size, context_length, action_dim] if using history
            return_distribution (bool): Whether to return distribution parameters
            model_idx (int): Index of the ensemble model to use, None means random or all
            
        Returns:
            tensor: Predicted next state or distribution parameters
        """
        batch_size = state.shape[0]
        
        # Handle single state-action pair vs. sequence
        if state.dim() == 2:
            # Expand to sequence with context_length=1
            state = state.unsqueeze(1)
            action = action.unsqueeze(1)
        
        # Ensure we have the right context length
        if state.shape[1] < self.context_length:
            # Pad with zeros if needed
            padding = torch.zeros(batch_size, self.context_length - state.shape[1], 
                                 self.observation_dim, device=state.device)
            state = torch.cat([padding, state], dim=1)
            
            action_padding = torch.zeros(batch_size, self.context_length - action.shape[1], 
                                       self.action_dim, device=action.device)
            action = torch.cat([action_padding, action], dim=1)
        
        # Project inputs to embedding space
        state_emb = self.state_projection(state)
        action_emb = self.action_projection(action)
        
        # Combine state and action embeddings
        combined = state_emb + action_emb
        
        # Add positional encoding
        combined = combined + self.position_encoding.unsqueeze(0)
        
        # Apply attention mask if needed (to focus on most recent states)
        # mask = self._generate_mask(combined.shape[1]).to(combined.device)
        
        # Select model from ensemble
        if model_idx is None:
            if self.training:
                model_idx = torch.randint(0, self.ensemble_size, (1,)).item()
            else:
                # In evaluation, we'll use all models and average/ensemble results
                outputs = []
                for idx in range(self.ensemble_size):
                    transformer_output = self.transformer_encoders[idx](combined)
                    # Take the last position's output (most recent)
                    final_output = transformer_output[:, -1]
                    
                    if self.probabilistic:
                        mean = self.mean_heads[idx](final_output)
                        log_std = self.log_std_heads[idx](final_output)
                        log_std = torch.clamp(log_std, -10, 2)
                        
                        if return_distribution:
                            outputs.append((mean, log_std))
                        else:
                            # Sample
                            std = torch.exp(log_std)
                            noise = torch.randn_like(std)
                            next_state = mean + noise * std
                            outputs.append(next_state)
                    else:
                        outputs.append(self.output_heads[idx](final_output))
                
                # Average all outputs
                if self.probabilistic and return_distribution:
                    means, log_stds = zip(*outputs)
                    mean = torch.stack(means).mean(dim=0)
                    # Use max uncertainty across ensemble
                    log_std = torch.stack(log_stds).max(dim=0)[0]
                    return mean, log_std
                else:
                    return torch.stack(outputs).mean(dim=0)
        
        # Process through transformer
        transformer_output = self.transformer_encoders[model_idx](combined)
        # Take the last position's output
        final_output = transformer_output[:, -1]
        
        # Output head
        if self.probabilistic:
            mean = self.mean_heads[model_idx](final_output)
            log_std = self.log_std_heads[model_idx](final_output)
            log_std = torch.clamp(log_std, -10, 2)
            
            if return_distribution:
                return mean, log_std
            else:
                # Sample from distribution
                std = torch.exp(log_std)
                noise = torch.randn_like(std)
                next_state = mean + noise * std
                return next_state
        else:
            return self.output_heads[model_idx](final_output)
    
    def calc_loss(self, state, action, next_state, reduction='mean'):
        """
        Calculate loss for the model
        
        Args:
            state (tensor): Current state [batch_size, observation_dim]
            action (tensor): Action [batch_size, action_dim]
            next_state (tensor): True next state [batch_size, observation_dim]
            reduction (str): 'mean' or 'none'
            
        Returns:
            tensor: Loss value
        """
        if not self.probabilistic:
            # MSE loss for deterministic prediction
            pred_next_state = self(state, action)
            loss = F.mse_loss(pred_next_state, next_state, reduction=reduction)
            return loss
        else:
            # Negative log likelihood for probabilistic prediction
            mean, log_std = self(state, action, return_distribution=True)
            std = torch.exp(log_std)
            
            # Gaussian negative log likelihood
            nll = 0.5 * (
                ((next_state - mean) / std) ** 2 + 
                2 * log_std + 
                math.log(2 * math.pi)
            )
            
            if reduction == 'none':
                return nll
            else:
                return nll.mean()
                
    def predict_n_steps(self, initial_state, actions, use_history=False):
        """
        Predict trajectory by rolling out n steps
        
        Args:
            initial_state (tensor): Initial state [batch_size, observation_dim]
            actions (tensor): Sequence of actions [batch_size, n_steps, action_dim]
            use_history (bool): Whether to use history for predictions
            
        Returns:
            tensor: Predicted states [batch_size, n_steps+1, observation_dim]
        """
        batch_size, n_steps, _ = actions.shape
        states = torch.zeros(batch_size, n_steps + 1, self.observation_dim, device=actions.device)
        states[:, 0] = initial_state
        
        # For history tracking
        if use_history and self.context_length > 1:
            state_history = torch.zeros(batch_size, self.context_length, self.observation_dim, device=actions.device)
            action_history = torch.zeros(batch_size, self.context_length, self.action_dim, device=actions.device)
            
            # Initialize with the first state
            state_history[:, -1] = initial_state
        
        for t in range(n_steps):
            current_action = actions[:, t]
            
            if use_history and self.context_length > 1:
                # Update action history
                action_history = torch.cat([
                    action_history[:, 1:], 
                    current_action.unsqueeze(1)
                ], dim=1)
                
                # Predict using history
                next_state = self(state_history, action_history)
                
                # Update state history
                state_history = torch.cat([
                    state_history[:, 1:],
                    next_state.unsqueeze(1)
                ], dim=1)
            else:
                # Simple prediction with just current state
                current_state = states[:, t]
                next_state = self(current_state, current_action)
            
            states[:, t + 1] = next_state
            
        return states