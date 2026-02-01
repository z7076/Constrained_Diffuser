import numpy as np
import torch
from torch import nn
import pdb
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def run_env(env_name, init_state, u_r):
    env_name.reset()
    env_name.set_state(init_state[:2], init_state[2:])
    states = np.expand_dims(init_state, axis=0)  # Initial state
    for k in range(u_r.shape[0]):
        next_state,_, _, _ = env_name.step(u_r[k, :])
        # print('next', next_state)
        next_state = np.expand_dims(next_state, axis=0)
        # print('next', next_state.shape)
        # print('state', states.shape)
        if k < u_r.shape[0] - 1:
            states = np.concatenate([states, next_state], axis=0)
        # print(states.shape)

    return states


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.norm_mins = [0.39643136, 0.44179875]
        self.norm_maxs = [7.2163844, 10.219488]
        

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.penalty = 2.5e-4
        self.use_equality = False
        
        algorithm = ['primal_dual', 'projected_gradient', 'augmented_lagrangian']
        
        self.algorithm = algorithm[4]
        if self.algorithm == 'augmented_lagrangian':
            self.use_equality = False


        def g_x1(state, t=None):
            # state = state.requires_grad_(True)
            xr1 = 2 * 1 / (self.norm_maxs[1] - self.norm_mins[1])
            yr1 = 2 * 1 / (self.norm_maxs[0] - self.norm_mins[0])
            off_x1 = 2 * (5.8 - 0.5 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
            off_y1 = 2 * (5 - 0.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
            if t is not None:
                g_x = torch.sqrt(torch.pow((state[:, t, 2] - off_y1) / yr1, 2) + torch.pow((state[:, t, 3] - off_x1) / xr1, 2)) - 1
            else:

                g_x = torch.sqrt(torch.pow((state[:, :, 2] - off_y1) / yr1, 2) + torch.pow((state[:, :, 3] - off_x1) / xr1, 2))- 1
                # g_x_t1 = torch.pow((state[:, :, 2:3] - off_y1) / yr1, 2) + torch.pow((state[:, :, 3:4] - off_x1) / xr1, 2) - 1
            return g_x


        g_x_funcs = [g_x1]
        self.g_x_funcs = g_x_funcs
        # dual variables for constraints
        self.is_cons = True
        self.alpha= 0.9
        self.num_batch = 1
        # num_constraints = 1

        
        # Initialize algorithm-specific parameters
        num_constraints = len(self.g_x_funcs)
        self.safe = torch.zeros(num_constraints)
        self.dual_vars = torch.zeros((num_constraints, self.num_batch, self.horizon), dtype=torch.float32, device='cuda:0')*(5 / self.n_timesteps)
        if self.algorithm == 'augmented_lagrangian':
            self.slack_variables = torch.zeros((num_constraints, self.num_batch, self.horizon), dtype=torch.float32, device='cuda:0')

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        # self.register_buffer('coeff_fx', 1 + betas/2)
        # self.register_buffer('coeff_gx2', betas)
        # self.register_buffer('coeff_gradient', torch.sqrt(1/((1 - alphas_cumprod) * alphas)))


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

        if self.is_cons:
            cbf = False
            # ['primal_dual', 'projected_gradient', 'augmented_lagrangian']
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

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance_langevin(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))
        # print('x_recon', x_recon)


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


    def calc_grad(self, x):

        with torch.enable_grad():
            state = x.clone().detach().requires_grad_(True)
            
            grads = []
            vios = []
            for g_x_func in self.g_x_funcs:
                g_x = g_x_func(state)

                grad = torch.autograd.grad(g_x.sum(), state)[0]
                vio = g_x

                grads.append(grad)
                vios.append(vio)
            self.safe = torch.vstack(vios)
            return torch.stack(grads, dim=0), torch.stack(vios, dim=0)
            
        
    def dual_update(self, x, cbf, learning_rate=1e-3):

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
                    self.penalty *= 1.004

        else:
            for i, g_x_func in enumerate(self.g_x_funcs):

                for t in range(self.horizon-1):

                    g_x = g_x_func(x)
                    
                    # Update dual variables
                    if self.use_equality:
                        self.dual_vars[i,:,t] = self.dual_vars[i,:,t] + self.penalty * (g_x.squeeze(-1) - self.slack_variables[i])
                    else:

                        self.slack_variables[i,:,t] = torch.clamp(self.dual_vars[i,:,t] / self.penalty + (g_x_func(x,t+1) - (1 - self.alpha)*g_x_func(x, t)).squeeze(-1), min=0)
                        self.dual_vars[i,:,t] = self.dual_vars[i,:,t] + self.penalty * ((g_x_func(x,t+1) - (1 - self.alpha)*g_x_func(x, t)).squeeze(-1) - self.slack_variables[i,:,t])
                        # self.penalty *= 1.0005

    def _project_to_feasible_region(self, x):


        xr1 = 2 * 1 / (self.norm_maxs[1] - self.norm_mins[1])
        yr1 = 2 * 1 / (self.norm_maxs[0] - self.norm_mins[0])
        off_x1 = 2 * (5.8 - 0.5 - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y1 = 2 * (5 - 0.5 - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
        for g_x_func in self.g_x_funcs:
            g_output = g_x_func(x)  # Expected shape (x.shape[0], x.shape[1]) or (x.shape[0], x.shape[1], 1)

            # Determine the source for condition mask based on g_output's shape
            if g_output.ndim == 3 and g_output.shape[-1] == 1 and g_output.shape[0] == x.shape[0] and g_output.shape[1] == x.shape[1]: # (B,H,1)
                g_condition_values = g_output.squeeze(-1) # Shape (B,H)
            elif g_output.ndim == 2 and g_output.shape[0] == x.shape[0] and g_output.shape[1] == x.shape[1]: # (B,H)
                g_condition_values = g_output
            else:
                raise ValueError(
                    f"Shape of g_x_func output ({g_output.shape}) is not directly usable for vectorized projection. "
                    f"Expected ({x.shape[0]}, {x.shape[1]}) or ({x.shape[0]}, {x.shape[1]}, 1)."
                )
            
            condition_mask = g_condition_values < 0  # Shape: (x.shape[0], x.shape[1])

            if torch.any(condition_mask):
                # Extract relevant channels and apply mask
                x_channel2 = x[:, :, 2]
                x_channel3 = x[:, :, 3]

                masked_x2_values = x_channel2[condition_mask]
                masked_x3_values = x_channel3[condition_mask]

                # Calculate r for the masked elements
                # r is the squared normalized distance from the center (off_y1, off_x1)
                term_y_sq = torch.pow((masked_x2_values - off_y1) / yr1, 2)
                term_x_sq = torch.pow((masked_x3_values - off_x1) / xr1, 2)
                r_masked = term_y_sq + term_x_sq

                # Initialize new values with current values for masked elements
                new_masked_x2 = masked_x2_values.clone()
                new_masked_x3 = masked_x3_values.clone()
                

                positive_r_sub_mask = r_masked > 1e-9  # Use a small epsilon

                if torch.any(positive_r_sub_mask):
                    r_to_process = r_masked[positive_r_sub_mask]
                    
                    x2_to_process = masked_x2_values[positive_r_sub_mask]
                    x3_to_process = masked_x3_values[positive_r_sub_mask]

                    # sqrt(1/r) is the scaling factor to project to the boundary
                    scaling_factor = torch.sqrt(1.0 / r_to_process)
                    
                    # Apply projection formula
                    projected_x2 = off_y1 + scaling_factor * (x2_to_process - off_y1)
                    projected_x3 = off_x1 + scaling_factor * (x3_to_process - off_x1)

                    # Update the new_masked_x values only for elements with positive r
                    new_masked_x2[positive_r_sub_mask] = projected_x2
                    new_masked_x3[positive_r_sub_mask] = projected_x3

                # Update the original tensor x using the main condition_mask
                x_channel2_updated = x_channel2.clone() # Clone to avoid in-place modification issues if x_channel2 is a view
                x_channel3_updated = x_channel3.clone()
                
                x_channel2_updated[condition_mask] = new_masked_x2
                x_channel3_updated[condition_mask] = new_masked_x3
                
                x = torch.stack([
                    x[:,:,0], x[:,:,1], 
                    x_channel2_updated, x_channel3_updated] 
                    + [x[:,:,i] for i in range(4, x.shape[2])], dim=2)
        # for g_x in self.g_x_funcs:
        #     g = g_x(x)
        #     for w in range(x.shape[0]):
        #         for h in range(x.shape[1]):
        #             if g[w,h,:].squeeze() < 0:
        #                 r = torch.pow((x[w, h, 2] - off_y1) / yr1, 2) + torch.pow((x[w, h, 3] - off_x1) / xr1, 2)
        #                 # print(torch.sqrt(1/r).shape)
        #                 # print(torch.tensor(off_y1).shape)
        #                 # print((x[:, :, 2] - off_y1).shape)
        #                 # print(x[:, h, 2].shape)
        #                 x[w, h, 2] = torch.tensor(off_y1) + torch.sqrt(1/r) * (x[w, h, 2] - off_y1)
        #                 x[w, h, 3] = torch.tensor(off_x1) + torch.sqrt(1/r) * (x[w, h, 3] - off_x1)
        return x

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        use_ddpm = True
        self.penalty = 2.5e-2
        if use_ddpm:
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        else:
            model_mean, _, model_log_variance = self.p_mean_variance_langevin(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)

        nonzero_mask = (1 - (t == 1).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        xp1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        if self.is_cons and self.algorithm == 'projected_gradient':
            xp1 = self._project_to_feasible_region(xp1)
        for g_x in self.g_x_funcs:
            self.safe = torch.relu(-g_x(xp1)).sum()
        self.record_safe.append(self.safe.item())

        return xp1
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, env=None, return_diffusion=False):
        device = self.betas.device
        self.record_safe = []
        

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        safe1, safe2 = [], []
        for i in reversed(range(-200, self.n_timesteps)):  #-100
            if i <= 0:
                i_ = 1
            elif i > self.n_timesteps:
                i_ = self.n_timesteps-1
            else:
                i_ = i
            timesteps = torch.full((batch_size,), i_, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
        

        self.record_safe = np.array(self.record_safe)
        # print(self.record_safe)
        # np.save('safe_cond_10.npy', self.record_safe)
        # plt.figure()
        # plt.plot(self.record_safe)
        # plt.show()
        
        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, env=None, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        print('shape',shape)
        if env is not None:
            return self.p_sample_cob_loop(shape, cond, return_diffusion= True, env=env, *args, **kwargs)
        return self.p_sample_loop(shape, cond, return_diffusion= True, env=None, *args, **kwargs)   ## debug

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, env=None):
        device = x_start.device
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon.to(torch.double), noise.to(torch.double))
        else:
            loss, info = self.loss_fn(x_recon.to(torch.double), x_start.to(torch.double))

        return loss, info
    

    def loss(self, x, cond, env=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, env)

    def forward(self, cond, env=None, *args, **kwargs):
        return self.conditional_sample(cond=cond, env=env, *args, **kwargs)

