import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from .common import *
import functools


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class GaussianFourierProjection(Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        embed_dim = 64
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, context_dim+embed_dim),
            ConcatSquashLinear(128, 256, context_dim+embed_dim),
            ConcatSquashLinear(256, 512, context_dim+embed_dim),
            ConcatSquashLinear(512, 256, context_dim+embed_dim),
            ConcatSquashLinear(256, 128, context_dim+embed_dim),
            ConcatSquashLinear(128, 3, context_dim+embed_dim)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        time_emb = self.act(self.embed(beta))
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.sigma = 2.5

    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        # batch_size, _, point_dim = x_0[:, :, -3:].size()
        # if t == None:
        #     t = self.var_sched.uniform_sample_t(batch_size)
        # alpha_bar = self.var_sched.alpha_bars[t]
        # beta = self.var_sched.betas[t]

        # c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        # c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        # e_rand = torch.randn_like(x_0[:, :, -3:])  # (B, N, d)
        # x_t = torch.cat((x_0[:, :, :3], c0 * x_0[:, :, -3:] + c1 * e_rand),dim=2)
        # e_theta = self.net(x_t, beta=beta, context=context)

        # loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')

        def marginal_prob_std(t, sigma):
            """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
            """
            t = torch.tensor(t)
            return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
        def diffusion_coeff(t, sigma):
            """Compute the diffusion coefficient of our SDE.
            """
            return torch.tensor(sigma**t, device="cuda")
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=self.sigma)
        diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=self.sigma)
        def loss_fn(x, marginal_prob_std, eps=1e-5):
            """The loss function for training score-based generative models.
            """
            random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
            z = torch.randn_like(x)
            std = marginal_prob_std(random_t)
            perturbed_x = x + z * std[:, None, None]
            score = self.net(perturbed_x, random_t, context)
            loss = torch.mean((score + z)**2)
            return loss
        loss = loss_fn(x_0, marginal_prob_std_fn)
        return loss

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        eps=1e-5
        num_steps = 1000
        t = torch.ones(batch_size, device=context.device)

        def marginal_prob_std(t, sigma):
            """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
            """    
            t = torch.tensor(t)
            return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
        def diffusion_coeff(t, sigma):
            """Compute the diffusion coefficient of our SDE.
            """
            return torch.tensor(sigma**t, device=context.device)
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=self.sigma)
        diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=self.sigma)
        
        init_x = torch.randn(batch_size, num_points, point_dim, device=context.device) * marginal_prob_std_fn(t)[:, None, None]
        time_steps = torch.linspace(1., eps, num_steps, device=context.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        for itime_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=context.device) * itime_step
            g = diffusion_coeff_fn(batch_time_step)
            mean_x = x + (g**2)[:, None, None] * (self.net(x, batch_time_step, context)/marginal_prob_std_fn(batch_time_step)[:, None, None])* step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)
        return mean_x

        # batch_size = context.size(0)
        # x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        # traj = {self.var_sched.num_steps: x_T}
        # for t in range(self.var_sched.num_steps, 0, -1):
        #     z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        #     alpha = self.var_sched.alphas[t]
        #     alpha_bar = self.var_sched.alpha_bars[t]
        #     sigma = self.var_sched.get_sigmas(t, flexibility)

        #     c0 = 1.0 / torch.sqrt(alpha)
        #     c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

        #     x_t = traj[t]
        #     beta = self.var_sched.betas[[t]*batch_size]
        #     e_theta = self.net(x_t, beta=beta, context=context)
        #     x_next = c0 * (x_t - c1 * e_theta) + sigma * z
        #     traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
        #     traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
        #     if not ret_traj:
        #         del traj[t]
        
        # if ret_traj:
        #     return traj
        # else:
        #     return torch.cat((pc_0,traj[0]),dim=2), torch.cat((pc_0,x_T),dim=2)

    def mcmc_sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        snr=0.0000016#0.0016
        batch_size = context.size(0)
        eps=1e-5
        num_steps = 1000
        t = torch.ones(batch_size, device=context.device)

        def marginal_prob_std(t, sigma):
            """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
            """    
            t = torch.tensor(t)
            return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
        def diffusion_coeff(t, sigma):
            """Compute the diffusion coefficient of our SDE.
            """
            return torch.tensor(sigma**t, device=context.device)
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=self.sigma)
        diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=self.sigma)
        
        init_x = torch.randn(batch_size, num_points, point_dim, device=context.device) * marginal_prob_std_fn(t)[:, None, None]
        time_steps = torch.linspace(1., eps, num_steps, device=context.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        for itime_step in time_steps:
            print('itime_step = ', itime_step)
            batch_time_step = torch.ones(batch_size, device=context.device) * itime_step
            grad = (self.net(x, batch_time_step, context)/marginal_prob_std_fn(batch_time_step)[:, None, None])
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
            g = diffusion_coeff(batch_time_step,self.sigma)
            x_mean = x + (g**2)[:, None, None] * (self.net(x, batch_time_step, context)/marginal_prob_std_fn(batch_time_step)[:, None, None]) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None] * torch.randn_like(x)
        
        # The last step does not include any noise
        return x_mean
