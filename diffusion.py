import torch
import torch.nn as nn
from diffusers import DDPMScheduler


class SimpleMLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 256),  # +1 for timestep embedding
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x, t, state):
        t_emb = t.unsqueeze(-1).float() / 1000.0
        x_input = torch.cat([x, t_emb, state], dim=-1)
        return self.net(x_input)


class ActionDiffusion(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        model,
        n_timesteps=1000,
        beta_schedule="linear",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.model = model

        self.scheduler = DDPMScheduler(
            num_train_timesteps=n_timesteps, beta_schedule=beta_schedule
        )

    def q_sample(self, x_start, noise, t):
        return self.scheduler.add_noise(x_start, noise, t)

    def loss(self, x_start, state):
        batch_size = x_start.shape[0]
        noise = torch.randn_like(x_start)
        t = torch.randint(
            0, self.scheduler.num_train_timesteps, (batch_size,), device=x_start.device
        ).long()
        x_noisy = self.q_sample(x_start, noise, t)
        x_recon = self.model(x_noisy, t, state)
        return nn.functional.mse_loss(x_recon, noise)

    @torch.no_grad()
    def sample(self, state, batch_size):
        device = next(self.parameters()).device
        x = torch.randn((batch_size, self.action_dim), device=device)
        for i in reversed(range(self.scheduler.num_train_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            model_pred = self.model(x, t, state)
            x = self.scheduler.step(model_pred, t, x).prev_sample
        return x.clamp(-self.max_action, self.max_action)
