import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from stable_baselines3 import TD3

from Network import CommunicationNetwork, Network

console = Console()


class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.max_size = max_size
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.infos = []

    def add(self, obs, action, reward, next_obs, done, info):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
        self.infos.append(info)

        if len(self.observations) > self.max_size:
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_observations.pop(0)
            self.dones.pop(0)
            self.infos.pop(0)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.observations), batch_size)
        return {
            "observations": [self.observations[i] for i in indices],
            "actions": [self.actions[i] for i in indices],
            "rewards": [self.rewards[i] for i in indices],
            "next_observations": [self.next_observations[i] for i in indices],
            "dones": [self.dones[i] for i in indices],
            "infos": [self.infos[i] for i in indices],
        }


class TD3_BC(TD3):
    def __init__(self, policy, env, bc_coef=0.5, eval_frequency=1000, **kwargs):
        super(TD3_BC, self).__init__(policy, env, **kwargs)
        self.bc_coef = bc_coef
        self.behavior_cloning_loss = nn.MSELoss().to(self.device)
        self.critic_loss = nn.MSELoss()
        self.env = env

        self.max_size = 10000000
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.infos = []
        self.total_time = 0

        self.training_rewards = []
        self.evaluation_metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "average_reward": [],
            "capture_rate": [],
            "sum_reward": [],
        }
        self.eval_frequency = eval_frequency

        self.action_shape = 5
        self.state_shape = 6
        self.max_action = 1
        self.beta_schedule = "vp"
        self.n_timesteps = 10
        self.bc_coef = False
        self.actor_lr = 0.0005
        self.wd = 1e-4
        self.algorithm_times = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        self.chasers_comm_network = CommunicationNetwork(
            input_dim=16, output_dim=14
        ).to(self.device)
        self.runners_comm_network = CommunicationNetwork(input_dim=14, output_dim=4).to(
            self.device
        )
        self.network = Network(input_dim=16, output_dim=14).to(self.device)

    def learn(
        self,
        total_timesteps,
        eval_env=None,
        batch_size=128,
    ):
        console.log("[bold blue]🚀 Start learning...[/bold blue]")
        total_steps = 0
        gradient_steps = 1000

        self._fill_replay_buffer(num_steps=100000)

        while total_steps < total_timesteps:
            gradient_steps = min(gradient_steps, total_timesteps - total_steps)
            self.train(gradient_steps, eval_env, batch_size=batch_size)
            total_steps += gradient_steps

        console.log("[bold green]✅ Learning finished![/bold green]")
        console.log(
            f"[yellow]Total training time: {sum(self.algorithm_times):.2f}s[/yellow]"
        )

    def train(self, gradient_steps, eval_env, batch_size=100):
        console.log("[cyan]🔥 Start training...[/cyan]")

        for gradient_step in range(gradient_steps):
            start_time = time.time()
            next_observations = []
            runners_obs, chasers_obs = [], []

            replay_data = self.ReplayBuffer.sample(batch_size)
            actions = torch.tensor(replay_data["actions"], dtype=torch.float32).to(
                self.device
            )

            for next_obs in replay_data["next_observations"]:
                padded_obs = (
                    F.pad(torch.tensor(next_obs, dtype=torch.float32), (0, 2))
                    if len(next_obs) == 14
                    else torch.tensor(next_obs, dtype=torch.float32)
                )
                next_observations.append(padded_obs)

            rewards = torch.tensor(replay_data["rewards"], dtype=torch.float32).to(
                self.device
            )
            dones = torch.tensor(replay_data["dones"], dtype=torch.float32).to(
                self.device
            )

            for obs in replay_data["observations"]:
                if isinstance(obs, torch.Tensor):
                    tensor_obs = obs.clone().detach()
                else:
                    tensor_obs = torch.tensor(obs, dtype=torch.float32)
                (runners_obs if len(obs) == 14 else chasers_obs).append(tensor_obs)

            chaser_obs_tensor = (
                torch.stack(chasers_obs) if chasers_obs else torch.tensor([])
            ).to(self.device)
            communicated_observations = self.chasers_comm_network(chaser_obs_tensor)

            if communicated_observations.size(0) % batch_size != 0:
                padding_size = batch_size - (
                    communicated_observations.size(0) % batch_size
                )
                padding = torch.zeros(
                    (padding_size, communicated_observations.size(1)),
                    device=self.device,
                )
                communicated_observations = torch.cat(
                    (communicated_observations, padding), dim=0
                )

            communicated_observations = communicated_observations.reshape(
                batch_size, -1
            )

            with torch.no_grad():
                noise = actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                noise = noise.unsqueeze(1).expand(-1, self.action_space.shape[0])
                next_observations = torch.stack(
                    [obs.float() for obs in next_observations]
                ).to(self.device)
                next_observations = self.network(next_observations)

                next_actions = (self.actor_target(next_observations) + noise).clamp(
                    -self.action_space.high[0], self.action_space.high[0]
                )
                target_q1, target_q2 = self.critic_target(
                    next_observations, next_actions
                )
                target_q = rewards.unsqueeze(1) + (
                    1 - dones.unsqueeze(1)
                ) * self.gamma * torch.min(target_q1, target_q2)

            actions = actions.unsqueeze(1).expand(-1, 5)

            current_q1, current_q2 = self.critic(communicated_observations, actions)
            critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(
                current_q2, target_q
            )

            self.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic.optimizer.step()

            if gradient_step % self.policy_delay == 0:
                actions_pred = self.actor(communicated_observations)
                actor_loss = -self.critic.q1_forward(
                    communicated_observations, actions_pred
                ).mean()

                replay_actions = torch.tensor(
                    replay_data["actions"], dtype=torch.float32
                ).to(self.device)
                if replay_actions.dim() == 1:
                    replay_actions = replay_actions.unsqueeze(1).expand(
                        -1, actions_pred.shape[1]
                    )

                bc_loss = self.behavior_cloning_loss(actions_pred, replay_actions)
                actor_loss += self.bc_coef * bc_loss

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                self.soft_update(self.critic_target, self.critic, self.tau)
                self.soft_update(self.actor_target, self.actor, self.tau)

            self.evaluation_metrics["actor_loss"].append(actor_loss.item())
            self.evaluation_metrics["critic_loss"].append(critic_loss.item())
            average_reward, capture_rate = self.evaluate(eval_env)
            sum_reward = sum(average_reward.values())
            self.evaluation_metrics["sum_reward"].append(sum_reward)

            console.log(
                f"[bold magenta]Step {gradient_step}[/bold magenta] | "
                f"🎯 Actor Loss: {actor_loss:.4f} | 💔 Critic Loss: {critic_loss:.4f} | "
                f"🏆 Avg Reward: {average_reward} | 🎯 Capture Rate: {capture_rate:.2f}"
            )

            end_time = time.time()
            self.total_time += end_time - start_time
            self.algorithm_times.append(end_time - start_time)

        console.log("[green]✅ Training completed![/green]")

    def evaluate(self, eval_env, num_episodes=10):
        capture_count = 0
        for _ in range(num_episodes):
            eval_env.pettingzoo_env.reset()
            obs = {agent: None for agent in eval_env.pettingzoo_env.agents}
            total_rewards = {agent: 0 for agent in eval_env.pettingzoo_env.agents}

            for agent in eval_env.pettingzoo_env.agent_iter():
                observation, reward, termination, truncation, info = (
                    eval_env.pettingzoo_env.last()
                )
                action = (
                    None
                    if termination or truncation
                    else eval_env.pettingzoo_env.action_space(agent).sample()
                )
                total_rewards[agent] += reward

                if "capture" in info and info["capture"]:
                    capture_count += 1

                eval_env.pettingzoo_env.step(action)
                obs[agent] = observation

        average_reward = {
            agent: total_rewards[agent] / num_episodes for agent in total_rewards
        }
        capture_rate = capture_count / num_episodes

        return average_reward, capture_rate

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _fill_replay_buffer(self, num_steps=0):
        console.log("[bold cyan]⏳ Filling replay buffer...[/bold cyan]")
        obs = self.env.reset()
        steps = 0
        self.ReplayBuffer = ReplayBuffer(max_size=self.max_size)
        while steps <= num_steps:
            for agent in self.env.pettingzoo_env.agent_iter():
                done = False
                action = self.env.pettingzoo_env.action_space(agent).sample()
                self.env.pettingzoo_env.step(action)
                next_obs = self.env.pettingzoo_env.observe(agent)
                reward, termination, truncation, info = self.env.pettingzoo_env.last()[
                    1:
                ]
                done = termination or truncation

                # if agent != "agent_0":
                obs = (
                    F.pad(torch.tensor(obs, dtype=torch.float32), (0, 2))
                    if len(obs) == 14
                    else obs
                )
                next_obs = (
                    F.pad(torch.tensor(next_obs, dtype=torch.float32), (0, 2))
                    if len(next_obs) == 14
                    else next_obs
                )
                if len(obs) == 16 and len(next_obs) == 16:
                    self.ReplayBuffer.add(obs, action, reward, next_obs, done, info)

                obs = next_obs
                steps += 1
                if done:
                    self.env.reset()
                    obs = self.env.pettingzoo_env.observe(agent)

                if steps > num_steps:
                    break

        console.log("[bold green]✅ Replay buffer filled![/bold green]")
