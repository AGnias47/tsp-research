"""
Example from https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae

Building a DQN network. Still in progress.
"""

import numpy as np
import torch
import random
from config import config
from src.algorithms.q_learning.base_q_learning import BaseQLearning
from src.models.dqn import DQN
import torch.optim as optim
from collections import deque
import torch.nn as nn

class QLearning(BaseQLearning):
    algorithm_name = "Q-Learning"
    abbreviation = "q"

    def __init__(
        self,
        filepath: str,
        alpha: float = config.q_learning["alpha"],
        gamma: float = config.q_learning["gamma"],
        epsilon_func_key: str = config.q_learning["epsilon"],
        reward_func_key: str = config.q_learning["reward"],
        episodes: int = config.q_learning["episodes"],
    ):
        super().__init__(
            filepath=filepath,
            alpha=alpha,
            gamma=gamma,
            epsilon_func_key=epsilon_func_key,
            reward_func_key=reward_func_key,
            episodes=episodes,
        )
        self.Q = np.zeros(shape=(self.n + 1, self.n + 1))
        self.input_dim = self.n
        self.output_dim = 1
        self.batch_size = 64
        self.target_update_freq = 1000
        self.policy_net = DQN(input_dim=self.input_dim, output_dim=self.output_dim)
        self.target_net = DQN(input_dim=self.input_dim, output_dim=self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.memory = deque(maxlen=10_000)

    @property
    def hyperparameters(self):
        return config.q_learning

    def update_Q_table(self, state: int, action: int, reward: float, a_t1: int) -> None:
        """
        References
        ----------
        * https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm
        """
        Q_t = self.Q[state, action]
        Q_t1 = self.Q[action, a_t1]
        temporal_difference_target = reward + self.gamma * Q_t1
        temporal_difference_error = temporal_difference_target - Q_t
        self.Q[state, action] = Q_t + self.alpha * temporal_difference_error

    def exploit(self, s: int, environment: set[int]) -> int:
        state = torch.FloatTensor(environment).unsqueeze(0)
        q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)
        # Compute Q-values for current states
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def q_learning(self):
        # Main training loop
        rewards_per_episode = []
        steps_done = 0
        for episode in range(self.episodes):
            episode_cost = 0
            episode_starting_node = self.starting_node
            route = np.array([episode_starting_node])
            while len(route) < self.n:
                state = route[-1]
                action = self.next_action(int(state), set(self.G.nodes) - set(route))
                reward = self.reward(int(state), action)
                updated_route = np.concatenate((route, np.array([action])))
                updated_environment = set(self.G.nodes) - set(updated_route)
                if updated_environment:
                    a_t1 = self.next_action(action, updated_environment)
                else:
                    a_t1 = episode_starting_node
                route = updated_route
                episode_cost += self.dist(route[-2], route[-1])
                self.memory.append((state, action, reward, a_t1))
                # Update state
                state = action
                episode_cost += reward
                # Optimize model
                self.optimize_model()

                # Update target network periodically
                if steps_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                steps_done += 1
            rewards_per_episode.append(episode_cost)
