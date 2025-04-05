"""
Example from https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Building a DQN network. Still in progress.
"""

import numpy as np
import torch
from config import config
from src.algorithms.q_learning.base_q_learning import BaseQLearning
from src.models.dqn import DQN
import torch.optim as optim
import torch.nn as nn
from src.models.replay_memory import ReplayMemory, Transition
import rainbow_tqdm

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class DeepQLearning(BaseQLearning):
    algorithm_name = "Deep Q-Learning"
    abbreviation = "dqn"

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
        self.batch_size = 128
        self.target_update_freq = 1000
        self.policy_net = DQN(n_observations=1, n_actions=self.n).to(device)
        self.target_net = DQN(n_observations=1, n_actions=self.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.memory = ReplayMemory(10_000)

    @property
    def hyperparameters(self):
        return config.q_learning

    def exploit(self, s: int, environment: set[int]) -> int:
        with torch.no_grad():
            rewards = self.policy_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device))
            max_reward = -torch.inf
            a_t1 = None
            for a in environment:
                reward = rewards[a]
                if reward > max_reward:
                    max_reward = reward
                    a_t1 = a
        return a_t1


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = (
            torch.tensor([s for s in batch.next_state if s is not None])
            .unsqueeze(1)
            .to(device)
        )
        # unsqueeze from gemini
        state_batch = torch.tensor(batch.state).unsqueeze(1).to(device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.tensor(batch.reward).unsqueeze(1).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(
            1, torch.tensor(action_batch, dtype=torch.int64)
        )

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def q_learning(self):
        # Main training loop
        steps_done = 0
        costs = []
        for self.episode in rainbow_tqdm.tqdm(range(self.episodes)):
            episode_cost = 0
            episode_starting_node = self.starting_node
            route = np.array([episode_starting_node])
            while len(route) < self.n:
                state = torch.tensor(route[-1], dtype=torch.float32)
                action = torch.tensor(
                    self.next_action(int(state), set(self.G.nodes) - set(route)),
                    dtype=torch.float32,
                )
                reward = torch.tensor(
                    self.reward(int(state), int(action)), dtype=torch.float32
                )
                updated_route = np.concatenate((route, np.array([action])))
                updated_environment = set(self.G.nodes) - set(updated_route)
                if updated_environment:
                    a_t1 = torch.tensor(
                        self.next_action(int(action), updated_environment),
                        dtype=torch.float32,
                    )
                else:
                    a_t1 = torch.tensor(episode_starting_node, dtype=torch.float32)
                route = updated_route
                self.memory.push(state, action, a_t1, reward)
                self.optimize_model()
                # Update target network periodically
                if steps_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                steps_done += 1
            episode_cost += self.dist(int(route[-1]), episode_starting_node)
            costs.append(episode_cost)
        return costs
