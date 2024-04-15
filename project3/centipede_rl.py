import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from itertools import count
import os
# To make Cuda work
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Set up matplotlib
plt.ion()

# If GPU is available, use it, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Named tuple to represent transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

episode_durations = []

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Replay memory class
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Deep Q-Network (DQN) class
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.linear1 = nn.Linear(in_features=5184, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
LIFE_LOST_PENALTY = 100

# Get the number of actions from the gym action space
env = gym.make('CentipedeNoFrameskip-v4')
env.metadata['render_fps'] = 30
#Atari preprocessing wrapper
env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
#Frame stacking
env = gym.wrappers.FrameStack(env, 4)

n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
# breakpoint()

# Argument parser
parser = argparse.ArgumentParser(prog='q-learning', description='learns to play a game')
parser.add_argument('-s', '--save', default="centipede.pytorch", help="file to save model to", type=str)
parser.add_argument('-l', '--load', default=None, help="file to load model from", type=str)
args = parser.parse_args()

# Initialize policy and target networks
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
if args.load is not None:
    print (f"loading {args.load}")
    policy_net.load_state_dict(torch.load(args.load))
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

# Select action function
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return torch.argmax(policy_net(state)).view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
        plt.savefig("centipede_plt.png")
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Training function
def train_model():
    num_episodes = 1000
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
        prev_lives = info['lives']  # Get the initial number of lives
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, info = env.step(action.item())
            current_lives = info['lives']  # Get the current number of lives
            reward = torch.tensor([reward], device=device)

            # Decrease reward if a life is lost
            if current_lives < prev_lives:
                reward -= LIFE_LOST_PENALTY

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(np.array(observation), dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state
            prev_lives = current_lives  # Update the previous lives

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

# Optimization function
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def run_model(count = 100):
    """You should probably not modify this, other than
    to load centipede.
    """
    env = gym.make('CentipedeNoFrameskip-v4', render_mode="human")
    env.metadata['render_fps'] = 30
    #Atari preprocessing wrapper
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    #Frame stacking
    env = gym.wrappers.FrameStack(env, 4)
    #breakpoint()
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
    prev_lives = info['lives']  # Get the initial number of lives
    for t in range(count):
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        current_lives = info['lives']  # Get the current number of lives
        reward = torch.tensor([reward], device=device)
        # Decrease reward if a life is lost
        if current_lives < prev_lives:
            reward -= LIFE_LOST_PENALTY

        done = terminated or truncated

        if terminated:
            state = None
        else:
            state = torch.tensor(np.array(observation), dtype=torch.float32, device=device).unsqueeze(0)
            prev_lives = current_lives  # Update the previous lives
        env.render()
        if done:
            break

# Main function
def main():
    #train_model()
    #torch.save(policy_net.state_dict(), args.save)
    #print('Training complete')
    plot_durations(show_result=True)
    run_model(1000000)

if __name__ == "__main__":
    main()