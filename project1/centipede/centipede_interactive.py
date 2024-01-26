import argparse
import sys
import pdb
import gymnasium as gym
import time
import numpy as np
from gymnasium import wrappers, logger
#import keyboard
import jsonlines

"""
See: https://gymnasium.farama.org/environments/atari/centipede/

For details about the environment.
"""
import itertools

from pynput import keyboard

pressed = set()

def on_press(key):
    """Triggered whenever a key is pressed.

    Adds pressed key to the "pressed" set.

    Args:
        key: the key pressed
    """

    global pressed
    val = ""
    try:
        val = key.char
    except AttributeError:
        print('special key {0} pressed'.format(
            key))
    pressed.add(val)


def on_release(key):
    """Triggered whenever a key is released.

    Removes released key to the "pressed" set.

    Args:
        key: the key pressed
    """

    global pressed
    try:
        pressed.remove(key.char)
    except: 
        print('{0} released'.format(
            key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
listener.start()


direction_keys = ['u', 'k', 'h', 'm', 'i', 'y', ',', 'n']


class Agent(object):
    """The world's simplesst agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        """Compute and action based on the current state.

        Args:
            observation: a 3d array of the game screen pixels.
                Format: rows x columns x rgb.
            reward: the reward associated with the current state.
            done: whether or not it is a terminal state.

        Returns:
            A numerical code giving the action to take. See
            See the Actions table at:
            https://gymnasium.farama.org/environments/atari/centipede/
        """
        global pressed
        global keys
        action = 0

        i = 1
        # Encode direction, using table from https://gymnasium.farama.org/environments/atari/centipede/
        for key in direction_keys:
            i = i + 1
            if str(key) in pressed:
                action = i

        # Encode fire action, if z is pressed
        if 'z' in pressed:
            if action > 1:
                action += 8      
            else:
                action = 1
    
        return action

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_id', nargs='?', default='Centipede-v0', help='Select the environment to run')
    parser.add_argument('-r', '--record', default=None, help="file to record to, if any")
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id, render_mode="human", obs_type="grayscale")

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'random-agent-results'


    env.unwrapped.seed(0)
    agent = Agent(env.action_space)

    episode_count = 100
    reward = 0
    done = False
    score = 0
    special_data = {}
    special_data['ale.lives'] = 3
    observation = env.reset()

    terminated = False
    truncated = False
    moves = []
    while not (truncated or terminated):
        
        action = agent.act(observation, reward, done)
        time.sleep(.05)

        #pdb.set_trace()
        observation_next, reward, terminated, truncated, info = env.step(action)

        score += reward

        env.render()
        observation = observation_next
        if terminated:
            print("terminated")
        if truncated:
            print("truncated")

     
    # Close the env and write monitor result info to disk
    print ("Your score: %d" % score)
    env.close()
    if args.record is not None:
        writer.close()
