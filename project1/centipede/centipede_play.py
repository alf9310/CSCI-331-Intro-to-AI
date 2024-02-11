import argparse
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.ndimage import measurements
import gymnasium as gym
from gymnasium import wrappers, logger


class Agent(object):
    """The world's simplest agent!"""

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

        # Store the 8 level colors
        mushroom_colors = np.array([[181, 83, 40], [45, 50, 184], [187, 187, 53], [184, 70, 162], [184, 50, 50]])
        centipede_colors = np.array([[184, 70, 162], [184, 50, 50], [146, 70, 192], [110, 156, 66], [84, 138, 210]])
        spider_colors = np.array([[146, 70, 192], [110, 156, 66], [84, 138, 210], [181, 83, 40], [45, 50, 184]])
        bar_colors = np.array([[110, 156, 66], [66, 114, 194], [198, 108, 58], [66, 72, 200], [162, 162, 42]])

        # Use bar color to identify levels 0-7
        bar_color = observation[183][16]  # static location of bar
        level = 0
        for index, color in enumerate(bar_colors):
            if (bar_color == color).all():
                level = index

        # Remove duplicates from observation
        observation = observation[:, :, ::3]

        # Sprite locations where values are [y, y, y][x, x, x]
        mushroom_loc = np.array(np.where(observation == mushroom_colors[level])[:2])
        centipede_loc = np.array(np.where(observation == centipede_colors[level])[:2])
        spider_loc = np.array(np.where(observation == spider_colors[level])[:2])
        # Remove sprite observations below bar
        spider_loc = np.delete(spider_loc, [np.where(spider_loc == 183), np.where(spider_loc == 184)], axis=1)
        centipede_loc = np.delete(centipede_loc, [np.where(centipede_loc == 183), np.where(centipede_loc == 184)],
                                  axis=1)

        # Identify player location by checking size of the sprite
        player_loc = []
        for y, x in zip(*mushroom_loc):
            # Check if the region around the detected point is the player size
            if np.sum(observation[y:y + 9, x:x + 4] == mushroom_colors[level]) >= 36:
                player_loc.append(y)
                player_loc.append(x)

        # Avoid the spider and centipede locations 
        spider_action = collision(player_loc, spider_loc)
        if spider_action is not None:
            return spider_action
        centipede_action = collision(player_loc, centipede_loc)
        if centipede_action is not None:
            return centipede_action

        # If possible, shoot the spider
        if len(player_loc) != 0:
            for x in flip(spider_loc[1]):
                for y in flip(spider_loc[0]):
                    if player_loc[1] - 1 < x < player_loc[1] + 4 and player_loc[0] > y:
                        return 1 # Fire
                    
        # Line up with and shoot the centipede
        if len(player_loc) != 0:
            for x in flip(centipede_loc[1]):
                for y in flip(centipede_loc[0]):
                    if player_loc[0] > y and player_loc[1] - 1 < x < player_loc[1] + 4:
                        return 1 # Fire
                    elif player_loc[0] > y and x < player_loc[1] + 2:
                        return 4 # Left
                    elif player_loc[0] > y and player_loc[1] + 1 < x:
                        return 3 # Right
                    elif player_loc[0] + 2 < y:
                        return 5 # Down
                    
        return self.action_space.sample() # Do random :)


def collision(player_loc, enemy_loc):
    """Avoid the location of the enemies.

        Args:
            player_loc: location of the upper left pixel of the player.
                Format: [y, x]
            enemy_loc: location of the enemy pixels.
                Format: [[y, y, y, ...][x, x, x, ...]]

        Returns:
            A numerical code giving the action to take. See
            See the Actions table at:
            https://gymnasium.farama.org/environments/atari/centipede/
        """
    if len(player_loc) != 0:
        for x in flip(enemy_loc[1]): # Search for lower enemies first
            for y in flip(enemy_loc[0]):
                # Hitbox buffer
                if player_loc[1] - 10 < x < player_loc[1] + 13 and player_loc[0] - 10 < y < player_loc[0] + 18:
                    if x < player_loc[1] and y > player_loc[0]:
                        return 3 # Right
                    elif player_loc[1] + 3 < x and y > player_loc[0]:
                        return 4 # Left
                    elif y > player_loc[0] + 8:
                        return 2 # Up
                    elif player_loc[0] > y:
                        return 13 # Down and shoot


## YOU MAY NOT MODIFY ANYTHING BELOW THIS LINE OR USE
## ANOTHER MAIN PROGRAM
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', nargs='?', default='Centipede-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id, render_mode="human")

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'random-agent-results'

    env.unwrapped.seed(0)
    agent = Agent(env.action_space)

    episode_count = 100
    reward = 0
    terminated = False
    score = 0
    special_data = {}
    special_data['ale.lives'] = 3
    observation = env.reset()[0]

    while not terminated:
        action = agent.act(observation, reward, terminated)
        observation, reward, terminated, truncated, info = env.step(action)
        # pdb.set_trace()
        score += reward
        env.render()

    # Close the env and write monitor result info to disk
    print("Your score: %d" % score)
    env.close()
