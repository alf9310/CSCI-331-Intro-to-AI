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

    # You should modify this function
    # You may define additional functions in this
    # file or others to support it.
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

        # new_array = observation.reshape((observation.shape[0] * observation.shape[1], 3))
        # colors = {tuple(x) for x in new_array}

        # Store the 8 level colors TODO
        mushroom_colors = np.array([[181, 83, 40]])
        centipede_colors = np.array([[184, 70, 162]])
        spider_colors = np.array([[146, 70, 192]])
        bar_colors = np.array([[110, 156, 66]])

        # Use bar color to identify levels 0-7
        bar_color = observation[183][16]  # static location of bar
        level = 0
        for index, color in enumerate(bar_colors):
            if (bar_color == color).all():
                level = index

        # CHANGE TODO
        observation = observation[:, :, ::3]
        # print(observation)
        # Location where values are [(y,x), (y,x)]
        '''
        mushroom_loc = np.array(list(zip(*np.where(observation == mushroom_colors[level])[:2])))
        centipede_loc = np.array(list(zip(*np.where(observation == centipede_colors[level])[:2])))
        spider_loc = np.array(list(zip(*np.where(observation == spider_colors[level])[:2])))
        '''

        # Locations where values are [y, y, y][x, x, x]
        mushroom_loc = np.array(np.where(observation == mushroom_colors[level])[:2])
        centipede_loc = np.array(np.where(observation == centipede_colors[level])[:2])
        spider_loc = np.array(np.where(observation == spider_colors[level])[:2])

        # Identify player location by checking size
        player_loc = []
        for y, x in zip(*mushroom_loc):
            # print("(" + str(y) + "," + str(x) + ") ")
            # Check if the region around the detected point is the player size
            if np.sum(observation[y:y + 9, x:x + 4] == mushroom_colors[level]) >= 36:
                # player_loc.append((y, x))
                player_loc.append(y)
                player_loc.append(x)

        # player_loc = np.array(player_loc)
        # print(player_loc)
        '''
        # Sanity check graph
        plt.scatter(mushroom_loc[1], mushroom_loc[0], alpha=.5, color="orange")
        plt.scatter(centipede_loc[1], centipede_loc[0], alpha=.5, color="pink")
        #plt.scatter(player_loc[1], player_loc[0], alpha=.5, color="blue")
        plt.scatter(player_loc[:, 1], player_loc[:, 0], alpha=.5, color="blue")
        #plotting hitbox to see what would be a reasonable size
        plt.scatter(player_loc[:, 1] - 4, player_loc[:, 0] - 4, alpha=.5, color="red")
        plt.scatter(player_loc[:, 1] + 7, player_loc[:, 0] - 4, alpha=.5, color="red")
        plt.scatter(player_loc[:, 1] - 4, player_loc[:, 0] + 12, alpha=.5, color="red")
        plt.scatter(player_loc[:, 1] + 7, player_loc[:, 0] + 12, alpha=.5, color="red")
        plt.scatter(spider_loc[1], spider_loc[0], alpha=.5, color="purple")
        plt.title('level = ' + str(level))
        plt.gca().invert_yaxis()
        plt.show()

        breakpoint()'''

        # Focuses on avoiding the spider first
        if len(player_loc) != 0:
            for x in spider_loc[1]:
                for y in spider_loc[0]:
                    if player_loc[1] - 10 < x < player_loc[1] + 13 and player_loc[0] - 10 < y < player_loc[0] + 18:
                        if x < player_loc[1]:
                            # avoid by moving right
                            return 3
                        elif player_loc[1] + 3 < x:
                            # avoid by moving left
                            return 4
                        elif y > player_loc[0] + 8:
                            # avoid by moving up
                            return 2
                        elif player_loc[0] > y:
                            # avoid by moving down and shooting
                            return 13

        # If available, shoot the spider
        if len(player_loc) != 0:
            for x in spider_loc[1]:
                for y in spider_loc[0]:
                    if player_loc[1] - 1 < x < player_loc[1] + 4 and player_loc[0] > y:
                        return 1

        # making notes here: so the way it's going through the centipede locations, it starts at
        # the top of the grid and makes its way down, which means that the sprite is trying to
        # fire at the centipede parts closest to the top of the screen rather than the one closest
        # to itself/the bottom of the screen. I want to figure out a way to either reverse traverse
        # the centipede's location OR to find all y points and aim for the one closest to the sprite

        # More notes: while basic firing and avoidance has been implemented, edge cases still need
        # defining for the literal edges of the screen. Basically, the agent needs to know that if
        # it is at the right edge of the screen, and it's normal direction would be to move right to
        # avoid an enemy, it should instead move up or down or something

        # thanks for reading <3 -Thalia K

        # Avoid the centipede if nearby
        if len(player_loc) != 0:
            for x in centipede_loc[1]:
                for y in centipede_loc[0]:
                    if player_loc[1] - 10 < x < player_loc[1] + 13 and player_loc[0] - 10 < y < player_loc[0] + 18:
                        if x < player_loc[1]:
                            # avoid by moving right
                            return 6
                        elif player_loc[1] + 3 < x:
                            # avoid by moving left
                            return 7
                        elif y > player_loc[0] + 8:
                            # avoid by moving up
                            return 2
                        elif player_loc[0] > y:
                            # avoid by moving down and shooting
                            return 13

        # Fire at the centipede, if possible
        if len(player_loc) != 0:
            for x in centipede_loc[1]:
                for y in centipede_loc[0]:
                    if player_loc[0] > y and player_loc[1]-2 < x < player_loc[1]+5:
                        # print("fire")
                        return 1
                    elif player_loc[0] > y and x < player_loc[1] + 2:
                        # print("left")
                        return 4
                    elif player_loc[0] > y and player_loc[1] + 1 < x:
                        # print("right")
                        return 3
                    elif player_loc[0]+2 < y:
                        return 5
        return 0
        # return self.action_space.sample()


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