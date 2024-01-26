import argparse
import sys
import pdb
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
        return self.action_space.sample()

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
        #pdb.set_trace()
        score += reward
        env.render()
     
    # Close the env and write monitor result info to disk
    print ("Your score: %d" % score)
    env.close()
