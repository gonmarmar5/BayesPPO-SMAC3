import time
import gym
import random

env = gym.make("LunarLander-v2", render_mode="human")

def Random_games():
    # Each of this episode is its own game.
    for episode in range(10):
        env.reset()
        # this is each frame, up to 500...but we wont make it that far with random.
        while True:
            # This will display the environment
            env.render()
            
            # This will just create a sample action in any environment.
            # In this environment, the action can be any of one how in list on 4, for example [0 1 0 0]
            action = env.action_space.sample()

            '''
            observation (object): this will be an element of the environment's observation_space.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a terminal state (as defined under the MDP of the task) is reached.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
            info (dictionary): info contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
            '''
            next_state, reward, terminated, truncated, info = env.step(action)
            print(next_state, reward, truncated, info, action)
            
            if terminated:
                break
                
Random_games()