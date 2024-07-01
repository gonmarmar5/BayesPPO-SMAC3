import datetime
import os
import gymnasium
from matplotlib import pyplot as plt
import matplotlib   
matplotlib.use('Agg')  # Set the backend to a non-interactive one
import numpy as np
from stable_baselines3 import PPO

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter

#ENV = 'CartPole'
ENV = 'LunarLander'
TOTAL_TIMESTEPS = 200000
BATCH_SIZE = 128
EARLY_STOPPING = 900

class GenericSolver:
    @property
    def configspace(self) -> ConfigurationSpace:
        """
        Defines the hyperparameter search space for optimizing a PPO agent on the specified environment. 
        This includes learning rate, discount factor, and GAE lambda.

        Returns:
            ConfigurationSpace: A SMAC ConfigurationSpace object representing the valid ranges and types
                                of hyperparameters. 
        """

        cs = ConfigurationSpace(seed=0)
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=1e-2, default_value=1e-3, log=True)
        discount_factor = UniformFloatHyperparameter("discount_factor", lower=0.9, upper=0.999, default_value=0.99)
        gae_lambda = UniformFloatHyperparameter("gae_lambda", lower=0.8, upper=0.999, default_value=0.95)  
        
        cs.add_hyperparameters([learning_rate, discount_factor, gae_lambda])

        return cs
    
    def evaluate_agent(agent, env):
        """
        Evaluates the performance of an agent within the specified environment over a single episode.

        Args:
            agent: The trained agent to be evaluated.
            env: The CartPole environment instance.

        Returns:
            total_reward (float): The cumulative reward obtained by the agent during the episode.
        """

        obs, info = env.reset() # observation is the state of the environment
        terminated = False
        total_reward = 0
        while not terminated:
            action = agent.predict(obs)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    def train(self, config: Configuration, seed: int = None) -> float:
        """
        Trains a Proximal Policy Optimization (PPO) agent in the specified environment, tracks performance, and calculates a metric for SMAC's optimization.

        Args:
            config (Configuration): A configuration object containing hyperparameters for PPO.
            seed (int): Random seed for reproducibility (default: 0).

        Returns:
            float: The negative average reward achieved over the training process. This is used by SMAC to minimize (i.e., find lower values for better performance).
        """
        if ENV == 'CartPole':
            env = gymnasium.make('CartPole-v1')
        else:
            env = gymnasium.make('LunarLander-v2')
        
        print(f"Training with config: {config}, seed: {seed}")

        ppo_params = {
            'policy': 'MlpPolicy', # indicates that the policy will be represented by a feedforward neural network
            'env': env,
            'learning_rate': config['learning_rate'],
            'gamma': config['discount_factor'],
            'n_steps': 1024,
            'batch_size': 64,
            'n_epochs': 10,
            'gae_lambda': config['gae_lambda'],
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1
        }   

        agent = PPO(**ppo_params)

        num_agents = 5
        num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
        
        rewards = {}  # Track rewards over training
        
        # Agent training
        for update in range(1, num_updates + 1):  
            agent.learn(total_timesteps = BATCH_SIZE) # Numero de pasos antes de cada actualización

            total_reward = 0
            for agent_index in range(num_agents):
                individual_reward = GenericSolver.evaluate_agent(agent, env)
                agent_key = str(agent_index + 1)
                if agent_key in rewards:
                    rewards[agent_key].append(individual_reward)
                else:
                    rewards[agent_key] = [individual_reward]
                total_reward += individual_reward
            mean_reward = total_reward / num_agents
           
            if 'mean_reward' in rewards:
                rewards['mean_reward'].append(mean_reward)
            else:
                rewards['mean_reward'] = [mean_reward]

        env.close()

        if not os.path.exists("models"):
            os.makedirs("models")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join("models", f"ppo_basic_agent_{timestamp}")        
        agent.save(model_filename)

        return -np.mean(rewards['mean_reward']) # Calculate negative mean for SMAC's minimization
