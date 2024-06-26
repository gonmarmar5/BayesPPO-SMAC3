import datetime
import os
import gymnasium
from matplotlib import pyplot as plt
import matplotlib   
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter

#ENV = 'CartPole'
ENV = 'LunarLander'
MIN_BUDGET = 1
MAX_BUDGET = 8
MAX_TIMESTEPS = 200000
MIN_TIMESTEPS = 75000
BATCH_SIZE = 512
EARLY_STOPPING = 3600

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, dropout_prob=0.2):
        """
        Initializes the feature extractor with a feedforward neural network.

        Args:
        - observation_space (gym.spaces.Space): The observation space of the environment.
        - features_dim (int): Dimensionality of the extracted features (default is 64).
        """
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),  # Capa de entrada más pequeña
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Batch normalization después de la primera capa lineal
            nn.Dropout(p=dropout_prob),  # Dropout después de la primera capa lineal
            nn.Linear(128, 64),  # Segunda capa oculta más pequeña
            nn.ReLU(),
            nn.BatchNorm1d(64),  # Batch normalization después de la segunda capa lineal
            nn.Dropout(p=dropout_prob),  # Dropout después de la segunda capa lineal
            nn.Linear(64, features_dim),  # Capa de salida
            nn.Tanh()  # Función de activación para la salida
        )

    def forward(self, x):
        """
        Computes forward pass of the neural network to extract features from input x.

        Args:
        - x (torch.Tensor): Input tensor containing observations.

        Returns:
        - torch.Tensor: Extracted features tensor.
        """
        return self.net(x)

class CustomMLPPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        """
        Initializes the policy with a custom MLP architecture and feature extractor.

        Args:
        - observation_space (gym.spaces.Space): The observation space of the environment.
        - action_space (gym.spaces.Space): The action space of the environment.
        - lr_schedule (callable): Learning rate schedule for the optimizer.
        - *args: Additional positional arguments for parent classes.
        - **kwargs: Additional keyword arguments for parent classes.
        """
        super(CustomMLPPolicy, self).__init__(observation_space, action_space, lr_schedule,
                                              features_extractor_class=CustomFeatureExtractor,
                                              features_extractor_kwargs=dict(features_dim=256),
                                              *args, **kwargs)

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
            env: The specified environment instance.

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
    
    def plot_training(mean_rewards, timestamp):
        if not os.path.exists("plots"):
            os.makedirs("plots")
        
        plt.figure()
        plt.plot(mean_rewards)
        plt.xlabel('Update')
        plt.ylabel('Mean Reward')
        plt.title('Training Progress')
        plot_filename = os.path.join("plots", f"ppo_training_plot_{timestamp}.png")
        plt.savefig(plot_filename)
        plt.close()

    def train(self, config: Configuration, seed: int = None, budget: float = None) -> float:
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
            'policy': CustomMLPPolicy, # the policy will be represented by the previous feedforward neural network
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

        # this calculates the total number of timesteps based on the budget
        total_timesteps = int(((budget - MIN_BUDGET) / (MAX_BUDGET - MIN_BUDGET)) * (MAX_TIMESTEPS - MIN_TIMESTEPS) + MIN_TIMESTEPS)
        print("Total TimeSteps: ", total_timesteps)
        
        num_agents = 5
        num_updates = total_timesteps // BATCH_SIZE
        rewards = {}
        
        for update in range(1, num_updates + 1):  
            agent.learn(total_timesteps = BATCH_SIZE)

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

        # Save model
        if not os.path.exists("models"):
            os.makedirs("models")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join("models", f"ppo_multifidelity_agent_{timestamp}")        
        agent.save(model_filename)
        
        # Plot rewards
        GenericSolver.plot_training(rewards['mean_reward'], timestamp)

        return -np.mean(rewards['mean_reward']) # Calculate negative mean for SMAC's minimization
    