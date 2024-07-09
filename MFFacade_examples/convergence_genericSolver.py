import datetime
import os
import time
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

ENV = 'CartPole'
#ENV = 'LunarLander'
MIN_BUDGET = 1
MAX_BUDGET = 10
MAX_TIMESTEPS = 10000
MIN_TIMESTEPS = 5000
EARLY_STOPPING = 180
REWARD_THRESHOLD = 350
NUM_REWARD_EVALS = 30 
COUNTER = 0

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

class EvalCallbackStopOnThreshold(EvalCallback):
    def __init__(
        self,
        eval_env,
        threshold,
        consecutive_evals,
        n_eval_episodes,
        eval_freq,
        log_path,
        best_model_save_path,
        deterministic,
        render,
        verbose
    ):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
        )
        self.threshold = threshold
        self.consecutive_evals = consecutive_evals
        self.last_mean_reward_recorded = float('-inf')
        global COUNTER  # Declarar la variable global
        COUNTER = 0


    def _on_step(self) -> bool:
        global COUNTER
        result = super()._on_step()
        if self.last_mean_reward >= self.threshold:
            if self.last_mean_reward_recorded != self.last_mean_reward:
                COUNTER += 1
                self.last_mean_reward_recorded = self.last_mean_reward
                if self.verbose > 0:
                    print(f"Evaluation {self.num_timesteps}: mean reward {self.last_mean_reward:.2f} - consecutive {COUNTER}")
        else:
            COUNTER = 0
            self.last_mean_reward_recorded = float('-inf')  # Reset to ensure future increases are counted

        if COUNTER >= self.consecutive_evals:
            if self.verbose > 0:
                print(f"Stopping training as the mean reward {self.last_mean_reward:.2f} was above the threshold {self.threshold} for {self.consecutive_evals} consecutive evaluations.")
            return False
        
        return result

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
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=1e-4, upper=1e-2, default_value=1e-3, log=True)
        discount_factor = UniformFloatHyperparameter("discount_factor", lower=0.9, upper=0.999, default_value=0.99)
        gae_lambda = UniformFloatHyperparameter("gae_lambda", lower=0.9, upper=0.999, default_value=0.95)  
        
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
        plt.xlabel('Evaluation')
        plt.ylabel('Accumulated Reward')
        plt.title('Training Progress')
        if ENV == 'CartPole':
            plt.ylim(-550, 550)  # Ajustar los límites del eje y
        else:
            plt.ylim(-350, 350)
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
            eval_env = Monitor(gymnasium.make('CartPole-v1'))
        else:
            env = gymnasium.make('LunarLander-v2')
            eval_env = Monitor(gymnasium.make('LunarLander-v2'))

        print(f"Training with config: {config}, seed: {seed}")
        
        ppo_params = {
            #'policy': CustomMLPPolicy, # indicates that the policy will be represented by a feedforward neural network
            'policy': 'MlpPolicy',
            'env': env,
            'learning_rate': config['learning_rate'],
            'gamma': config['discount_factor'],
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gae_lambda': config['gae_lambda'],
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1
        }   
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
        
        agent = PPO(
            policy_kwargs=policy_kwargs,
            **ppo_params)
    
        # this calculates the total number of timesteps based on the budget
        total_timesteps = int(((budget - MIN_BUDGET) / (MAX_BUDGET - MIN_BUDGET)) * (MAX_TIMESTEPS - MIN_TIMESTEPS) + MIN_TIMESTEPS)
        print("Total TimeSteps: ", total_timesteps)
        
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=100,
                                 deterministic=True, render=False)
        
        #OPTION 2: STOP WHEN REACHING N TIMES THE TRESHOLD
        eval_callback = EvalCallbackStopOnThreshold(
            eval_env=eval_env,
            threshold=REWARD_THRESHOLD,
            consecutive_evals=NUM_REWARD_EVALS,
            n_eval_episodes=5,
            eval_freq=total_timesteps // 100,
            log_path='./logs/',
            best_model_save_path='./logs/',
            deterministic=True,
            render=False,
            verbose=1
        )

        # Measure start time
        start_time = time.time()
        
        agent.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
        
        # Measure end time
        end_time = time.time()
        training_time = end_time - start_time
        
        env.close()
        results_path = os.path.join('./logs/', 'evaluations.npz')
        evaluations = np.load(results_path)
        mean_rewards = evaluations['results'].mean(axis=1)

        # Save model
        if not os.path.exists("models"):
            os.makedirs("models")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join("models", f"ppo_multifidelity_agent_{timestamp}")            
        agent.save(model_filename)
        
        # Plot rewards
        GenericSolver.plot_training(mean_rewards, timestamp)

        return training_time - np.mean(mean_rewards)

