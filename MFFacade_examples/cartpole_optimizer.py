import datetime
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, Integer
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from smac import MultiFidelityFacade as MFFacade
from logger import Logger 

class CartpoleFunction:
    @property
    def configspace(self) -> ConfigurationSpace:
        """
        Defines the hyperparameter search space for optimizing a PPO agent on the CartPole environment. 
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

        fidelity = Integer('fidelity', (1,3), default=3)  # 1: Low, 2: Medium, 3: High
        cs.add_hyperparameter(fidelity)

        return cs
    
    def evaluate_agent(agent, env):
        """
        Evaluates the performance of an agent within the CartPole environment over a single episode.

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

        return total_reward
    
    def plot_rewards(rewards):
        """
        Generates a plot visualizing the evolution of rewards for multiple agents, 
        along with their mean, and saves the plot to the specified file.

        Args:
            rewards (dict): A dictionary containing reward histories. Keys represent agent identifiers 
                            and the 'mean_reward' key holds the average across agents.
            filename (str): The name of the file to save the plot to (default: 'rewards_plot.png').
        """
        # Plot individual rewards for each agent
        for agent_key, agent_rewards in rewards.items():
            if agent_key != 'mean_reward':  # Exclude the mean_reward key from plotting individual rewards
                plt.plot(agent_rewards, label=f'Agent {agent_key}', alpha=0.5)  # Use alpha to make individual lines lighter

        # Plot mean rewards
        mean_rewards = rewards['mean_reward']
        plt.plot(mean_rewards, label='Mean Reward', linewidth=2, color='black')  # Make the line wider and black

        plt.xlabel('Training Updates')
        plt.ylabel('Individual Reward')
        plt.title('PPO Individual Rewards Progress')
        plt.legend()

        filename = "plots/" + datetime.datetime.now().strftime("%m-%d %H:%M:%S") + "_cartpole_rewards.png"
        plt.savefig(filename)
        plt.show()

    def train(self, config: Configuration, seed: int = 0, budget: int = 500, instance=None) -> float:
        """
        Trains a Proximal Policy Optimization (PPO) agent in the CartPole environment, tracks performance, and calculates a metric for BO optimization.

        Args:
            config (Configuration): A configuration object representing hyperparameters for PPO.
            seed (int): Random seed for reproducibility (default: 0).
            budget (int): The maximum number of steps allowed in an episode, controlling the fidelity level (default: 500).

        Returns:
            float: The negative mean average reward achieved over the training process. 
                    This is used by BO to prioritize configurations that achieve  higher rewards (lower negative values).
        """
        
        env = gymnasium.make('CartPole-v1')

        ppo_params = {
            'policy': 'MlpPolicy', 
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

        total_timesteps = 50000 
        batch_size = 1024
        num_updates = total_timesteps // batch_size

        rewards = {}  

        # Determine max steps based on fidelity budget
        max_episode_steps = 500  # Default for high fidelity
        if budget == 1:
            max_episode_steps = 100
        elif budget == 2:
            max_episode_steps = 250

        # Agent training Loop
        for update in range(1, num_updates + 1):  
            agent.learn(total_timesteps = batch_size)

            total_reward = 0
            for num_agent in range(5):
                individual_reward = CartpoleFunction.evaluate_agent(agent, env)
                agent_key = str(num_agent + 1)
                if agent_key in rewards:
                    rewards[agent_key].append(individual_reward)
                else:
                    rewards[agent_key] = [individual_reward]
                total_reward += individual_reward
            mean_reward = total_reward / 5
            
            if 'mean_reward' in rewards:
                rewards['mean_reward'].append(mean_reward)
            else:
                rewards['mean_reward'] = [mean_reward]

            # Early termination based on budget (fidelity)
            for name, info in env.info.items():
                if 'terminated' in name or 'truncated' in name or 'time_limit_reached' in name and info[name]:
                    if info[name] or  info['steps'] >= max_episode_steps:
                        break 

        env.close()
        CartpoleFunction.plot_rewards(rewards)        

        return -np.mean(rewards['mean_reward']) # Optimize for minimizing (max reward as BO will focus on lower losses)

if __name__ == "__main__":

    filename = "logs/" + datetime.datetime.now().strftime("%m-%d %H:%M:%S") + "_cartpole_optimizer.log"
    logger = Logger(filename)
    
    model = CartpoleFunction()

    min_budget = 100 
    max_budget = 500  
    n_workers = 8  # Use available cores

    scenario = Scenario(
        model.configspace,
        min_budget=min_budget,
        max_budget=max_budget,
        n_workers=n_workers
    )

    intensifier = Hyperband(scenario, incumbent_selection="highest_budget")

    smac = MFFacade(
            scenario=scenario,
            target_function=model.train,
            intensifier=intensifier,
            overwrite=True
        )
    incumbent = logger.log_optimization(smac)

    incumbent_config = dict(incumbent)
    
    # Calculate the cost of the incumbent and save it in the log file
    logger.log_results(smac, incumbent, incumbent_config)
