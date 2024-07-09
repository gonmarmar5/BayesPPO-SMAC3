import datetime
import os

import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO

import genericSolver
import convergence_genericSolver

from smac.scenario import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from logger import Logger 
import threading
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

#ENV = genericSolver.ENV
#EARLY_STOPPING = genericSolver.EARLY_STOPPING
ENV = convergence_genericSolver.ENV
EARLY_STOPPING = convergence_genericSolver.EARLY_STOPPING

def render_agent(best_model_dir, num_episodes = 10):
    """
    Renders the behavior of the agent in the specified environment for multiple episodes.

    Args:
    - best_model_dir (str): Directory path where the trained agent model is saved.
    - num_episodes (int): Number of episodes to render (default is 10).
    """
    if ENV == 'CartPole':
        env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
    else:
        env = gymnasium.make("LunarLander-v2", render_mode="rgb_array")

    agent = PPO.load(best_model_dir, env = env)
    
    vec_env = agent.get_env()
    for episode in range(num_episodes):
        observation = vec_env.reset()
        terminated = False
        while not terminated:
            action, _ = agent.predict(observation, deterministic=True)
            observation, reward, done, info  = vec_env.step(action)
            vec_env.render("human")
            if done:
                break
    env.close()

def agents_validation(models_folder = "models", n_eval_episodes=50):
    """
    Validates multiple PPO agents stored as models in the specified folder.

    Args:
    - models_folder (str): Directory path where trained PPO agent models are saved (default is "models").

    Returns:
    - str: Path to the best performing model's file.
    """

    if ENV == 'CartPole':
        env = Monitor(gymnasium.make('CartPole-v1'))
    else:
        env = Monitor(gymnasium.make('LunarLander-v2'))

    # A list of the models in the folder ./models
    model_files = [f for f in os.listdir(models_folder) if f.startswith('ppo_basic_agent')]

    best_mean_reward = -float('inf')
    best_model_file = None

    for model_file in model_files:
        
        trained_agent = PPO.load(os.path.join(models_folder, model_file))

        rewards, _ = evaluate_policy(trained_agent, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True, deterministic=True)

        mean_reward = np.mean(rewards)
        print(f"Model {model_file}. Mean Reward = {mean_reward}")

        if mean_reward >= best_mean_reward:
            best_mean_reward = mean_reward
            best_model_file = model_file
            best_model_rewards = rewards

    env.close()

    # Plot the best model validation rewards 
    if best_model_rewards is not None:
        if not os.path.exists("plots"):
            os.makedirs("plots")

        plt.figure()
        plt.plot(best_model_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Validation Rewards for Best Model')
        if ENV == 'CartPole':
            plt.ylim(-550, 550)  
        else:
            plt.ylim(-350, 350)
        plot_filename = os.path.join("plots", f"validation_rewards_{best_model_file}.png")
        plt.savefig(plot_filename)
        plt.close()

    return os.path.join(models_folder, best_model_file)

if __name__ == "__main__":
    
    if ENV == 'CartPole':
        filename = "logs/cartpole_optimizer.log"
    else: 
        filename = "logs/lunarlander_optimizer.log"

    logger = Logger(filename)
    
    #model = genericSolver.GenericSolver()
    model = convergence_genericSolver.GenericSolver()
    
    scenario = Scenario(model.configspace, 
                        deterministic=True,    
                        seed=-1,
                        n_trials=100,   
                        walltime_limit=EARLY_STOPPING) 

    smac = HPOFacade(scenario=scenario, 
                     target_function=model.train, 
                     overwrite=True)

    incumbent = logger.log_optimization(smac)
    
    incumbent_config = dict(incumbent)

    print("Incumbent configuration: ", incumbent)
    
    # Validation of the trained agent
    best_model_dir = agents_validation()

    # Optionally we can render the agent to check its performance
    render_agent(best_model_dir)

    models_folder = "models"
    #shutil.rmtree(models_folder)

