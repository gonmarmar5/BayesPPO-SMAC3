import datetime
import os
import shutil
import threading

import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from smac.scenario import Scenario
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from logger import Logger 
import genericSolver 

ENV = genericSolver.ENV
MIN_BUDGET = genericSolver.MIN_BUDGET
MAX_BUDGET = genericSolver.MAX_BUDGET
EARLY_STOPPING = genericSolver.EARLY_STOPPING

def render_agent(best_model_dir, num_episodes = 10):
    """
    Renders the behavior of the agent in the specified environment for multiple episodes.

    Args:
    - best_model_dir (str): Directory path where the trained agent model is saved.
    - num_episodes (int): Number of episodes to render (default is 10).
    """
    if ENV == 'CartPole':
        env = gymnasium.make("CartPole-v1", render_mode = "human")
    else:
        env = gymnasium.make("LunarLander-v2", render_mode = "human")

    agent = PPO.load(best_model_dir, env = env)
    
    vec_env = agent.get_env()
    observation, info = vec_env.reset()
    for episode in range(num_episodes):
        observation, _ = vec_env.reset()
        terminated = False
        while not terminated:
            vec_env.render("human")
            action, _ = agent.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = vec_env.step(action)
            if terminated or truncated:
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
    model_files = [f for f in os.listdir(models_folder) if f.startswith('ppo_multifidelity_agent')]

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
        plot_filename = os.path.join("plots", f"validation_rewards_{best_model_file}.png")
        plt.savefig(plot_filename)
        plt.close()

    return os.path.join(models_folder, model_file)

if __name__ == "__main__":
    
    if ENV == 'CartPole':
        log_filename = "logs/cartpole_optimizer.log"
    else: 
        log_filename = "logs/lunarlander_optimizer.log"
    
    model = genericSolver.GenericSolver()

    logger = Logger(log_filename)

    scenario = Scenario(model.configspace, 
                        deterministic=True,  # If deterministic is set to true, only one seed is passed to the target function. 
                        seed=-1,  
                        n_trials=100,   # Maximum number of different hyperparameter configurations SMAC will evaluate
                        walltime_limit=EARLY_STOPPING,
                        min_budget=MIN_BUDGET,
                        max_budget=MAX_BUDGET
                        )  

    # important to efficiently start hyperparameter optimization.
    initial_design = MFFacade.get_initial_design(scenario=scenario, n_configs=5)
    
    #Hyperband explores a large number of configurations with small budgets and dynamically prunes poorly performing configurations.
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget", instance_seed_order="shuffle_once")

    smac = MFFacade(scenario=scenario,
                    target_function=model.train, 
                    initial_design=initial_design, 
                    intensifier=intensifier,
                    overwrite=True
                    )

    incumbent = logger.log_optimization(smac)
    
    print("Incumbent configuration: ", incumbent)
    
    # Validation of the trained agent
    best_model_dir = agents_validation()
    # best_model_dir = "./models/ppo_multifidelity_agent_20240630_030407"

    # Optionally we can render the agent to check its performance
    render_agent(best_model_dir)

    models_folder = "models"
    shutil.rmtree(models_folder)
