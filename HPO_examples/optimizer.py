import datetime
import os

import gymnasium
import numpy as np
from stable_baselines3 import PPO

import genericSolver

from smac.scenario import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from logger import Logger 
import threading
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

ENV = genericSolver.ENV
EARLY_STOPPING = 180

def render_agent(best_model_dir):
    """
    Renders the agent's behavior within the specified environment for a single episode.

    Args:
        agent: The trained agent to be evaluated.
        env: The CartPole environment instance.
    """
    if ENV == 'CartPole':
        env = gymnasium.make("CartPole-v1", render_mode = "human")
    else:
        env = gymnasium.make("LunarLander-v2", render_mode = "human")

    agent = PPO.load(best_model_dir, env = env)
    observation, info = env.reset()

    terminated = False
    for i in range(100000):
        env.render()
        action, _ = agent.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        # Reset the sim everytime the lander makes contact with the surface of the moon
        if terminated or truncated:
            observation, info = env.reset()
    env.close()

def agents_validation():
    
    models_folder = "models"
    if ENV == 'CartPole':
        env = Monitor(gymnasium.make('CartPole-v1'))
    else:
        env = Monitor(gymnasium.make('LunarLander-v2'))

    # Obtener la lista de archivos de modelos en la carpeta
    model_files = [f for f in os.listdir(models_folder) if f.startswith('ppo_basic_agent')]

    best_mean_reward = -float('inf')
    best_model_file = None

    for model_file in model_files:
        # Cargar el modelo desde el archivo
        trained_agent = PPO.load(os.path.join(models_folder, model_file))

        # Evaluar el modelo en el entorno
        rewards, _ = evaluate_policy(trained_agent, env, n_eval_episodes=30, deterministic=True)

        mean_reward = np.mean(rewards)
        print(f"Modelo {model_file}: Mean Reward = {mean_reward}")

        # Actualizar el mejor modelo si se encuentra uno con mejor rendimiento
        if mean_reward >= best_mean_reward:
            best_mean_reward = mean_reward
            best_model_file = model_file
        
    print(best_model_file)
    print(best_mean_reward)

    return os.path.join(models_folder, model_file)

if __name__ == "__main__":
    if ENV == 'CartPole':
        filename = "logs/cartpole_optimizer.log"
    else: 
        filename = "logs/lunarlander_optimizer.log"

    logger = Logger(filename)
    
    model = genericSolver.GenericSolver()
    
    scenario = Scenario(model.configspace, 
                        deterministic=True,     # If deterministic is set to true, only one seed is passed to the target function. Otherwise, multiple seeds are passed to ensure generalization.
                        seed=-1,
                        n_trials=100,   # n_trials determines the maximum number of different hyperparameter configurations SMAC will evaluate during its search for the optimal setup.
                        walltime_limit=EARLY_STOPPING) 

    smac = HPOFacade(scenario=scenario, 
                     target_function=model.train, 
                     overwrite=True)

    incumbent = logger.log_optimization(smac)
    
    incumbent_config = dict(incumbent)

    print("Incumbent configuration: ", incumbent)
    
    # Calculate the cost of the incumbent and save it in the log file
    #logger.log_results(smac, incumbent, incumbent_config)
    
     # Validation of the trained agent
    best_model_dir = agents_validation()

    # Optionally we can render the agent to check its performance
    render_agent(best_model_dir)

    models_folder = "models"
    #shutil.rmtree(models_folder)

