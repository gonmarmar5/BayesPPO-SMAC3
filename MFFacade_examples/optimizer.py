import datetime
import os
import shutil
import threading

import gymnasium
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
    model_files = [f for f in os.listdir(models_folder) if f.startswith('ppo_multifidelity_agent')]

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

    env.close()

    print(best_model_file)
    print(best_mean_reward)

    return os.path.join(models_folder, model_file)

if __name__ == "__main__":
    
    if ENV == 'CartPole':
        filename = "logs/cartpole_optimizer.log"
    else: 
        filename = "logs/lunarlander_optimizer.log"
    
    model = genericSolver.GenericSolver()

    logger = Logger(filename)

    # n_trials determines the maximum number of different hyperparameter configurations SMAC will evaluate during its search for the optimal setup.
    # If deterministic is set to true, only one seed is passed to the target function. Otherwise, multiple seeds are passed to ensure generalization.
    scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        seed=-1,  
                        n_trials=100,
                        walltime_limit=EARLY_STOPPING,
                        min_budget=MIN_BUDGET,
                        max_budget=MAX_BUDGET, # Establece el número de configuraciones iniciales deseado
                        #n_workers=8, #The number of workers to use for parallelization
                        )  

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario=scenario, n_configs=5)
    
    '''
        Successive Halving is a classic method used in hyperparameter optimization that gradually increases the budget for configurations that perform well.
        It starts by allocating a smaller budget to all configurations and iteratively promotes configurations that show promise to higher budgets.
    '''

    '''
        Hyperband is an adaptive method that explores a large number of configurations early on with smaller budgets and dynamically prunes poorly performing configurations.
        It aims to maximize resource allocation efficiency by quickly discarding underperforming configurations.
    '''
    
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget", instance_seed_order="shuffle_once")

    smac = MFFacade(scenario=scenario,
                    target_function=model.train, 
                    initial_design=initial_design, 
                    intensifier=intensifier,
                    overwrite=True
                    )

    incumbent = logger.log_optimization(smac)
    
    incumbent_config = dict(incumbent)

    print("Incumbent configuration: ", incumbent)
    
    # Calculate the cost of the incumbent and save it in the log file
    #logger.log_results(smac, incumbent, incumbent_config)
    
     # Validation of the trained agent
    best_model_dir = agents_validation()
    #best_model_dir = "./models/ppo_multifidelity_agent_20240629_230530"

    # Optionally we can render the agent to check its performance
    render_agent(best_model_dir)

    models_folder = "models"
    #shutil.rmtree(models_folder)
