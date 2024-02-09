import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter
from smac.scenario import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade

import sys
import logging

# Configurar el logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='optimization.log',
    filemode='w'
)

# Crear un custom logger
logger = logging.getLogger()

# Crear un manejador de archivo
file_handler = logging.FileHandler('optimization.log')
file_handler.setLevel(logging.INFO)

# Crear un formatter y establecerlo para el manejador de archivo
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Añadir el manejador de archivo al logger
logger.addHandler(file_handler)

class CartpoleFunction:
    @property
    def configspace(self) -> ConfigurationSpace:
        '''
        Configure the hyperparameters for the PPO agent.
        '''
        cs = ConfigurationSpace(seed=0)
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=1e-2, default_value=1e-3, log=True)
        discount_factor = UniformFloatHyperparameter("discount_factor", lower=0.9, upper=0.999, default_value=0.99)
        max_timesteps = UniformFloatHyperparameter("max_timesteps", lower=1000, upper=10000, default_value=5000)
        cs.add_hyperparameters([learning_rate, discount_factor, max_timesteps])

        return cs
    
    def evaluate_agent(agent, env):
        '''
        Evaluate the agent in the given environment and return the total reward.
        '''
        obs, info = env.reset() # observation is the state of the environment
        terminated = False
        total_reward = 0
        while not terminated:
            action = agent.predict(obs)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        return total_reward

    def train(self, config: Configuration, seed: int = 0) -> float:
        '''
        Train the PPO agent with the given hyperparameters and return the average reward.
        '''
        env = gymnasium.make('CartPole-v1')

        # Configurar los parámetros del algoritmo PPO
        ppo_params = {
            'policy': 'MlpPolicy', # indicates that the policy will be represented by a feedforward neural network
            'env': env,
            'learning_rate': config['learning_rate'],
            'gamma': config['discount_factor'],
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1
        }

        # Crear el agente PPO
        agent = PPO(**ppo_params)

        # Entrenar el agente
        agent.learn(total_timesteps=int(config['max_timesteps']))

        # Evaluar el agente
        mean_reward = np.mean([CartpoleFunction.evaluate_agent(agent, env) for _ in range(5)])

        # Cerrar el entorno
        env.close()

        return -mean_reward  # SMAC busca minimizar

if __name__ == "__main__":

    def optimize_with_logging(smac):
        # Redirigir sys.stdout temporalmente a un archivo
        original_stdout = sys.stdout
        log_file = open('optimize_log.log', 'a')  # Abrir en modo append (añadir a un archivo existente)
        sys.stdout = log_file

        # Ejecutar la función optimize() con las salidas redirigidas al archivo de registro
        incumbent = smac.optimize()

        # Restaurar sys.stdout a su comportamiento original
        sys.stdout = original_stdout
        log_file.close()  # Cerrar el archivo

        return incumbent

    def log_results(logger, incumbent_config, incumbent_cost):
        # Redirigir sys.stdout temporalmente a un archivo
        original_stdout = sys.stdout
        log_file = open('optimize_log.log', 'a')  # Abrir en modo append (añadir a un archivo existente)
        sys.stdout = log_file

        # Imprimir los resultados en la consola y también en el archivo de registro
        print(f"Incumbent configuration: {incumbent_config}")
        print(f"Incumbent cost: {incumbent_cost}")
        logger.info("==============================")
        logger.info(f"Incumbent configuration: {incumbent_config}")
        logger.info(f"Incumbent cost: {incumbent_cost}")

        # Restaurar sys.stdout a su comportamiento original
        sys.stdout = original_stdout
        log_file.close()  # Cerrar el archivo

    model = CartpoleFunction()

    scenario = Scenario(model.configspace, deterministic=True, n_trials=3)

    # Crear la instancia HPOFacade
    smac = HPOFacade(scenario=scenario, target_function=model.train, overwrite=True)

    # Ejecutar la optimización pasando la función objetivo directamente
    incumbent = optimize_with_logging(smac)

    #plot_learning_rate_variation(smac)
    
    # Retrieve the incumbent configuration
    incumbent_config = dict(incumbent)
    
    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)

    log_results(logger, incumbent_config, incumbent_cost)
