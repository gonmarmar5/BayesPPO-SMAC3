import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter
from smac.scenario import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from logger import Logger 

# Optimizacion bayesiana tradicional

class LunarLanderFunction:
    @property
    def configspace(self) -> ConfigurationSpace:
        '''
        Configure the hyperparameters for the PPO agent.
        '''
        cs = ConfigurationSpace(seed=0)
        learning_rate = UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=1e-2, default_value=1e-3, log=True)
        discount_factor = UniformFloatHyperparameter("discount_factor", lower=0.9, upper=0.999, default_value=0.99)
        gae_lambda = UniformFloatHyperparameter("gae_lambda", lower=0.8, upper=0.999, default_value=0.95)  
        cs.add_hyperparameters([learning_rate, discount_factor, gae_lambda])

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
        env = gymnasium.make('LunarLander-v2')

        # Configurar los parámetros del algoritmo PPO
        ppo_params = {
            'policy': 'MlpPolicy', # indicates that the policy will be represented by a feedforward neural network
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

        # Crear el agente PPO
        agent = PPO(**ppo_params)

        # Entrenar el agente
        agent.learn(total_timesteps=13000)  # Pendiente de revisión este valor

        # Evaluar el agente
        mean_reward = np.mean([LunarLanderFunction.evaluate_agent(agent, env) for _ in range(5)])

        # Cerrar el entorno
        env.close()

        return -mean_reward  # SMAC busca minimizar

if __name__ == "__main__":

    logger = Logger('logs/lunarlander_optimizer.log')
    
    model = LunarLanderFunction()

    scenario = Scenario(model.configspace, deterministic=True, n_trials=1)

    # Crear la instancia HPOFacade
    smac = HPOFacade(scenario=scenario, target_function=model.train, overwrite=True)

    # Ejecutar la optimización pasando la función objetivo directamente
    incumbent = logger.log_optimization(smac)

    # Retrieve the incumbent configuration
    incumbent_config = dict(incumbent)
    
    # Let's calculate the cost of the incumbent and save it in the log file
    logger.log_results(smac, incumbent, incumbent_config)
