import datetime

from genericSolver import GenericSolver

from smac.scenario import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from logger import Logger 
import threading

ENV = 'CartPole'
#ENV = 'LunarLander'

def optimize_and_log():
    
    if ENV == 'CartPole':
        filename = "logs/cartpole_optimizer.log"
    else: 
        filename = "logs/lunarlander_optimizer.log"

    logger = Logger(filename)
    
    model = GenericSolver()

    # n_trials determines the maximum number of different hyperparameter configurations SMAC will evaluate during its search for the optimal setup.
    # If deterministic is set to true, only one seed is passed to the target function. Otherwise, multiple seeds are passed to ensure generalization.
    scenario = Scenario(model.configspace, deterministic=True, seed=-1, n_trials=10) 

    smac = HPOFacade(scenario=scenario, target_function=model.train, overwrite=True)

    incumbent = logger.log_optimization(smac)

    incumbent_config = dict(incumbent)

    # Calculate the cost of the incumbent and save it in the log file
    logger.log_results(smac, incumbent, incumbent_config)

def run_optimization_thread():
    optimization_thread = threading.Thread(target=optimize_and_log)
    optimization_thread.start()
    return optimization_thread

if __name__ == "__main__":
    optimization_thread = run_optimization_thread()
    optimization_thread.join()