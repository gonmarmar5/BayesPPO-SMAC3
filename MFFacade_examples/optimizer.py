import datetime
import threading

from smac.scenario import Scenario
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
from logger import Logger 
from genericSolver import GenericSolver

#ENV = 'CartPole'
ENV = 'LunarLander'

def optimize_and_log():
    
    if ENV == 'CartPole':
        filename = "logs/cartpole_optimizer.log"
    else: 
        filename = "logs/lunarlander_optimizer.log"

    logger = Logger(filename)
    
    model = GenericSolver()

    max_budget = 25000
    min_budget = 7500

    # n_trials determines the maximum number of different hyperparameter configurations SMAC will evaluate during its search for the optimal setup.
    # If deterministic is set to true, only one seed is passed to the target function. Otherwise, multiple seeds are passed to ensure generalization.
    scenario = Scenario(model.configspace, 
                        deterministic=True, seed=-1,  n_trials=100,
                        min_budget=min_budget,
                        max_budget=max_budget, # Establece el número de configuraciones iniciales deseado
                        #n_workers=8, #The number of workers to use for parallelization
                        ) 

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=3)

    '''
        Successive Halving is a classic method used in hyperparameter optimization that gradually increases the budget for configurations that perform well.
        It starts by allocating a smaller budget to all configurations and iteratively promotes configurations that show promise to higher budgets.
    '''

    '''
    Hyperband is an adaptive method that explores a large number of configurations early on with smaller budgets and dynamically prunes poorly performing configurations.
    It aims to maximize resource allocation efficiency by quickly discarding underperforming configurations.
    '''
    intensifier = SuccessiveHalving(
        scenario=scenario,
        incumbent_selection="highest_budget",
    ) 
    smac = MFFacade(scenario=scenario,
                    target_function=model.train, 
                    initial_design=initial_design, 
                    intensifier=intensifier,
                    overwrite=True)

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