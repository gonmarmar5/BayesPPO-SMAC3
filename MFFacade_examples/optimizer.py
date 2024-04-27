import datetime
import threading

from smac.scenario import Scenario
from smac import MultiFidelityFacade as MFFacade
from logger import Logger 
from genericSolver import GenericSolver

ENV = 'CartPole'
#ENV = 'LunarLander'

def optimize_and_log():
    
    if ENV == 'CartPole':
        filename = "logs/cartpole_optimizer.log"
    else: 
        filename = "logs/lunarlander_optimizer.log"

    logger = Logger(filename)
    
    model = GenericSolver()

    max_budget = 30000
    min_budget = 10000

    # n_trials determines the maximum number of different hyperparameter configurations SMAC will evaluate during its search for the optimal setup.
    # If deterministic is set to true, only one seed is passed to the target function. Otherwise, multiple seeds are passed to ensure generalization.
    scenario = Scenario(model.configspace, 
                        deterministic=True, seed=-1,  n_trials=10,
                        walltime_limit=60, 
                        min_budget=min_budget,
                        max_budget=max_budget,) # Establece el número de configuraciones iniciales deseado
                        #n_workers=8) #The number of workers to use for parallelization

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    intensifier = MFFacade.get_intensifier(
        # Input that controls the proportion of configurations discarded in each round of Successive Halving.
        eta = 3,
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