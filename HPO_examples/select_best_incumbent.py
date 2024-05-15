import re
from genericSolver import GenericSolver

from smac.scenario import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter
import sys
import io

ENV = 'CartPole'
#ENV = 'LunarLander'

def extract_training_info(log_file):
    with open(log_file, 'r') as file:
        log = file.read()
    

    # Define the configuration space
    cs = ConfigurationSpace()

    # Add hyperparameters to the configuration space
    discount_factor = UniformFloatHyperparameter("discount_factor", 0, 1)
    gae_lambda = UniformFloatHyperparameter("gae_lambda", 0, 1)
    learning_rate = UniformFloatHyperparameter("learning_rate", 0, 0.01)

    cs.add_hyperparameters([discount_factor, gae_lambda, learning_rate])
    
    discount_factors = re.findall("'discount_factor': ([0-9.]+)", log)
    gae_lambdas = re.findall("'gae_lambda': ([0-9.]+)", log)
    learning_rates = re.findall("'learning_rate': ([0-9.e-]+)", log)

    discount_factor_list = [float(x) for x in discount_factors]
    gae_lambda_list =  [float(x) for x in gae_lambdas]
    learning_rate_list = [float(x) for x in learning_rates]

    incumbents = []

    for i in range(len(discount_factor_list)):
        config_dict = {
        'discount_factor': discount_factor_list[i],
        'gae_lambda': gae_lambda_list[i],
        'learning_rate': learning_rate_list[i],
        }
        incumbent = Configuration(cs, values=config_dict)
        incumbents.append(incumbent)

    return incumbents

if __name__ == "__main__":
    if ENV == 'CartPole':
        log_file = "./logs/cartpole_optimizer.log"
    else:
        log_file = "./logs/lunarlander_optimizer.log"

    print("Starting the evaluation of the best incumbent...")
    incumbents = extract_training_info(log_file)

    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    incumbent_dict = {}
    for incumbent in incumbents:
        incumbent_cost = GenericSolver.evaluate(incumbent)
        incumbent_dict[incumbent] = incumbent_cost
    
    sys.stdout = original_stdout

    best_incumbent = max(incumbent_dict, key=incumbent_dict.get)

    print("Mejor incumbent: ", best_incumbent, ". Reward obtenido: ", incumbent_dict[best_incumbent])
        
        
