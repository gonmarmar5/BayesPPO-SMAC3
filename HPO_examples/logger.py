import logging
import sys


class CartpoleLogger:
    def __init__(self, log_file='cartpole_optimization.log'):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Crear un manejador de archivo
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        
        # Crear un formatter y establecerlo para el manejador de archivo
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        
        # Añadir el manejador de archivo al logger
        self.logger.addHandler(self.file_handler)
        
    def redirect_stdout_to_log_file(self, log_file):
        self.original_stdout = sys.stdout
        self.log_file = open(log_file, 'a')
        sys.stdout = self.log_file
        
    def restore_stdout(self):
        sys.stdout = self.original_stdout
        self.log_file.close()
        
    def log_optimization(self, smac):
        # Redirigir sys.stdout temporalmente a un archivo
        self.redirect_stdout_to_log_file('cartpole_optimization.log')
        
        # Ejecutar la función optimize() con las salidas redirigidas al archivo de registro
        incumbent = smac.optimize()

        # Restaurar sys.stdout a su comportamiento original
        self.restore_stdout()
        
        return incumbent
    
    def log_results(self, logger, incumbent_config, incumbent_cost):
        # Redirigir sys.stdout temporalmente a un archivo
        self.redirect_stdout_to_log_file('cartpole_optimization.log')

        # Imprimir los resultados en la consola y también en el archivo de registro
        print(f"Incumbent configuration: {incumbent_config}")
        print(f"Incumbent cost: {incumbent_cost}")
        logger.info(f"Incumbent configuration: {incumbent_config}")
        logger.info(f"Incumbent cost: {incumbent_cost}")

        # Restaurar sys.stdout a su comportamiento original
        self.restore_stdout()
