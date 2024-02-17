import logging
import sys

class Logger:
    def __init__(self, log_file):

        self.log_file = log_file

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_file,
            filemode='w'
        )

        # Create a custom logger
        self.logger = logging.getLogger()

        # Create a file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the file handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)
        
    def redirect_stdout_to_log_file(self, log_file):
        self.original_stdout = sys.stdout
        self.log_file = open(log_file, 'a')
        sys.stdout = self.log_file
        
    def restore_stdout(self):
        sys.stdout = self.original_stdout
        self.log_file.close()
        
    def log_optimization(self, smac):
        # Redirigir sys.stdout temporalmente a un archivo
        self.redirect_stdout_to_log_file(self.log_file)
        
        # Ejecutar la función optimize() con las salidas redirigidas al archivo de registro
        incumbent = smac.optimize()

        # Restaurar sys.stdout a su comportamiento original
        self.restore_stdout()
        
        return incumbent
    
    def log_results(self, smac, incumbent, incumbent_config):
        # Redirigir sys.stdout temporalmente a un archivo
        self.redirect_stdout_to_log_file(self.log_file.name)

        print("############# Validation")
        # Calcular el costo del incumbent y registrar la salida en el log
        incumbent_cost = smac.validate(incumbent)

        # Log the results using the logger instance
        self.logger.info(f"Incumbent configuration: {incumbent_config}")
        self.logger.info(f"Incumbent cost: {incumbent_cost}")

        # Restaurar sys.stdout a su comportamiento original
        self.restore_stdout()
