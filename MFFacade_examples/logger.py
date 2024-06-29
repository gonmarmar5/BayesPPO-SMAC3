import datetime
import logging
import sys

class Logger:
    def __init__(self, log_file):
        """
        Initializes the Logger class, configures logging format, and creates a custom logger.

        Args:
            log_file (str): The filename for storing log messages.
        """

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
        """ 
        Temporarily redirects standard output (sys.stdout) to the specified log file.

        Args:
            log_file: (str) The filename where standard output will be written.
        """

        self.original_stdout = sys.stdout
        self.log_file = open(log_file, 'a')
        sys.stdout = self.log_file
        
    def restore_stdout(self):
        """ 
        Restores standard output (sys.stdout) to its original state. 
        """

        sys.stdout = self.original_stdout
        self.log_file.close()
        
    def log_optimization(self, smac):
        """
        Wraps the SMAC optimization process, capturing its standard output in the log file. 

        Args:
            smac: An instance of the SMAC optimizer.

        Returns:
            incumbent: The best configuration found during the optimization process.
        """

        # Temporarily redirect sys.stdout to a file
        self.redirect_stdout_to_log_file(self.log_file)

        start_training_time = datetime.datetime.now()
        print("Start training time: " + start_training_time.strftime("%Y-%m-%d %H:%M:%S"))

        print(smac)
        incumbent = smac.optimize()
        end_training_time = datetime.datetime.now()
        training_duration = end_training_time - start_training_time 
        print("End training time: " + end_training_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("Training time: " + str(training_duration))
        
        # Restore sys.stdout to its original behavior
        self.restore_stdout()
        
        return incumbent
    
    def log_results(self, smac, incumbent, incumbent_config):
        """
        Calculates the validation cost of the best configuration (incumbent) found by SMAC 
        and logs the results to the specified log file.

        Args:
            smac: An instance of the SMAC optimizer.
            incumbent: The best configuration found during optimization.
            incumbent_config: The configuration parameters of the incumbent solution.
        """

        # Temporarily redirect sys.stdout to a file
        self.redirect_stdout_to_log_file(self.log_file.name)

        # Calculate the incumbent's cost and record the output in the log.
        print("############# Validation")
        incumbent_cost = smac.validate(incumbent)

        self.logger.info(f"Incumbent configuration: {incumbent_config}")
        self.logger.info(f"Incumbent cost: {incumbent_cost}")

        # Restore sys.stdout to its original behavior
        self.restore_stdout()
