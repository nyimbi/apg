import yaml

class ConfigManager:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = {}

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_network_config(self):
        return self.config['network']

    def get_routing_table(self):
        return self.config['routing']

    def get_database_config(self):
        return self.config['database']