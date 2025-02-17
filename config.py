import yaml

with open("config.yaml") as F:
    data = yaml.safe_load(F)


class Config:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.problems_parent_path = raw_data["problems_parent_path"]
        self.problems_file_extension = raw_data["problems_file_extension"]
        self.ant_system = raw_data["algorithms"]["ant_system"]
        self.mmas = raw_data["algorithms"]["min_max_ant_system"]
        self.random_number_seed = raw_data["random_number_seed"]
        self.q_learning = raw_data["algorithms"]["q_learning"]
        self.double_q_learning = raw_data["algorithms"]["double_q_learning"]
        self.mlflow_uri = raw_data["mlflow_uri"]

    @property
    def debug(self):
        raw_value = self.raw_data.get("debug", False)
        if isinstance(raw_value, bool):
            return raw_value
        elif isinstance(raw_value, str):
            return raw_value.casefold() == "true"
        raise ValueError("Invalid value specified for debug")


config = Config(data)
