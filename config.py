import yaml

with open("config.yaml") as F:
    data = yaml.safe_load(F)


class Config:
    def __init__(self, raw_data):
        self.problems_parent_path = raw_data["problems_parent_path"]
        self.problems = raw_data["problems"]
        self.ant_system = raw_data["algorithms"]["ant_system"]
        self.mmas = raw_data["algorithms"]["min_max_ant_system"]


config = Config(data)
