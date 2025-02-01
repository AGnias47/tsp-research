import yaml

with open("config.yaml") as F:
    data = yaml.safe_load(F)


class Config:
    def __init__(self, raw_data):
        self.problems_parent_path = raw_data["problems_parent_path"]
        self.problems = raw_data["problems"]
        try:
            self.ant_system_iterations = int(
                raw_data["algorithms"]["ant_system"]["iterations"]
            )
        except (KeyError, ValueError):
            self.ant_system_iterations = None


config = Config(data)
