import yaml

with open("config.yaml") as F:
    data = yaml.safe_load(F)


class Config:
    def __init__(self, raw_data):
        self.problems_parent_path = raw_data["problems_parent_path"]


config = Config(data)
