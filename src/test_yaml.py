import yaml
from pathlib import Path
import itertools

configs = Path('configs')
config_file = configs/'DeepAR.yaml'
 
with open(config_file) as f:
    config = yaml.safe_load(f)


config_dict = {}
for key in config:
    for k, v in config[key].items():
        config_dict[k] = v

config_dict

config_dict.values()
params = []
for key, val in config_dict.items():
    params.append(val)

itertools.product(*config_dict.values())
sorted(config_dict)
param_key = list(config_dict.keys())

combinations = list(itertools.product(*(config_dict[Name] for Name in param_key)))
len(combinations)