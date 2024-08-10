import json
import os

from transformers import BertConfig

import utils as ut


def load_config(model, data, conf_type):
    if model == "bert_pretrained":
        config = BertConfig().from_pretrained("Rostlab/prot_bert")
        return config
    elif model == "bert":
        with open(ut.abspath(f'configs/{data}_configs/{model}_config_default.json')) as f:
            data = json.load(f)
            config = BertConfig(**data)
            return config
    else:
        conf_path = ut.abspath(f'configs/{data}_configs/{model}_config_{conf_type}.json')
        if os.path.exists(conf_path):
            with open(conf_path, 'r') as f:
                config = json.load(f)
                return config
        else:
            raise FileNotFoundError("Config file does not exist")
