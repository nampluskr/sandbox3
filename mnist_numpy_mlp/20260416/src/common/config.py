import os
from dotenv import load_dotenv
import yaml


def load_config(config_dir, config_file):
    with open(os.path.join(config_dir, config_file), "r") as f:
        config = yaml.safe_load(f)

    load_dotenv()
    config["dataset_dir"] = os.environ["DATASET_DIR"]
    config["backbone_dir"] = os.environ["BACKBONE_DIR"]
    return config
