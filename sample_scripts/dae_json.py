import sys
sys.path.append('src/')

import runners.dae_train as dae_train
import utils.conf as conf

import os
os.environ["NLU_ENV_CONFIG_PATH"] = "env_configs/env_config.json"

if __name__ == "__main__":
    config_ = conf.Configuration.from_json_arg()
    trainer = dae_train.DAETrainer.from_config(config_)
    trainer.run_train()
