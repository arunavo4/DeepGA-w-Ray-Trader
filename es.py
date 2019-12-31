"""
    Evolutionary Strategies
"""

from ray import tune
from es.es import ESTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from lib.env.StockTraderEnv import StockTraderEnv
from lib.model.vision_network import VisionNetwork

tf = try_import_tf()

# register_env("StockTradingEnv", lambda _: StockTradingEnv(10))
ModelCatalog.register_custom_model("NatureCNN", VisionNetwork)

# restore = '/home/skywalker/ray_results/DQN/DQN_StockTradingEnv_2f8c5cc4_2019-11-12_23-08-06i2in8uwu/checkpoint_600',
# resume = True,

tune.run(ESTrainer,
         max_failures=10,
         checkpoint_freq=10,  # iterations
         checkpoint_at_end=True,
         config={"env": StockTraderEnv,
                 "l2_coeff": 0.005,
                 "noise_stdev": 0.02,
                 "episodes_per_batch": 1000,
                 "train_batch_size": 10000,
                 "eval_prob": 0.003,
                 "return_proc_mode": "centered_rank",
                 "stepsize": 0.01,
                 "observation_filter": "MeanStdFilter",
                 "noise_size": 250000000,
                 "report_length": 10,
                 "num_gpus": 0,
                 "num_workers": 6,
                 "model": {
                     "custom_model": "NatureCNN"
                 },
                 "env_config": {
                     "initial_balance": 10000,
                     "day_step_size": 375,  # IN 375 | US 390
                     "look_back_window_size": 375 * 10,  # US 390 * 10 | 375 * 10
                     "enable_env_logging": False,
                     "observation_window": 84,
                     "frame_stack_size": 4,
                     "use_leverage": False,
                     "hold_reward": False,
                     "market": 'in_mkt',  # 'in_mkt' | 'us_mkt'
                 },
                 })  # "eager": True for eager execution
# "num_workers": 4,
