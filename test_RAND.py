import random
import sys
import yaml
import pandas as pd
from tqdm import tqdm

sys.path.append("./util/")
from evaluation_gem5 import evaluation_gem5
from config_analyzer import config_self,config_self_new

##initial evaluation
default_state = {
    "core": 3,
    "l1i_size": 256,
    "l1d_size": 256,
    "l2_size": 64,
    "l1d_assoc": 8,
    "l1i_assoc": 8,
    "l2_assoc": 8,
    "sys_clock": 2,
}
evaluation = evaluation_gem5(default_state)


with open('util/config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

config = config_self_new(config_data)
constraints = config.constraints

def random_search():
    epochs = 400
    # 定义每个变量的取值范围
    design_dim_ranges = {
        'core': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'l1i_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        'l1d_size': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        'l2_size': [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        'l1i_assoc': [1, 2, 4, 8, 16],
        'l1d_assoc': [1, 2, 4, 8, 16],
        'l2_assoc': [1, 2, 4, 8, 16],
        'sys_clock': [2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
    }

    reward_array ,design_point_array, metric_array = [], [], []
    pbar = tqdm(range(epochs), desc="Searching", unit="iteration", leave=False)
    for _ in pbar:
        status = {}
        for dim, values in design_dim_ranges.items():
            status[dim] = random.choice(values)
        status_val = status.values()
        metrics = evaluation.eval(status_val)
        if metrics != None:
            energy = metrics["latency"]
            area = metrics["Area"]
            runtime = metrics["latency"]
            power = metrics["power"]

            design_point_array.append(status_val)
            metric_array.append(metrics.values())
            constraints.update({"AREA": area,"POWER":power})
            reward = 1000 / (runtime * 100000 * constraints.get_punishment())
            reward_array.append(reward)
    
    print("search done")

    reward_array = pd.DataFrame(reward_array,columns=["reward"])
    obs_array = pd.DataFrame(design_point_array,columns=["core","l1i_size","l1d_size","l2_size","l1i_assoc","l1d_assoc","l2_assoc","clock_rate"])
    metric_array = pd.DataFrame(metric_array,columns=['latency','Area','energy','power'])
    result_df = pd.concat([reward_array,obs_array,metric_array],axis=1)
    result_df.to_csv(f"./data/{config.benchmark}_{config.target}_RANDOM_DATA.csv")


if __name__ == "__main__":
    random_search()
