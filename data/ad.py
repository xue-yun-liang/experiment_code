import pandas as pd

benchmark = 'cannel'
target = 'normal'
algo = 'erdse'

metric_path = f'{target}/{benchmark}/{algo}_metric.csv'
obs_path = f'{target}/{benchmark}/{algo}_obs.csv'
reward_path = f'{target}/{benchmark}/{algo}_reward.csv'

metric = pd.read_csv(metric_path)
obs = pd.read_csv(obs_path)
reward = pd.read_csv(reward_path)
metric.columns = ['latency','Area','energy','power']
obs.columns = ['core','l1i_size','l1d_size','l2_size','l1i_assoc','l1d_assoc','l2_assoc','clock_rate']
reward.columns = ['reward']

result_df = pd.concat([reward,obs,metric],axis=1)
result_df.to_csv(f"canneal_{target}_{algo}.csv")