import pandas as pd
import json

benchmark = 'canneal'
target = 'normal'

crldse_df = pd.read_csv(f'{benchmark}_{target}_crldse.csv')
erdse_df = pd.read_csv(f'{benchmark}_{target}_erdse.csv')
momprdse_df = pd.read_csv(f'{benchmark}_{target}_momprdse.csv')
acdse_df = pd.read_csv(f'{benchmark}_{target}_acdse.csv')
nsga2_df = pd.read_csv(f'{benchmark}_{target}_nsga2.csv')
mopso_df = pd.read_csv(f'{benchmark}_{target}_mopso.csv')
sac_df = pd.read_csv(f'{benchmark}_{target}_sac.csv')
ppo_df = pd.read_csv(f'{benchmark}_{target}_ppo.csv')


crldse_reward = crldse_df['reward']
erdse_reward = erdse_df['reward']
momprdse_reward = momprdse_df['reward1']
acdse_reward = acdse_df['reward']
nsga2_reward = nsga2_df['reward']
mopso_reward = mopso_df['reward']
sac_reward = sac_df['reward']
ppo_reward = ppo_df['reward']

crldse_max_r = crldse_df['reward'].max()
erdse_max_r = erdse_df['reward'].max()
momprdse_max_r = momprdse_df['reward1'].max()
acdse_max_r = acdse_df['reward'].max()
nsga2_max_r = nsga2_df['reward'].max()
mopso_max_r = mopso_df['reward'].max()
sac_max_r = sac_df['reward'].max()
ppo_max_r = ppo_df['reward'].max()


crldse_max_indices = crldse_df[crldse_df['reward'] == crldse_max_r].index
erdse_max_indices = erdse_df[erdse_df['reward'] == erdse_max_r].index
momprdse_max_indices = momprdse_df[momprdse_df['reward1'] == momprdse_max_r].index
acdse_max_indices = acdse_df[acdse_df['reward'] == acdse_max_r].index
nsga2_max_indices = nsga2_df[nsga2_df['reward'] == nsga2_max_r].index
mopso_max_indices = mopso_df[mopso_df['reward'] == mopso_max_r].index
sac_max_indices = sac_df[sac_df['reward'] == sac_max_r].index
ppo_max_indices = ppo_df[ppo_df['reward'] == ppo_max_r].index

max_index_dict = {
    'crldse': crldse_max_indices.tolist(),
    'erdse': erdse_max_indices.tolist(),
    'momprdse': momprdse_max_indices.tolist(),
    'acdse': acdse_max_indices.tolist(),
    'nsga2': nsga2_max_indices.tolist(),
    'mopso': mopso_max_indices.tolist(),
    'sac': sac_max_indices.tolist(),
    'ppo': ppo_max_indices.tolist()
}

print("crldse最大值的索引位置:", crldse_max_indices)
print("erdse最大值的索引位置:", erdse_max_indices)
print("momprdse最大值的索引位置:", momprdse_max_indices)
print("acdse最大值的索引位置:", acdse_max_indices)
print("nsga2最大值的索引位置:", nsga2_max_indices)
print("mopso最大值的索引位置:", mopso_max_indices)
print("sac最大值的索引位置:", sac_max_indices)
print("ppo最大值的索引位置:", ppo_max_indices)

with open('max_index_data.json', 'w') as f:
    json.dump(max_index_dict, f, indent=4)