import pandas as pd
import json

benchmark = 'blackscholes'
target = 'normal'


mopso_df = pd.read_csv(f'{benchmark}_{target}_mopso.csv')

mopso_df['reward'] = mopso_df['reward'].apply(lambda x: 0 if x > 1000 else x)

# 保存修改后的DataFrame回CSV文件（如果需要）
mopso_df.to_csv(f'{benchmark}_{target}_mopso_modified.csv', index=False)
