import pandas as pd
import numpy as np
import os.path


# file paths
all_social_vars_path = os.path.expanduser('~/alspac/social/workspace/all_social.txt')
chosen_social_vars_path = os.path.expanduser('~/alspac/social/workspace/chosen_social.txt')
save_path = os.path.expanduser('~/alspac/social/workspace/variables.txt')

# read to dataframes
all_df = pd.read_csv(all_social_vars_path, sep='\t')
df = pd.read_csv(chosen_social_vars_path, sep='\t')

# create column indicating timepoints with multiple similar observations (e.g. bullied, teased, etc)
df['is_multiple'] = (df.Questionnaire.str.len() > 2)

# rename Questionnaire names
df.Questionnaire = df.Questionnaire.str.slice(0, 2)

# join tables to form final dataframe
df = df.merge(all_df, on=['Questionnaire', 'Details'], how='left')

# save result
df.to_csv(save_path, sep=',', index=False)


