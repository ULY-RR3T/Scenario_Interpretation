import pandas as pd
import numpy as np
from tqdm import tqdm
from .util import *
from .config import *
from collections import defaultdict

def download_data_round(round=11):
    if round == 9:
        df_list = {
            'Ensemble': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/Ensemble/2021-09-11-Ensemble.csv?raw=true',
            'Ensemble_LOP': 'https://raw.githubusercontent.com/midas-network/covid19-scenario-modeling-hub/master/data-processed/Ensemble_LOP/2021-09-11-Ensemble_LOP.csv',
            'Ensemble_LOP_untrimmed': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/Ensemble_LOP_untrimmed/2021-09-11-Ensemble_LOP_untrimmed.csv?raw=true',
            'JHUAPL-Bucky': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/JHUAPL-Bucky/2021-09-13-JHUAPL-Bucky.csv?raw=true',
            'JHU_IDD-CovidSP': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/JHU_IDD-CovidSP/2021-09-14-JHU_IDD-CovidSP.csv?raw=true',
            'MOBS_NEU-GLEAM_COVID': 'https://raw.githubusercontent.com/midas-network/covid19-scenario-modeling-hub/master/data-processed/MOBS_NEU-GLEAM_COVID/2021-09-14-MOBS_NEU-GLEAM_COVID.csv',
        }
    elif round == 11:
        df_list = {
            'USC_SIkJalpha': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/USC-SIkJalpha/2021-12-19-USC-SIkJalpha.csv?raw=true',
            'Ensemble': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/Ensemble/2021-12-18-Ensemble.csv?raw=true',
            'Ensemble_LOP': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/Ensemble_LOP/2021-12-18-Ensemble_LOP.csv?raw=true',
            'Ensemble_LOP_untrimmed': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/Ensemble_LOP_untrimmed/2021-12-18-Ensemble_LOP_untrimmed.csv?raw=true',
            'JHU_IDD-CovidSP': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/JHU_IDD-CovidSP/2021-12-21-JHU_IDD-CovidSP.csv?raw=true',
            'MOBS_NEU-GLEAM_COVID': 'https://github.com/midas-network/covid19-scenario-modeling-hub/blob/master/data-processed/MOBS_NEU-GLEAM_COVID/2021-12-17-MOBS_NEU-GLEAM_COVID.csv?raw=true',
            'NotreDame-FRED': 'https://github.com/midas-network/covid19-scenario-modeling-hub/raw/master/data-processed/NotreDame-FRED/2021-12-21-NotreDame-FRED.csv',
        }
    elif round == 15:
        df_list = {

        }
    else:
        raise Exception("Only rounds 9 and 11 are supported!")
    for k,v in tqdm(df_list.items()):
        out_folder = folder(f"{config['data_dir']}/round{round}/")
        pd.read_csv(v).to_csv(f"{out_folder}/{k}.csv",low_memory=False)

def download_all_data():
    rounds = [9,11]
    for round in rounds:
        download_data_round(round)

def load_data(round):
    models = [path[:-4] for path in os.listdir(f"{config['data_dir']}/round{round}")]
    rslt = {}
    for model in models:
        rslt[model] = pd.read_csv(f"{config['data_dir']}/round{round}/{model}.csv")
    return rslt

def round_metadata(round):
    rslt = json.loads(open('metadata.json').read())
    rslt = rslt[f"round_{round}"]
    rslt['models'] = [path[:-4] for path in os.listdir(f"{config['data_dir']}/round{round}/")]
    for i in range(len(rslt['scenarios'])):
        rslt['scenarios'][i] = rslt['scenarios'][i].split('-')
    return rslt

def extract_data(df, date_str, scenario_name, num_weeks, target_type, inc_cum, q, round_=True, state='US'):
    """

    extract_data(int,int) => pandas.Dataframe

    Keyword arguments:
    scenario_id -> 0 if we are using the first scenario, 1 if we are looking to extract the second scenario
    """
    rslt = []
    scenario = scenario_name + f"-{date_str}"
    for target_wk in range(1,num_weeks+1):
        target = f"{str(target_wk)} wk ahead {inc_cum} {target_type.lower()}"
        rslt_curr = df[(df['scenario_id'] == scenario) & (df['target'] == target)
                       & (df['location'] == state) & (df['quantile'].isin(q))]
        rslt_curr = np.round(rslt_curr['value'],-1)
        if round_:
            rslt_curr = np.round(rslt_curr,-4)
        rslt.append(rslt_curr)
    if len(rslt) == 0:
        return None
    return np.array(rslt)

# def extract_df_list(df_list):
#     df_list_rslt = {}
#     for model,df in tqdm(df_list.items()):
#         curr_model_rslt = {}
#         for index,row in tqdm(df.iterrows()):
#             if row['location'].isdigit():
#                 location_str = str(int(row['location']))
#             else:
#                 location_str = row['location']
#             key = row['scenario_id'] + ' ' + row['target'] + ' ' + location_str
#             if key in curr_model_rslt:
#                 curr_model_rslt.append(row['value'])
#             else:
#                 curr_model_rslt = [row['value']]
#         df_list_rslt[model] = curr_model_rslt
#     return df_list_rslt


def extract_df_list(df_list,q,include_list=None):
    df_list_rslt = {}

    for model, df in df_list.items():
        if include_list:
            if model not in include_list:
                continue
        df['location'] = df['location'].apply(lambda x: str(int(x)) if str(x).isdigit() else x)
        df['key'] = df['scenario_id'] + ' ' + df['target'] + ' ' + df['location']

        curr_model_rslt = defaultdict(list)
        for key, value, quantile in zip(df['key'], df['value'], df['quantile']):
            if quantile in q:
                curr_model_rslt[key].append(value)

        df_list_rslt[model] = curr_model_rslt

    return df_list_rslt

def extract_data_dict(df_dict, date_str, scenario_name, num_weeks, target_type, inc_cum, q, round_=True, state_id='US'):
    rslt = []
    scenario = scenario_name + f"-{date_str}"
    for target_wk in range(1,num_weeks+1):
        target = f"{str(target_wk)} wk ahead {inc_cum} {target_type.lower()}"
        rslt_curr = df_dict[f"{scenario} {target} {state_id}"]
        rslt_curr = np.round(rslt_curr,-1)
        if round_:
            rslt_curr = np.round(rslt_curr,-4)
        rslt.append(rslt_curr)
    return np.array(rslt)
