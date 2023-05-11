import os
import matplotlib.pyplot as plt
import numpy as np
from .config import *
import seaborn as sns
sns.set(style='white')
from datetime import datetime
from datetime import timedelta
import pandas as pd

def folder(path):
   if not os.path.exists(path):
      os.makedirs(path)
   return path


def last_max(list):
   for i in range(len(list))[::-1]:
      if list[i]:
         return i
   return 0


def first_min(list):
   for i in range(len(list)):
      if list[i]:
         return i
   return len(list) - 1


def plot_result(result, q, save_fig=False, plot_meta=None, naive=False):
   date_obj = datetime.strptime(plot_meta['date_str'], '%Y-%m-%d')
   wks = [date_obj + timedelta(weeks=i) for i in list(range(1, result.shape[0] + 1))]
   fig, ax = plt.subplots(figsize=(8, 7))
   ax.plot(wks, result[:, 1], label='$Z^L median$', linewidth=2)
   ax.plot(wks, result[:, 2], '--', label='$Z^L mean$', linewidth=2)
   ax.plot(wks, result[:, 3], '--', label='$Z^U mean$', linewidth=2)
   ax.plot(wks, result[:, 4], label='$Z^U median$', linewidth=2)
   ax.fill_between(wks, result[:, 0], result[:, 5], color='b', alpha=0.25)
   ax.tick_params(labelrotation=20)
   ax.yaxis.set_tick_params(labelsize=13)
   plt.legend()
   plt.ylabel("DFRV")
   plt.xlabel("Weeks Since Prediction")
   if plot_meta:
      if plot_meta['dev'] is not None:
         title = f"{plot_meta['model']} for round {plot_meta['round']} {plot_meta['state']} {plot_meta['target_type']} \n" \
                 f"dev = {plot_meta['dev']}, Scenario {plot_meta['sceanrios'][0]}-{plot_meta['sceanrios'][1]}"
      else:
         title = f"{plot_meta['model']} for round {plot_meta['round']} {plot_meta['state']} {plot_meta['target_type']} \n" \
                 f"epsilon = ({plot_meta['e_l']},{plot_meta['e_u']}), Scenario {plot_meta['sceanrios'][0]}-{plot_meta['sceanrios'][1]}"

      plt.title(title)
      if save_fig:
         if naive:
            path = folder(f"{config['plot_dir']}_naive/round{plot_meta['round']}/{plot_meta['state']}/{plot_meta['target_type']}/"
                          f"{plot_meta['sceanrios'][0]}-{plot_meta['sceanrios'][1]}/")
         else:
            path = folder(f"{config['plot_dir']}/round{plot_meta['round']}/{plot_meta['state']}/{plot_meta['target_type']}/"
                          f"{plot_meta['sceanrios'][0]}-{plot_meta['sceanrios'][1]}/")
         save_title = title.replace('\n','')
         plt.savefig(f"{path}/{save_title}.png",dpi=300)
   plt.show()

def plot_result_naive(result, q, save_fig=False, plot_meta=None):
   date_obj = datetime.strptime(plot_meta['date_str'], '%Y-%m-%d')
   wks = [date_obj + timedelta(weeks=i) for i in list(range(1, result.shape[0] + 1))]
   fig, ax = plt.subplots(figsize=(8, 7))
   ax.fill_between(wks, result[:, 0], result[:, 1], color='b', alpha=0.25)
   ax.tick_params(labelrotation=20)
   ax.yaxis.set_tick_params(labelsize=13)
   plt.legend()
   plt.ylabel("DFRV")
   plt.xlabel("Weeks Since Prediction")
   if plot_meta:
      title = f"{plot_meta['model']} for round {plot_meta['round']} {plot_meta['state']} {plot_meta['target_type']} \n" \
              f"epsilon = ({plot_meta['e_l']},{plot_meta['e_u']}), Scenario {plot_meta['sceanrios'][0]}-{plot_meta['sceanrios'][1]}"
      plt.title(title)
      if save_fig:
         path = folder(
            f"{config['plot_dir']}_naive/round{plot_meta['round']}/{plot_meta['state']}/{plot_meta['target_type']}/"
            f"{plot_meta['sceanrios'][0]}-{plot_meta['sceanrios'][1]}/")
         save_title = title.replace('\n', '')
         plt.savefig(f"{path}/{save_title}.png", dpi=300)
   plt.show()

def id_to_state(id):
   convert = pd.read_csv("data/state_convert.csv", index_col=[0]).set_index('location')
   target = convert.at[id, "location_name"]
   return target

def state_to_id(state):
   if state == 'US':
      return state
   convert = pd.read_csv("data/state_convert.csv", index_col=[0]).set_index('location_name').to_dict()['location']
   return convert[state]