from pathlib import Path
import copy 
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


import pandas as pd
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import linear_sum_assignment as linear_assignment

import cv2

from collections import Counter

np.set_printoptions(suppress=True)

basePath = Path(os.getcwd()).parent


color_cycle = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#ffbb78',  # Light Orange
    '#98df8a'   # Light Green
]

color_cycle2 = ["#D8BFD8","#DDA0DD", "#DA70D6", "#D15FEE", "#A020F0", "#4B0082"]

rainbow_colors = ['#FF0000',  # Red
                  '#FF7F00',  # Orange
                  '#FFFF00',  # Yellow
                  '#00FF00',  # Green
                  '#0000FF',  # Blue
                  '#4B0082',  # Indigo
                  '#8B00FF',  # Violet
                  '#444444']  

prediction_range = 7*24

history_range = 12*24

smoothing_window = 12

d_average = 3
market_demands = [d_average]*(prediction_range//24)

def count_mature_time(array, bar=0.8):
    n = len(array)
    counts = np.zeros(n, dtype=int)  
    count = 0
    for i in range(n):
        if array[i] < bar:
            count = 0
            # continue  
        else:
            count += 1
        # for j in range(i, -1, -1):  # Traverse backwards from current index to start
        #     if array[j] < precision:
        #         count += 1
        #     else:
        #         break  # Stop counting at the first non-zero element
        counts[i] = count
    
    return counts

def continue_score(pred, precision=1e-2, min_pred=0, max_pred=100):
    arr = copy.copy(pred)
    min_pred = np.min([arr.min(), min_pred])
    max_pred = np.max([arr.max(), max_pred])
    arr = (arr-min_pred)/max_pred
    return arr

def binary_score(pred, bar=0.8):
    arr = continue_score(pred)
    return arr>bar

def continue_score_lim(pred, precision=1e-2, lower_bar=0.2, upper_bar=0.8):
    arr = continue_score(pred)
    if np.min(arr) < lower_bar:
        arr[arr<lower_bar] = lower_bar-(lower_bar-arr[arr<lower_bar])*2
    if np.max(arr)> upper_bar:
        arr[arr>upper_bar] = arr[arr>upper_bar]+(arr[arr>upper_bar]-upper_bar)*2
    return arr
    
def rot_score_squared(pred, a = 1e-4, mature_time=72, lower_lim=-0.5, upper_lim=0.8):
    score = continue_score(pred)
    rot = np.where(count_mature_time(score, bar=upper_lim-0.1)>=mature_time)
    if len(rot[0])>0:
        rot_idx = rot[0][0]
        if rot_idx==mature_time-1:
            rot_idx = mature_time-24
        x = np.arange(rot_idx, prediction_range+25) - rot_idx
        score[rot_idx:] = -a*(x**2)+score[rot_idx-1]
    return np.clip(score, lower_lim, upper_lim)
    
def rot_score_cubic(arr, a = 1e-6, mature_time=72, lower_lim=-0.5, upper_lim=0.8):
    score = continue_score(arr)
    rot = np.where(count_mature_time(score, bar=upper_lim-0.1)>=mature_time)
    if len(rot[0])>0:
        rot_idx = rot[0][0]
        if rot_idx==mature_time-1:
            rot_idx = mature_time-24
        x = np.arange(rot_idx+24, prediction_range+49) - rot_idx
        score[rot_idx:] = -a*(x**3)+score[rot_idx-1] + a*(24**3)
    return np.clip(score, lower_lim, upper_lim)

def rot_score_cubic_neg(arr, a = 1e-6, mature_time=72, lower_lim=-0.5, upper_lim=1):
    score = continue_score_lim(arr)
    rot = np.where(count_mature_time(score, bar=upper_lim-0.1)>=mature_time)
    if len(rot[0])>0:
        rot_idx = rot[0][0]
        if rot_idx==mature_time-1:
            rot_idx = mature_time-24
        x = np.arange(rot_idx+24, prediction_range+49) - rot_idx
        score[rot_idx:] = -a*(x**3)+score[rot_idx-1] + a*(24**3)
    return np.clip(score, lower_lim, upper_lim)
    
def step_wise(arr, time=108, upper_bar=80, lower_bar=20):
    score = continue_score_lim(arr, lower_bar=lower_bar)
    score[score<0] = -0.2
    rot = np.where(count_mature_time(score, bar=upper_bar/100)>=time)
    if len(rot[0])>0:
        rot_idx = rot[0][0]
        if rot_idx==time:
            rot_idx = time-24
        score[rot_idx:] = -0.4
    return np.clip(score, lower_lim, upper_lim)

def binary_score_matrix(matrix):
    res = []
    for i in range(len(matrix)):
        res.append(binary_score(matrix[i]))
    return np.array(res)

def continue_score_matrix(matrix):
    res = []
    for i in range(len(matrix)):
        res.append(continue_score(matrix[i]))
    return np.array(res)

def stepwise_score_matrix(matrix):
    res = []
    for i in range(len(matrix)):
        res.append(step_wise(matrix[i]))
    return np.array(res)

def combined_score_matrix(matrix):
    res = []
    for i in range(len(matrix)):
        res.append(rot_score_cubic_neg(matrix[i]))
    return np.array(res)

def harvest_cost(rewards, market_demands, over_supply=0, over_supply_discount=0.5, dummy_demands=False, dummy_reward=0.21):
    trade_dates = []
    cost_matrix = []
    for i, dm in enumerate(market_demands):
        dm_ = int(dm+over_supply*dm)
        if over_supply>0:
            dm_ = np.max([dm_, dm+1])
            # print(dm_)
        for j in range(dm_):
            trade_dates.append(i+1)
            reward_at_harvest = rewards[:,i*24+24]
            if j<dm:
                cost_matrix.append(reward_at_harvest)
            else:
                cost_matrix.append(np.min([reward_at_harvest,reward_at_harvest*over_supply_discount], axis=0))
    cost_matrix = np.stack(cost_matrix)
    if dummy_demands:
        cost_matrix = np.vstack((cost_matrix, dummy_reward*np.ones((len(rewards),len(rewards)))))
        trade_dates = trade_dates + [trade_dates[-1]+1]*len(rewards)
    return cost_matrix, trade_dates

def harvest_optimize(cost_matrix, trade_dates):
    harvest_time = np.ones(cost_matrix.shape[1])*(trade_dates[-1]+1)
    assert len(cost_matrix) == len(trade_dates), print("cost matrix shape mismatch with trade dates")
        # if len(cost_matrix) == len(trade_dates)+cost_matrix.shape[1]:
        #     trade_dates = trade_dates+[trade_dates[-1]+1]*cost_matrix.shape[1]
    row_indices, col_indices = linear_assignment(cost_matrix, maximize=True)
    for i,strb_idx in enumerate(col_indices):
        harvest_time[strb_idx] = trade_dates[row_indices[i]]
    return harvest_time

# def harvest_optimize(cost_matrix, trade_dates):
#     row_indices, col_indices = linear_assignment(cost_matrix.T)
#     # predicted_states_ = predicted_states[:,np.argsort(-col_indices)]
#     # col_indices_ = col_indices[np.argsort(-col_indices)]
#     trade_time = []
#     for idx in col_indices:
#         # trade_time = trade_dates[col_indices_[i]]*24
#         if idx<len(trade_dates):
#             trade_time.append(trade_dates[idx]*24)
#         else:
#             trade_time.append((trade_dates[-1]+1)*24)
#     return trade_time

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plan_evaluation(growth_pred, daily_demand, duration=7, over_supply=0.5, dummy_demands=True):
    market_demands = [daily_demand]*duration
    prediction_range = 24*duration
    
    harvest_score = combined_score_matrix(growth_pred)
    cost_matrix, trade_dates = harvest_cost(harvest_score, market_demands, over_supply=over_supply, dummy_demands=dummy_demands)
    harvest_time = harvest_optimize(cost_matrix, trade_dates)

    harvest_score = 0
    for i in range(len(growth_pred)):
        ending_x = np.min([growth_pred.shape[1],int(harvest_time[i])*24])
        if ending_x<=7*24:
            harvest_score+=rot_score_cubic_neg(growth_pred[i])[ending_x]

    element_counts = Counter(harvest_time)
    harvest_count = np.zeros(prediction_range//24+1)
    for i,ht in enumerate(np.arange(1,8,1)):
        if ht in element_counts.keys():
            harvest_count[i] = element_counts[ht]
    harvest_count = np.vstack((np.arange(24,prediction_range+36,24),harvest_count.reshape(1,-1))) 
    fulfillment = np.min((daily_demand*np.ones(prediction_range//24+1),harvest_count[1]),axis=0).sum()/(daily_demand*duration)*100   

    return harvest_score, fulfillment