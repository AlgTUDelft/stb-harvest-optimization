
from pathlib import Path
import os
import copy 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from sklearn.preprocessing import StandardScaler

from scipy.spatial import distance
from scipy.stats import chisquare
from scipy.spatial.distance import chebyshev

from scipy.interpolate import interp1d
import math

import json
import time

from tslearn.metrics import lcss, lcss_path, lcss_path_from_metric, \
                            ctw, ctw_path, \
                            dtw, soft_dtw, soft_dtw_alignment,\
                            gak,\
                            lb_keogh

from tslearn.metrics import lb_envelope
from tslearn.metrics import cdist_dtw, cdist_soft_dtw, cdist_soft_dtw

from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.k_means import TimeSeriesKMeans
from tslearn.clustering import KShape
from tslearn.svm import TimeSeriesSVC
from tslearn.clustering import silhouette_score

from sktime.distances import edr_distance
from frechetdist import frdist

from scipy.signal import correlate
from scipy.optimize import curve_fit

import cv2



np.set_printoptions(suppress=True)

basePath = Path(os.getcwd()).parent


len_thre = 24
a_thre = 30
l_thre = 30

smoothed_frames = 3

a_decrease_thre = 15
s_decrease_thre = 15
trend_thre = 5

smoother = 1e-7

ripe_thre = 40

prediction_range = 7*24

history_range = 12*24

smoothing_window = 12

flexible_range = 7


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

def sigmoid(x, A, B, M, C):
    y = A / (1 + np.exp(-B*(x-M))) + C
    return (y)

def sigmoid_(x, A, B, M, C):
    y = 100 / (1 + np.exp(-B*(x-M))) + 0
    return (y)

def richards_curve(x, A, k, B, v, M, C):
    y = A / (1 + k * np.exp(-B * (x - M)))**(1/v) + C
    return (y)

def mse(y_real,y_pred):
    return ((y_real-y_pred)**2).sum()/len(y_real)

def mae(y_real,y_pred):
    return (np.abs(y_real-y_pred)).sum()/len(y_real)

def me(y_real,y_pred):
    return np.abs((y_real-y_pred).sum()/len(y_real))

def lbk(y_real,y_pred, r=1):
    env_low, env_up = lb_envelope(y_real, radius=r)
    return lb_keogh(y_pred, envelope_candidate=(env_low, env_up))

def env_smooth(data, window_size=3):
    env_low, env_up = lb_envelope(data, radius=(window_size-2))
    return (env_low+env_up).mean()

def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def mse_smoothed(y_real,y_pred):
    y_ = moving_average(y_real,4)
    y_pred = y_pred[3:]
    return ((y_-y_pred)**2).sum()/len(y_real)

def rmse_predicted_states(y_real, y_pred):
    y_pred = y_pred[:len(y_real)]
    mean_se = mse(y_real[:,1], y_pred)
    return np.sqrt(mean_se)

def moving_average_continuous(index, values, window_size):
    continuous_index = np.arange(np.min(index), np.max(index)+1,1)
    new_idx = []
    result = []
    for idx in continuous_index:
        start_idx = idx - window_size + 1
        window_indices = np.where((index >= start_idx) & (index <= idx))[0]
        if len(window_indices) > 0:
            window_values = values[window_indices]
            result.append(np.median(window_values))
            new_idx.append(idx)
    return np.array(idx), np.array(result)


def filter_history_observations(data, check_frame, history_range=7*24):
    arr = copy.copy(data[(data[:,0]<=check_frame)&(data[:,0]>=check_frame-history_range)])
    x = arr[:, 0]
    x_new = np.linspace(x.min(), x.max(), num=int(x.max()-x.min())+1)
    y = arr[:, 20]
    valid_indices = ~np.isnan(y)
    y_new = np.interp(x_new, x[valid_indices], y[valid_indices])
    new_data = np.column_stack((x_new, y_new))

    return new_data

def filter_history_observations_s(data, check_frame, history_range=7*24, smoothing_window=3):
    arr = copy.copy(data[(data[:,0]<=check_frame)&(data[:,0]>=check_frame-history_range)])
    x = arr[:, 0]
    x_new = np.linspace(x.min()-smoothing_window, x.max(), num=int(x.max()-x.min()+smoothing_window+1))
    y = arr[:, 20]
    valid_indices = ~np.isnan(y)
    y_new = np.interp(x_new, x[valid_indices], y[valid_indices])
    y_new_smoothed = moving_average(y_new, smoothing_window)
    new_data = np.column_stack((x_new[smoothing_window-1:], y_new_smoothed))

    return new_data

def normalize_future_observations(data, check_frame, future_range=7*24):
    arr = data[data[:,0]>check_frame]
    x = arr[:, 0]
    max_x = np.min([x.max(), check_frame+future_range])
    x_new = np.linspace(x.min(), max_x, num=int(max_x-x.min()+1))
    y = arr[:, 20]
    valid_indices = ~np.isnan(y)
    y_new = np.interp(x_new, x[valid_indices], y[valid_indices])
    new_data = np.column_stack((x_new, y_new))

    return new_data

def filter_observed_curves(data, check_frame, smoothing_window=12):
    cr = filter_history_observations(data, check_frame=check_frame)
    cr_smoothed = moving_average(cr[:,1], smoothing_window)
    observations = len(cr_smoothed)
    if cr_smoothed.max()<5:
        # still green   
        return 0
    elif cr_smoothed.min()>5:
        # all red
        return 2
    elif cr_smoothed[observations//2:].mean() - cr_smoothed[:observations//2].mean() < -5:
        return 3 
    else:
        return 1

def score_curve(arr_tar, arr_src, metric='l2'):
    assert metric in ['me','l1','l2','lcss','ctw','dtw','softdtw','gak','lbk','lbk-6','lbk-12', 'lbk-24']
    best_fit = np.inf
    saved_start = 0
    alter = False
    if len(arr_src) < len(arr_tar):
        arr_tar, arr_src = arr_src, arr_tar
        alter = True
    # if len(arr_src) >= len(arr_tar):

    for i in range(len(arr_src)-len(arr_tar)):
        arr_src_ = arr_src[i:i+len(arr_tar)]
        if metric == 'l2':
            fit = mse(arr_tar, arr_src_)
        elif metric == 'l1':
            fit = mae(arr_tar, arr_src_)
        elif metric == 'me':
            fit = me(arr_tar, arr_src_)
        elif metric == 'lcss':
            fit = lcss(arr_tar, arr_src_)
        elif metric == 'ctw':
            fit = ctw(arr_tar, arr_src_)
        elif metric == 'dtw':
            fit = dtw(arr_tar, arr_src_)
        elif metric == 'softdtw':
            fit = soft_dtw(arr_tar, arr_src_)
        elif metric == 'gak':
            fit = gak(arr_tar, arr_src_)
        elif metric == 'lbk':
            fit = lbk(arr_tar, arr_src_, r=1)
        elif metric == 'lbk-6':
            fit = lbk(arr_tar, arr_src_, r=4)
        elif metric == 'lbk-12':
            fit = lbk(arr_tar, arr_src_, r=12)
        elif metric == 'lbk-24':
            fit = lbk(arr_tar, arr_src_, r=24)


        if metric not in ['lcss','gak'] and fit < best_fit:
            best_fit = fit
            saved_start = i
        elif fit > best_fit:
            best_fit = fit
            saved_start = i
    if alter:
        saved_start = -i
    return best_fit, saved_start

def best_curves(arr_tar, known_curves, num_best=3, metric='l2'):
    assert metric in ['me','l1','l2','lcss','ctw','dtw','softdtw','gak','lbk','lbk-6','lbk-12', 'lbk-24']
    curvefit = []
    for i,kc in enumerate(known_curves):
        curvefit.append([i]+list(score_curve(arr_tar, kc[:,1], metric=metric)))
    curvefit = np.array(curvefit)
    if metric not in ['lcss','gak']:
        curvefit = curvefit[np.argsort(curvefit[:,1])]
    else:
        curvefit = curvefit[np.argsort(curvefit[:,1], ascending=False)]
    return curvefit[:num_best]

def score_curve_fixed(arr_tar, arr_src_2d, frame, metric='l2'):
    assert metric in ['me','l1','l2','lcss','ctw','dtw','softdtw','gak','lbk','lbk-6','lbk-12', 'lbk-24']
    best_fit = np.inf
    saved_start = 0

    arr_src = arr_src_2d[arr_src_2d[:,0]<=frame+3*24][:,1]
    arr_src_frames = arr_src_2d[arr_src_2d[:,0]<=frame+3*24][:,0]

    alter = False
    if len(arr_src) < len(arr_tar):
        arr_tar, arr_src = arr_src, arr_tar
        alter = True
    # if len(arr_src) >= len(arr_tar):

    # for i in range(len(arr_src)-len(arr_tar)):
    for i in range(np.max([0, len(arr_src)-len(arr_tar)-3*24]),(len(arr_src)-len(arr_tar))):
        arr_src_ = arr_src[i:i+len(arr_tar)]
        if metric == 'l2':
            fit = mse(arr_tar, arr_src_)
        elif metric == 'l1':
            fit = mae(arr_tar, arr_src_)
        elif metric == 'me':
            fit = me(arr_tar, arr_src_)
        elif metric == 'lcss':
            fit = lcss(arr_tar, arr_src_)
        elif metric == 'ctw':
            fit = ctw(arr_tar, arr_src_)
        elif metric == 'dtw':
            fit = dtw(arr_tar, arr_src_)
        elif metric == 'softdtw':
            fit = soft_dtw(arr_tar, arr_src_)
        elif metric == 'gak':
            fit = gak(arr_tar, arr_src_)
        elif metric == 'lbk':
            fit = lbk(arr_tar, arr_src_, r=1)
        elif metric == 'lbk-6':
            fit = lbk(arr_tar, arr_src_, r=4)
        elif metric == 'lbk-12':
            fit = lbk(arr_tar, arr_src_, r=12)
        elif metric == 'lbk-24':
            fit = lbk(arr_tar, arr_src_, r=24)

        if metric not in ['lcss','gak'] and fit < best_fit:
            best_fit = fit
            saved_start = i
        elif fit > best_fit:
            best_fit = fit
            saved_start = i

    if alter:
        saved_start = -i
    
    return best_fit, saved_start

def best_curves_fixed_shift(arr_tar, known_curves, frames=[0], num_best=3, metric='l2'):
    assert metric in ['me','l1','l2','lcss','ctw','dtw','softdtw','gak','lbk','lbk-6','lbk-12', 'lbk-24']
    curvefit = []
    for frame in frames:
        for i,kc in enumerate(known_curves):
            if kc[0,0]<frame and kc[-1,0]>frame:
                curvefit.append([i]+list(score_curve_fixed(arr_tar, kc, frame=frame, metric=metric)))
    curvefit = np.array(curvefit)
    if metric not in ['lcss','gak']:
        curvefit = curvefit[np.argsort(curvefit[:,1])]
    else:
        curvefit = curvefit[np.argsort(curvefit[:,1], ascending=False)]
    return curvefit[:num_best]

def filter_known_curves(known_curves, frames=[0]):
    for frame in frames:
        for i,kc in enumerate(known_curves):
            if kc[0,0]<frame and kc[-1,0]>frame:
                return True
    return False

def score_curve_auto_corr(arr_tar, arr_src):
    alter = False
    if len(arr_src) < len(arr_tar):
        arr_tar, arr_src = arr_src, arr_tar
        alter = True
    cross_corr = correlate(arr_tar, arr_src, mode='full')
    lag = np.argmax(cross_corr) - (len(arr_src) - 1)
    corr = np.max(cross_corr)
    if alter:
        lag = -lag
    return -corr, lag

def best_curves_auto_corr(arr_tar, known_curves, num_best=3):
    curvefit = []
    for i,kc in enumerate(known_curves):
        curvefit.append([i]+list(score_curve_auto_corr(arr_tar, kc[:,1])))
    
    curvefit = np.array(curvefit)
    curvefit = curvefit[np.argsort(curvefit[:,1])]
    return curvefit[:num_best]

def score_curve_fixed_corr(arr_tar, arr_src_2d, frame):
    arr_src = arr_src_2d[arr_src_2d[:,0]<=frame+3*24][:,1]
    arr_src_frames = arr_src_2d[arr_src_2d[:,0]<=frame+3*24][:,0]
    alter = False
    if len(arr_src) < len(arr_tar):
        arr_tar, arr_src = arr_src, arr_tar
        alter = True

    cross_corr = correlate(arr_tar, arr_src, mode='full')
    starting_idx = np.max([0, len(arr_src)-len(arr_tar)-3*24])
    lag = np.argmax(cross_corr[starting_idx:]) - (len(arr_src) - 1)
    corr = np.max(cross_corr[starting_idx:])
    if alter:
        lag = -lag
    return -corr, lag

def best_curves_fixed_corr(arr_tar, known_curves, frames=[0], num_best=3):
    curvefit = []
    for frame in frames:
        for i,kc in enumerate(known_curves):
            if kc[0,0]<frame and kc[-1,0]>frame:
                curvefit.append([i]+list(score_curve_fixed_corr(arr_tar, kc, frame=frame)))
    curvefit = np.array(curvefit)

    curvefit = curvefit[np.argsort(curvefit[:,1])]
    return curvefit[:num_best]

def growth_curve_chara(x,fit_sigmoid, gp_thre=0.1):
    gp_value = (fit_sigmoid.max() - fit_sigmoid.min())*gp_thre + fit_sigmoid.min()
    gp_date = x[np.where(fit_sigmoid>=gp_value)[0][0]-1]
    sp_value = (fit_sigmoid.max() - fit_sigmoid.min())*(1-gp_thre) + fit_sigmoid.min()
    sp_date = x[np.where(fit_sigmoid>=sp_value)[0][0]-1]

    return gp_date, sp_date


def box_bounds(arr):
    box = plt.boxplot(arr)
    whiskers = box['whiskers']
    caps = box['caps']
    lower_bound = whiskers[0].get_ydata()[1]
    upper_bound = whiskers[1].get_ydata()[1]

    lower_cap = caps[0].get_ydata()[1]
    upper_cap = caps[1].get_ydata()[1]

    print(f"Lower Bound (whisker): {lower_bound}, Cap: {lower_cap}")
    print(f"Upper Bound (whisker): {upper_bound}, Cap: {upper_cap}")

    return lower_bound, upper_bound


def evaluate(actual_states, y_pred):
    y_real = actual_states[:,1]
    y_pred = y_pred[:len(y_real)]
    y_real = y_real/100
    y_pred = y_pred/100
    res_l2 = mse(y_pred, y_real)
    res_l1 = mae(y_pred, y_real)
    res_me = me(y_pred, y_real)
    res_lcss = lcss(y_pred, y_real)
    try:
        res_ctw = ctw(y_pred, y_real)
    except ValueError:
        res_ctw = np.nan
    res_dtw = dtw(y_pred, y_real)
    res_softdtw = soft_dtw(y_pred, y_real)
    res_autocorr = correlate(y_pred, y_real, mode='full')[0]

    res_gak = gak(y_pred, y_real)
    res_lbk = lbk(y_pred, y_real, r=1)
    res_lbk6 = lbk(y_pred, y_real, r=4)
    res_lbk12 = lbk(y_pred, y_real, r=10)

    res_frdist = frdist1d(actual_states, y_pred)
    res_edr = edr_distance(y_pred, y_real)

    return [res_l2,res_l1,res_me,res_lcss,res_ctw,res_dtw,res_softdtw,res_autocorr,res_gak,res_lbk,res_lbk6,res_lbk12,res_frdist,res_edr]

def evaluate_s(actual_states, y_pred):
    y_real = moving_average(actual_states[:,1])
    y_pred = y_pred[:len(y_real)]
    y_real = y_real/100
    y_pred = y_pred/100
    res_l2 = mse(y_pred, y_real)
    res_l1 = mae(y_pred, y_real)
    res_me = me(y_pred, y_real)
    res_lcss = lcss(y_pred, y_real)
    try:
        res_ctw = ctw(y_pred, y_real)
    except ValueError:
        res_ctw = np.nan
    res_dtw = dtw(y_pred, y_real)
    res_softdtw = soft_dtw(y_pred, y_real)
    res_autocorr = correlate(y_pred, y_real, mode='full')[0]

    res_gak = gak(y_pred, y_real)
    res_lbk = lbk(y_pred, y_real, r=1)
    res_lbk6 = lbk(y_pred, y_real, r=4)
    res_lbk12 = lbk(y_pred, y_real, r=10)

    res_frdist = frdist1d(actual_states, y_pred)
    res_edr = edr_distance(y_pred, y_real)

    return [res_l2,res_l1,res_me,res_lcss,res_ctw,res_dtw,res_softdtw,res_autocorr,res_gak,res_lbk,res_lbk6,res_lbk12,res_frdist,res_edr]


def frdist1d(actual_states,y_pred):
    y_real = actual_states/100
    y_pred = y_pred[:len(y_real)]
    y_real = y_real[:len(y_pred)]
    y_pred = np.hstack((y_real[:,0].reshape(-1,1),y_pred.reshape(-1,1)))
    return frdist(y_real,y_pred)


