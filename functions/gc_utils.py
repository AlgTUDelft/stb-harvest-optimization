import numpy as np

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

def richards_curve(x, A, k, B, v, M, C):
    y = A / (1 + k * np.exp(-B * (x - M)))**(1/v) + C
    return (y)

def mse(y_real,y_pred):
    return ((y_real-y_pred)**2).sum()/len(y_real)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def mse_smoothed(y_real,y_pred):
    y_ = moving_average(y_real,5)
    y_pred = y_pred[4:]
    return ((y_-y_pred)**2).sum()/len(y_real)

def growth_curve_chara(fit_sigmoid, gp_thre=0.1):
    gp_value = (fit_sigmoid.max() - fit_sigmoid.min())*gp_thre + fit_sigmoid.min()
    gp_date = arr[np.where(fit_sigmoid>=gp_value)[0][0]-1,0]
    sp_value = (fit_sigmoid.max() - fit_sigmoid.min())*(1-gp_thre) + fit_sigmoid.min()
    sp_date = arr[np.where(fit_sigmoid>=sp_value)[0][0]-1,0]

    return gp_date, sp_date