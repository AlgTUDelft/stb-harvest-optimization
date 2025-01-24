from pathlib import Path
import os
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from gc_utils import *
# from gc_params import *

basePath = Path(os.getcwd()).parent

# set parameters
smoother = 1e-7
len_thre = 24
a_thre = 5
l_thre = 30

smoothed_frames = 3

# load data
for cam in [1,3,5]:
    camera = f'RGBCAM{cam}'
    color_feature = np.load(os.path.join(basePath, f'example_data/color_feature{cam}.npy'))

    items_to_check = []
    saved_coefs_sigmoid = []

    for obj_id in np.unique(color_feature[:,1]):
        obj_cr = color_feature[color_feature[:,1]==obj_id]
        obj_a = obj_cr[:,20]
        if (obj_a[:smoothed_frames]).mean()<0 and (obj_a[-smoothed_frames:]).mean()>a_thre:
            arr = color_feature[(color_feature[:,1]==obj_id)&(color_feature[:,19]>l_thre)]
            if len(arr)>len_thre:
                param_bounds = ([-np.inf, -np.inf, arr[0,0]/1000, -np.inf], [np.inf, np.inf, arr[-1,0]/1000, np.inf])
                try:
                    popt, pcov = curve_fit(sigmoid, arr[:,0]/1000, arr[:,20], method='trf', maxfev=3000,bounds=param_bounds) #can be switched to "richard"
                    y_pred = sigmoid(arr[:,0]/1000, *popt)
                    saved_coefs_sigmoid.append([obj_id, arr[0,0], arr[-1,0]]+list(popt)+[y_pred[-1]-y_pred[0], mse(arr[:,20],y_pred), mse_smoothed(arr[:,20],y_pred)])
                except RuntimeError:
                    items_to_check.append(obj_id)

    df = pd.DataFrame(saved_coefs_sigmoid)
    df.columns = ['obj_id','frame_0','frame_-1','A', 'B', 'M', 'C', 'pred_diff','pred_err','pred_err_s']
    df.to_csv(f'sigmoid_analysis_2021c{cam}.csv')