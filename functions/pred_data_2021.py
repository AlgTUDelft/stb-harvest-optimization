from pred_utils import *

# year = 2021
camera = 'RGBCAM3'
check_frame = 1000

# Frame info of training set
frame_info = pd.read_csv(os.path.join(basePath, 'example_data', 'img_dict_cam1.csv'), index_col=0)
frame_info['frame_date'] = frame_info['file_name'].str[13:17]
frame_info['Date'] = pd.to_datetime(frame_info['frame_date'], format='%m%d', errors='coerce')
frame_info['Date'] = frame_info['Date'].apply(lambda x: x.replace(year=2021))
frame_info['DayOfYear'] = frame_info['Date'].dt.dayofyear
frame_dict = dict(zip(frame_info['frame_n'], frame_info['DayOfYear']))
frame_dict_rev = dict(zip(frame_info['DayOfYear'], frame_info['frame_n']))

frame_info_ = pd.read_csv(os.path.join(basePath, 'example_data', f'img_dict_cam{camera[-1]}.csv'), index_col=0)

def map_values(val):
    return frame_dict[val]

# Known coefficients from training set
df1 = pd.read_csv('../example_data/sigmoid_fitting_2021c1_20241205-GW_50_lbksmoothed.csv', index_col=0)
saved_coefs_sigmoid = df1.values
saved_coefs = saved_coefs_sigmoid[(saved_coefs_sigmoid[:,-4]>10)&(saved_coefs_sigmoid[:,-2]<700)]

vectorized_map = np.vectorize(map_values)
mapped_dates = vectorized_map(saved_coefs[:,2])
known_dates = np.unique(mapped_dates)

mapped_dates_ = vectorized_map(saved_coefs[:,1])
known_dates_ = np.unique(mapped_dates_)

df1 = df1[(df1['pred_diff']>10)&(df1['pred_err_s']<700)].reset_index(drop=True)
df1['date_0'] = mapped_dates_
df1['date_1'] = mapped_dates

lower_cap,upper_cap = box_bounds(saved_coefs[saved_coefs[:,4]>0][:,4])

known_curves = []
saved_idx = []
saved_idx_old = []
for i in range(len(saved_coefs)):
    obj_id, fr0, fr1, A, B, M, C, _, _, _, _ = saved_coefs[i]
    x = np.arange(fr0, fr1)
    y = sigmoid(x/1000, 100, B, M, 0)
    if B<lower_cap or B>upper_cap:
        continue
    else:
        known_curves.append(np.column_stack((x,y)))
        saved_idx.append(i)
print('the training set has', len(known_curves), 'curves')

saved_coefs = saved_coefs[saved_idx]
idx_refer = {value:index for index, value in enumerate(saved_idx)}
df1 = df1.loc[saved_idx].reset_index(drop=True)


# Frame info of testing set
frame_info_ = pd.read_csv(os.path.join(basePath, 'example_data', 'img_dict_cam3.csv'), index_col=0)
frame_info_['frame_date'] = frame_info_['file_name'].str[13:17]
frame_info_['Date'] = pd.to_datetime(frame_info_['frame_date'], format='%m%d', errors='coerce')
frame_info_['Date'] = frame_info_['Date'].apply(lambda x: x.replace(year=2021))
frame_info_['DayOfYear'] = frame_info_['Date'].dt.dayofyear
frame_dict_val = dict(zip(frame_info_['frame_n'], frame_info_['DayOfYear']))

# Color info of the testing set
cr_gt_gsd = np.load(os.path.join(basePath, 'example_data', f'2021_RGBCAM{camera[-1]}_20241205-GW_final.npy'))

check_tracks = np.unique(cr_gt_gsd[cr_gt_gsd[:,0]==check_frame][:,1])
check_tracks = check_tracks[check_tracks!=154]

track_states = []
check_tracks_sorted = []
for obj_id in check_tracks:
    if len(cr_gt_gsd[cr_gt_gsd[:,1]==obj_id])>len_thre:
        arr = copy.copy(cr_gt_gsd[(cr_gt_gsd[:,1]==obj_id)&(cr_gt_gsd[:,19]>l_thre)])
        track_states.append([obj_id, filter_observed_curves(arr, check_frame)])

track_states = np.array(track_states)

# Classify tracks in the testing set
upper_bound = 0
lower_bound = 0
for tr in check_tracks:
    arr = copy.copy(cr_gt_gsd[(cr_gt_gsd[:,1]==tr)&(cr_gt_gsd[:,19]>50)])
    arr = moving_average(arr[:,20],window_size=6)
    if arr.max()>upper_bound:
        upper_bound = arr.max()
    if arr.min()<lower_bound:
        lower_bound = arr.min()

first_frame = cr_gt_gsd[np.isin(cr_gt_gsd[:,1],check_tracks)][:,0].min()
last_frame = cr_gt_gsd[np.isin(cr_gt_gsd[:,1],check_tracks)][:,0].max()

scaler_b = -lower_bound
scaler_a = 100/(upper_bound-lower_bound)

check_tracks_sorted = np.hstack((track_states[track_states[:,1]==1][:,0],track_states[track_states[:,1]==0][:,0]))
check_tracks_sorted = np.hstack((check_tracks_sorted,track_states[track_states[:,1]==2][:,0]))


# Weather information
dfw1 = pd.read_csv('../example_data/weather_2021.csv', index_col=0)
dfw2 = pd.read_csv('../example_data/weather_2021.csv', index_col=0)

feature_cols = ['2t', 'cdir', 'uvb', 'e', 'lai_lv', 'stl1', 'ssrdc', 'tp', 'fdir', 'vitoe']

scaler = StandardScaler()
scaler.fit(dfw1[feature_cols])
scaled_features = scaler.transform(dfw1[feature_cols])
scaled_features_val = scaler.transform(dfw2[feature_cols])
dfw1[feature_cols] = scaled_features
dfw2[feature_cols] = scaled_features_val
scaled_features = np.hstack((dfw1[['dataDate']].values,scaled_features))
scaled_features_val = np.hstack((dfw2[['dataDate']].values,scaled_features_val))

scaled_features = []
for d0 in dfw1['dataDate'].values[7:]:
    scaled_features.append(dfw1[(dfw1.dataDate>=d0-7)&(dfw1.dataDate<d0)][feature_cols].values.reshape(-1))
    
scaled_features = np.array(scaled_features)
scaled_features = np.hstack((dfw1['dataDate'].values[7:].reshape(-1,1),scaled_features))
print('scaled_features', scaled_features.shape)

test_features = []
for d0 in dfw2['dataDate'].values[7:]:
    test_features.append(dfw2[(dfw2.dataDate>=d0-7)&(dfw2.dataDate<d0)][feature_cols].values.reshape(-1))
    
test_features = np.array(test_features)
test_features = np.hstack((dfw2[['dataDate']].values[7:].reshape(-1,1),test_features))

print('test_features', test_features.shape)