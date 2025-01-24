from sklearn.neighbors import KNeighborsClassifier

def ensemble_correlate(feature_tar,feature_src):
    lags = []
    for i in range(feature_tar.shape[1]):
        cross_corr = correlate(feature_tar[:,i],feature_src[:,1], mode='full')
        lag = np.argmax(cross_corr) - (len(feature_src) - 1)
        lags.append(lag)
    return int(np.median(lags))

#%%
#-----------------------growing: predict and voting at same stage
using_coef = []
using_coef_dates = []
history_states = []
predicted_states = []
actual_states = []
history_fr0 = []
check_tracks_sorted = []

votes_all = []
shifts_all = []
starts_all = []
knns_all = []

all_evaluations = []

for mtx in ['cosine','euclidean']:
    start_time = time.time()

    knn_predictor = KNeighborsClassifier(metric=mtx,n_neighbors=3)
    knn_predictor.fit(scaled_features[:,1:], scaled_features[:,0])

    observation_date = frame_dict_val[check_frame]
    weather_arr = test_features[test_features[:,0]==observation_date][:,1:]
    date_using = knn_predictor.predict(weather_arr).mean()
    median_shift = (observation_date - date_using +1)*24
    # frame_using = frame_dict_rev[date_using]
    frame_using = date_using*24
    
    if len(track_states[track_states[:,1]==1])>0:
        for obj_id in track_states[track_states[:,1]==1][:,0]:
            arr = copy.copy(cr_gt_gsd[(cr_gt_gsd[:,1]==obj_id)&(cr_gt_gsd[:,19]>l_thre)])
            arr[:,20] = arr[:,20]*scaler_a + scaler_b
            history_fr0.append(frame_dict_val[arr[0,0]])
            transformed_arr = filter_history_observations_s(arr, check_frame=check_frame, history_range=history_range)      

            future_arr = normalize_future_observations(arr, check_frame=check_frame)
            actual_states.append(future_arr)
            
            knn = best_curves_fixed_shift((transformed_arr[:,1]), known_curves, frames=[frame_using], num_best=3, metric='l2')
            selection = np.argmin(knn[:,2])
            if knn[selection][1]>knn[0][1]*1.1:
                selection=0
            bc = knn[selection]

            obj_id, fr0, fr1, A, B, M0, C, _, _, _,_ = saved_coefs[int(bc[0])]
            C = np.max([0,C])
            M = M0+median_shift/1000+bc[2]/1000
            using_coef.append([obj_id,A,B,M,C])
            history_states.append(transformed_arr)
            predicted_states.append(sigmoid(np.arange(check_frame,check_frame+prediction_range+25)/1000, 100, B, M, C))

    if len(track_states[track_states[:,1]==0])>0:
        for obj_id in track_states[track_states[:,1]==0][:,0]:
            arr = copy.copy(cr_gt_gsd[(cr_gt_gsd[:,1]==obj_id)&(cr_gt_gsd[:,19]>l_thre)])
            arr[:,20] = arr[:,20]*scaler_a + scaler_b
            history_fr0.append(frame_dict_val[arr[0,0]])
            transformed_arr = filter_history_observations_s(arr, check_frame=check_frame, history_range=history_range)      

            future_arr = normalize_future_observations(arr, check_frame=check_frame)
            actual_states.append(future_arr)
            
            knn = best_curves_fixed_shift((transformed_arr[:,1]), known_curves, frames=[frame_using], num_best=3, metric='l2')
            selection = np.argmin(knn[:,2])
            if knn[selection][1]>knn[0][1]*1.1:
                selection=0
            bc = knn[selection]

            obj_id, fr0, fr1, A, B, M0, C, _, _, _,_ = saved_coefs[int(bc[0])]
            C = np.max([0,C])
            M = M0+median_shift/1000+bc[2]/1000
            using_coef.append([obj_id,A,B,M,C])
            history_states.append(transformed_arr)
            predicted_states.append(sigmoid(np.arange(check_frame,check_frame+prediction_range+25)/1000, 100, B, M, C))

    evaluations = []
    for i in range(len(actual_states)):
        evaluations.append(evaluate(actual_states[i], predicted_states[i]))
        # plt.plot(predicted_states[i])
    all_evaluations.append(evaluations)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{mtx} uses time: {elapsed_time} seconds")
        
    # plt.title(mtx)
    # plt.savefig(f'2022-weather-knn-{mtx}.png')

    np.save(f'../results/2022-weather-knn-{mtx}_prediction_1209.npy', np.stack(predicted_states))


mean_evalr = []
mean_evalg = []
mean_evala = []
for idx1,mtx in enumerate(['euclidean','l2']):
        evr = all_evaluations[idx1][:12]
        mean_evalr.append(np.nanmean(evr,axis=0))

        evg = all_evaluations[idx1][12:]
        mean_evalg.append(np.nanmean(evg,axis=0))

        mean_evala.append(np.nanmean(all_evaluations[idx1],axis=0))

dfa = pd.DataFrame(mean_evala, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])
dfr = pd.DataFrame(mean_evalr, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])
dfg = pd.DataFrame(mean_evalg, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])

file_path = '../results/2022-weather-knn-1209.xlsx'

with pd.ExcelWriter(file_path) as writer:
    dfa.to_excel(writer, sheet_name='all', index=False)
    dfr.to_excel(writer, sheet_name='red', index=False)
    dfg.to_excel(writer, sheet_name='green', index=False)