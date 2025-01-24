from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.k_means import TimeSeriesKMeans

def ensemble_correlate(feature_tar,feature_src):
    lags = []
    for i in range(feature_tar.shape[1]):
        cross_corr = correlate(feature_tar[:,i],feature_src[:,1], mode='full')
        lag = np.argmax(cross_corr) - (len(feature_src) - 1)
        lags.append(lag)
    return int(np.median(lags))

n_clusters = 6

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

for mtx in ['dtw','euclidean','ddtw', 'wdtw']:
    for method in ['means','medoids']:
        start_time = time.time()

        if method=='means':
            clf = TimeSeriesKMeans(n_clusters=n_clusters,metric=mtx)
        else:
            clf = TimeSeriesKMedoids(n_clusters=n_clusters,metric=mtx)
        clf.fit(scaled_features[1:,:])

        observation_date = frame_dict_val[check_frame]
        weather_arr = test_features[test_features[:,0]==observation_date][:,1:]
        # date_using = knn_predictor.predict(weather_arr).mean()
        train_label = clf.predict(scaled_features[1:,:])
        print(n_clusters,mtx,silhouette_score(scaled_features[1:,:], train_label))

        label = clf.predict(weather_arr)
        if label in train_label:
            dates_using = scaled_features[np.where(train_label==label)[0],0].astype(int)

            # frames_using = [frame_dict_rev[date_using] for date_using in dates_using if date_using in frame_dict_rev.keys()]
            frames_using = [dd*24 for dd in dates_using]
            if filter_known_curves(known_curves, frames=frames_using):
                median_shift = (observation_date -np.median(dates_using) +1)*24
            
                if len(track_states[track_states[:,1]==1])>0:
                    for obj_id in track_states[track_states[:,1]==1][:,0]:
                        arr = copy.copy(cr_gt_gsd[(cr_gt_gsd[:,1]==obj_id)&(cr_gt_gsd[:,19]>l_thre)])
                        arr[:,20] = arr[:,20]*scaler_a + scaler_b
                        history_fr0.append(frame_dict_val[arr[0,0]])
                        transformed_arr = filter_history_observations_s(arr, check_frame=check_frame, history_range=history_range)      

                        future_arr = normalize_future_observations(arr, check_frame=check_frame)
                        actual_states.append(future_arr)
                        
                        if 'dtw' not in mtx:
                            knn = best_curves_fixed_shift((transformed_arr[:,1]), known_curves, frames=frames_using, num_best=3, metric='l2')
                        else:
                            knn = best_curves_fixed_shift((transformed_arr[:,1]), known_curves, frames=frames_using, num_best=3, metric='dtw')
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

                        if 'dtw' not in mtx:
                            knn = best_curves_fixed_shift((transformed_arr[:,1]), known_curves, frames=frames_using, num_best=3, metric='l2')
                        else:
                            knn = best_curves_fixed_shift((transformed_arr[:,1]), known_curves, frames=frames_using, num_best=3, metric='dtw')
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
                print(f"{mtx}-{method} uses time: {elapsed_time} seconds")
                    
                # plt.title(f"{mtx}-{method}")
                # plt.savefig(f'2022-weather-cluster_{n_clusters}-{mtx}-{method}.png')

                np.save(f'../results/2022-weather-cluster_{n_clusters}-{mtx}-{method}_prediction_1209.npy', np.stack(predicted_states))
            
            else:
                print(f'cluster_{n_clusters}-{mtx}-{method} does not match a shift.')

mean_evalr = []
mean_evalg = []
mean_evala = []
# for idx1,mtx in enumerate(['dtw-medoids','wdtw-means']):
# for idx1,mtx in enumerate(['ddtw-means']):
for idx1 in range(len(all_evaluations)):
    evr = all_evaluations[idx1][:12]
    mean_evalr.append(np.nanmean(evr,axis=0))

    evg = all_evaluations[idx1][12:]
    mean_evalg.append(np.nanmean(evg,axis=0))

    mean_evala.append(np.nanmean(all_evaluations[idx1],axis=0))


dfa = pd.DataFrame(mean_evala, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])
dfr = pd.DataFrame(mean_evalr, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])
dfg = pd.DataFrame(mean_evalg, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])

file_path = f'../results/2022-weather-cluster_{n_clusters}_1209.xlsx'

with pd.ExcelWriter(file_path) as writer:
    dfa.to_excel(writer, sheet_name='all', index=False)
    dfr.to_excel(writer, sheet_name='red', index=False)
    dfg.to_excel(writer, sheet_name='green', index=False)

dfa[['l2','dtw','softdtw','autocorr','lbk']].round(2)