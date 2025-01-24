def ensemble_correlate(feature_tar,feature_src):
    lags = []
    for i in range(feature_tar.shape[1]):
        cross_corr = correlate(feature_tar[:,i],feature_src[:,1], mode='full')
        lag = np.argmax(cross_corr) - (len(feature_src) - 1)
        lags.append(lag)
    return int(np.median(lags))

known_uniform_curves = []
y_train = []
for i in range(len(saved_coefs)):
    obj_id, fr0, fr1, A, B, M, C, _, _, _,_ = saved_coefs[i]
    y_train.append(fr0)
    fr0_ = fr1-340
    x = np.arange(fr0_, fr1)
    y = sigmoid(x/1000, 1, B, M, 0)
    known_uniform_curves.append(y)

known_uniform_curves = np.array(known_uniform_curves)

#-----------------------growing: predict and voting at same stage
using_coef = []
using_coef_dates = []
history_states = []
predicted_states = []
actual_states = []
history_fr0 = []
check_tracks_sorted = []

all_evaluations = []

for mtx in ['euclidean','dtw','wdtw']:
    start_time = time.time()

    number_of_days = len(np.unique([i//24 for i in saved_coefs[:,1]]))
    clf = TimeSeriesKMeans(n_clusters=number_of_days,metric=mtx)
    clf.fit(known_uniform_curves)
    train_label = clf.predict(known_uniform_curves)
    print(mtx, silhouette_score(known_uniform_curves, train_label))

    for mtx2 in ['corr', mtx]:
        flag = False

        tryout_votes = []
        tryout_shifts = []
        tryout_starts = []

        actual_states = []
        predicted_states = []

        if len(track_states[track_states[:,1]==1])>0:
            for obj_id in track_states[track_states[:,1]==1][:,0]:
                arr = copy.copy(cr_gt_gsd[(cr_gt_gsd[:,1]==obj_id)&(cr_gt_gsd[:,19]>l_thre)])
                arr[:,20] = arr[:,20]*scaler_a + scaler_b
                history_fr0.append(frame_dict_val[arr[0,0]])
                transformed_arr = filter_history_observations_s(arr, check_frame=check_frame, history_range=history_range)      

                future_arr = normalize_future_observations(arr, check_frame=check_frame)
                actual_states.append(future_arr)

                label = clf.predict(transformed_arr[:,1].reshape(1,-1)/100)
                selected_curve_idx = np.where(train_label==label)[0]
                if len(selected_curve_idx)==0:
                    flag = True
                    break

                if mtx2 == 'corr':
                    bc = best_curves_auto_corr((transformed_arr[:,1]), [known_curves[ii] for ii in selected_curve_idx], 1)[0]
                else:
                    if mtx == 'euclidean':
                        bc = best_curves((transformed_arr[:,1]), [known_curves[ii] for ii in selected_curve_idx], 1, metric='l2')[0]
                    else:
                        bc = best_curves((transformed_arr[:,1]), [known_curves[ii] for ii in selected_curve_idx], 1, metric='dtw')[0]

                obj_id, fr0, fr1, A, B, M0, C, _, _, _,_ = saved_coefs[int(bc[0])]
                tryout_votes.append(M0*1000+bc[2])
                tryout_shifts.append(bc[2])
                tryout_starts.append(fr0)
                C = np.max([0,C])
                M = M0-(fr0+bc[2])/1000+transformed_arr[0,0]/1000
                # using_coef_dates.append(frame_dict[fr0])

                # using_coef.append([obj_id,A,B,M,C])
                history_states.append(transformed_arr)
                predicted_states.append(sigmoid(np.arange(check_frame,check_frame+prediction_range+25)/1000, 100, B, M, C))
                # plt.plot(predicted_states[-1])

        if flag:
            break

        median_vote = np.median(tryout_shifts)
        sorted_indices = np.argsort(tryout_votes)
        sorted_arr = arr[sorted_indices]
        mid1 = len(tryout_votes) // 2 - 1
        median_shift = tryout_shifts[sorted_indices[mid1]]
        median_shifted_start = tryout_starts[mid1]

        if len(track_states[track_states[:,1]==0])>0:
            for obj_id in track_states[track_states[:,1]==0][:,0]:
                check_tracks_sorted.append(obj_id)
                arr = copy.copy(cr_gt_gsd[(cr_gt_gsd[:,1]==obj_id)&(cr_gt_gsd[:,19]>l_thre)])
                arr[:,20] = arr[:,20]*scaler_a + scaler_b
                transformed_arr = filter_history_observations_s(arr, check_frame=check_frame, history_range=history_range)
                future_arr = normalize_future_observations(arr, check_frame=check_frame)
                actual_states.append(future_arr)

                # fr0_shifted_back = arr[0,0] - (median_start-median_vote)
                fr0_shifted_back = frame_dict_val[arr[0,0]]*24 - median_shift
                day0_shifted_back = fr0_shifted_back//24
                if len(df1[df1['date_0']==day0_shifted_back])>0 and df1[df1['date_0']==day0_shifted_back].B.max()>0:
                    closest_indices = df1[df1['date_0']==day0_shifted_back].index
                else:            
                    for t_expand in range(7):
                        df_ = df1[(df1['date_0']>=day0_shifted_back-t_expand)&(df1['date_0']<=day0_shifted_back+t_expand)]
                        if len(df_)>0 and df_.B.max()>0:
                            closest_indices = df1[(df1['date_0']>=day0_shifted_back-t_expand)&(df1['date_0']<=day0_shifted_back+t_expand)].index
                            break

                filtered_curve_ref = closest_indices
                filtered_known_curves = [known_curves[i] for i in closest_indices]
                
                if mtx2 =='corr':
                    knn = best_curves_auto_corr((transformed_arr[:,1]), filtered_known_curves, 3) 
                else:
                    if mtx == 'euclidean':
                        knn = best_curves((transformed_arr[:,1]), filtered_known_curves, 3, metric='l2')
                    else:
                        knn = best_curves((transformed_arr[:,1]), filtered_known_curves, 3, metric='dtw')

                selection = np.argmin(knn[:,2])
                if (mtx=='corr' and -knn[selection][1]<=-1.1*knn[0,1]) or (knn[selection][1]>knn[0][1]*1.1) or len(filtered_curve_ref)<3:
                    selection=0
                bc = knn[selection]

                obj_id, fr0, fr1, A, B, M0, C, _, _, _,_ = saved_coefs[filtered_curve_ref[int(bc[0])]]
                C = np.max([0,C])
                M = M0+median_shift/1000+bc[2]/1000
                using_coef.append([obj_id,A,B,M,C])
                history_states.append(transformed_arr)
                predicted_states.append(sigmoid(np.arange(check_frame,check_frame+prediction_range+25)/1000, 100, B, M, C))
        #         plt.plot(predicted_states[-1])

        # plt.title(f'mnclustering_{mtx}-{mtx2}')
        # plt.savefig(f'2022-mnclustering_{mtx}-{mtx2}_1216.png')
        
        evaluations = []
        for i in range(len(actual_states)):
            evaluations.append(evaluate(actual_states[i], predicted_states[i]))
        
        all_evaluations.append(evaluations)
        # evaluation_green.append(evaluations)

        np.save(f'../results/2022-mnclustering_{mtx}-{mtx2}-1216.npy', np.stack(predicted_states))
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{mtx}-{mtx2} uses time: {elapsed_time} seconds")

print(len(all_evaluations))

mean_evalr = []
mean_evalg = []
mean_evala = []
for i in range(len(all_evaluations)):
    evr = all_evaluations[i][:12]
    evg = all_evaluations[i][12:]
    mean_evalr.append(np.nanmean(evr,axis=0))
    mean_evalg.append(np.nanmean(evg,axis=0))
    mean_evala.append(np.nanmean(all_evaluations[i],axis=0))

dfa = pd.DataFrame(mean_evala, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])
dfr = pd.DataFrame(mean_evalr, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])
dfg = pd.DataFrame(mean_evalg, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])

file_path = '../results/2022-mnclustering-1216.xlsx'

with pd.ExcelWriter(file_path) as writer:
    dfa.to_excel(writer, sheet_name='all', index=False)
    dfr.to_excel(writer, sheet_name='red', index=False)
    dfg.to_excel(writer, sheet_name='green', index=False)