#%%
#-----------------------complete from start
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
evaluation_all = []

for mtx in ['corr','l2']:#,'lbk-24','lbk','lbk-6','lbk-12','dtw','softdtw']:
    start_time = time.time()
    tryout_votes = []
    tryout_shifts = []
    tryout_starts = []
    knns = []
    if len(track_states[track_states[:,1]==1])>0:
        for obj_id in track_states[track_states[:,1]==1][:,0]:
            arr = copy.copy(cr_gt_gsd[(cr_gt_gsd[:,1]==obj_id)&(cr_gt_gsd[:,19]>l_thre)])
            arr[:,20] = arr[:,20]*scaler_a + scaler_b
            history_fr0.append(frame_dict_val[arr[0,0]])
            transformed_arr = filter_history_observations_s(arr, check_frame=check_frame, history_range=history_range)      

            if mtx =='corr':
                knn = best_curves_auto_corr((transformed_arr[:,1]), known_curves, 5)
            else:
                knn = best_curves((transformed_arr[:,1]), known_curves, 5, metric=mtx)

            knns.append(knn)

            if mtx=='corr':
                for ii, bc in enumerate(knn):
                    if -bc[1]<=-1.1*knn[0,1]:
                        obj_id, fr0, fr1, A, B, M0, C, _, _, _,_ = saved_coefs[int(bc[0])]
                        tryout_votes.append(M0*1000+bc[2])
                        tryout_shifts.append(bc[2])
                        tryout_starts.append(fr0)
            else:
                for ii, bc in enumerate(knn):
                    if bc[1]<=1.25*knn[0,1]:
                        obj_id, fr0, fr1, A, B, M0, C, _, _, _,_ = saved_coefs[int(bc[0])]
                        tryout_votes.append(M0*1000+bc[2])
                        tryout_shifts.append(bc[2])
                        tryout_starts.append(fr0)

    votes_all.append(tryout_votes)
    shifts_all.append(tryout_shifts)
    starts_all.append(tryout_starts)
    knns_all.append(knns)

# evaluation_all = []
# check_tracks_sorted = []
# for mi,mtx in enumerate(['corr','l2','dtw','softdtw','lbk','lbk-6']):#
    # plt.figure()
    # tryout_shifts = shifts_all[mi]
    # tryout_votes = votes_all[mi]
    # tryout_starts = starts_all[mi]
    # knns = knns_all[mi]

    actual_states = []
    predicted_states = []

    median_vote = np.median(tryout_shifts)
    sorted_indices = np.argsort(tryout_votes)
    sorted_arr = arr[sorted_indices]
    mid1 = len(tryout_votes) // 2 - 1
    median_shift = tryout_shifts[sorted_indices[mid1]]
    median_shifted_start = tryout_starts[mid1]

    history_fr0 = []
    history_states = []
    using_coef_dates = []
    using_coef = []
    # #-----------------------growing
    if len(track_states[track_states[:,1]==1])>0:
        for oi, obj_id in enumerate(track_states[track_states[:,1]==1][:,0]):
            check_tracks_sorted.append(obj_id)
            # plt.figure()
            arr = copy.copy(cr_gt_gsd[(cr_gt_gsd[:,1]==obj_id)&(cr_gt_gsd[:,19]>l_thre)])
            arr[:,20] = arr[:,20]*scaler_a + scaler_b
            history_fr0.append(frame_dict_val[arr[0,0]])
            
            transformed_arr = filter_history_observations_s(arr, check_frame=check_frame, history_range=history_range)
            future_arr = normalize_future_observations(arr, check_frame=check_frame)
            actual_states.append(future_arr)

            # knn = best_curves((transformed_arr[:,1]), known_curves, 3)
            knn = knns[oi]

            vote_diff = []
            for ii, bc in enumerate(knn):
                obj_id, fr0, fr1, A, B, M0, C, _, _, _,_ = saved_coefs[int(bc[0])]
                vote_diff.append(np.abs(M0*1000+bc[2]-median_vote))

            selection = np.argmin(vote_diff)

            if (mtx=='corr' and -knn[selection][1]<=-1.1*knn[0,1]) or (knn[selection][1]>knn[0][1]*1.25):
                selection=0
            bc = knn[selection]
            
            obj_id, fr0, fr1, A, B, M0, C, _, _, _,_ = saved_coefs[int(bc[0])]
            C = np.max([0,C])
            M = M0-(fr0+bc[2])/1000+transformed_arr[0,0]/1000
            using_coef_dates.append((fr0-12)//24)

            using_coef.append([obj_id,A,B,M,C])
            history_states.append(transformed_arr)
            predicted_states.append(sigmoid(np.arange(check_frame,check_frame+prediction_range+25)/1000, 100, B, M, C))
            # plt.plot(predicted_states[-1])

    # #-----------------------all green
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
            
            if mtx =='corr':
                knn = best_curves_auto_corr((transformed_arr[:,1]), filtered_known_curves, 3) 
            else:
                knn = best_curves((transformed_arr[:,1]), filtered_known_curves, 3, metric=mtx)

            selection = np.argmin(knn[:,2])
            # if knn[selection][1]>knn[0][1]*1.1:
            if (mtx=='corr' and -knn[selection][1]<=-1.1*knn[0,1]) or (knn[selection][1]>knn[0][1]*1.1):
                selection=0
            bc = knn[selection]

            obj_id, fr0, fr1, A, B, M0, C, _, _, _,_ = saved_coefs[filtered_curve_ref[int(bc[0])]]
            C = np.max([0,C])
            M = M0+median_shift/1000+bc[2]/1000
            using_coef.append([obj_id,A,B,M,C])
            history_states.append(transformed_arr)
            predicted_states.append(sigmoid(np.arange(check_frame,check_frame+prediction_range+25)/1000, 100, B, M, C))
            # plt.plot(predicted_states[-1])
        
    # plt.title(mtx)
    # plt.savefig(f'2022-{mtx}_.png')

    np.save(f'{mtx}_prediction.npy', np.stack(predicted_states))

    evaluations = []
    for i in range(len(actual_states)):
        evaluations.append(evaluate(actual_states[i], predicted_states[i]))
    evaluation_all.append(evaluations)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{mtx} uses time: {elapsed_time} seconds")

# np.save("../results/2022-votes_all-1209_.npy", np.array(votes_all))
# np.save("../results/2022-shifts_all-1209_.npy", np.array(shifts_all))
# np.save("../results/2022-starts_all-1209_.npy", np.array(starts_all))
# np.save("../results/2022-knns_all-1209_.npy", np.array(knns_all))

# %%
mean_evalr = []
mean_evalg = []
mean_evala = []
for i in range(len(evaluation_all)):
    evr = evaluation_all[i][:12]
    evg = evaluation_all[i][12:]
    mean_evalr.append(np.nanmean(evr,axis=0))
    mean_evalg.append(np.nanmean(evg,axis=0))
    mean_evala.append(np.nanmean(evaluation_all[i],axis=0))

dfa = pd.DataFrame(mean_evala, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])
dfr = pd.DataFrame(mean_evalr, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])
dfg = pd.DataFrame(mean_evalg, columns=['me','l1','l2','lcss','ctw','dtw','softdtw','autocorr','gak','lbk','lbk6','lbk12','frdist','edr'])

file_path = '../results/2022-pred-1209_.xlsx'

with pd.ExcelWriter(file_path) as writer:
    dfa.to_excel(writer, sheet_name='all', index=False)
    dfr.to_excel(writer, sheet_name='red', index=False)
    dfg.to_excel(writer, sheet_name='green', index=False)