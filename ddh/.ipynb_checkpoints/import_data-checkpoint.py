import pandas as pd
import numpy as np
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.preprocessing.discretization import make_cuts



##### USER-DEFINED FUNCTIONS
def f_get_Normalization(X, norm_mode):    
    num_Patient, num_Feature = np.shape(X)
    
    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.nanstd(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.nanmean(X[:, j]))/np.nanstd(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.nanmean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.nanmin(X[:,j]))/(np.nanmax(X[:,j]) - np.nanmin(X[:,j]))
    else:
        print("INPUT MODE ERROR!")
    
    return X


def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
        mask3 is required to get the contional probability (to calculate the denominator part)
        mask3 size is [N, num_Event, num_Category]. 1's until the last measurement time
    '''
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category]) # for denominator
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0]+1)] = 1 # last measurement time

    return mask


def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss 
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            mask[i,int(label[i,0]-1),int(time[i,0])] = 1
        else: #label[i,2]==0: censored
            mask[i,:,int(time[i,0]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category]. 
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements 
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0]) # censoring/event time
            mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask



##### TRANSFORMING DATA
def f_construct_dataset(df, feat_list):
    '''
        id   : patient indicator
        tte  : time-to-event or time-to-censoring
            - must be synchronized based on the reference time
        times: time at which observations are measured
            - must be synchronized based on the reference time (i.e., times start from 0)
        label: event/censoring information
            - 0: censoring
            - 1: event type 1
            - 2: event type 2
            ...
    '''

    grouped  = df.groupby(['id'])
    id_list  = pd.unique(df['id'])
    max_meas = np.max(grouped.count())[0]

    data     = np.zeros([len(id_list), max_meas, len(feat_list)+1])
    pat_info = np.zeros([len(id_list), 7])

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)

        pat_info[i,4] = tmp.shape[0]             #number of measurement
        pat_info[i,3] = np.max(tmp['times'])     #last measurement time
        pat_info[i,2] = tmp['label'][0]      #cause
        pat_info[i,1] = tmp['tte'][0]         #time_to_event
        pat_info[i,0] = tmp['id'][0] 
        if 'tte_original' in df.columns:
            pat_info[i,5] = tmp['tte_original'][0]    #original time_to_event at first step
            pat_info[i,6] = tmp.time_left.min() #original time_to_event at last step
        else:
            pat_info[i,5] = None
            pat_info[i,6] = None
        

        data[i, :int(pat_info[i, 4]), 1:]  = tmp[feat_list]
        data[i, :int(pat_info[i, 4]-1), 0] = np.diff(tmp['times_original'])
    
    return pat_info, data


def import_dataset(file, data_mode='eeg'):
    df_       = pd.read_csv(file)
    df_['time_left'] = df_.tte - df_.times
    ## discretize time
    n_cuts = 120 # number of discretized durations, hyper-para
    all_labels = df_.groupby('id').mean()[['tte', 'label']]
    durations, events = np.array(all_labels.tte), np.array(all_labels.label)

    lower_cuts = np.arange(97) # per hour for the first 4 days or 96 hours, 0 to 96
    remaining_cuts = make_cuts(n_cuts=120-96, scheme='quantiles', durations=durations[durations>96], events=events[durations>96], min_=96)
    cuts = np.concatenate([lower_cuts, remaining_cuts[1:]])

    trans_discrete_time = LabTransDiscreteTime(cuts=cuts, scheme='quantiles')
    trans_discrete_time.fit(durations, events)

    trans_durations, trans_events = trans_discrete_time.transform(df_.tte, df_.label)
    df_['tte_discrete'] = trans_durations
    df_['times_to'] = df_.tte - df_.times
    trans_time_to, trans_events = trans_discrete_time.transform(df_.times_to, df_.label)
    df_['times_original'] = df_.times # the code require the col name to be speficific things, so back up the original 
    df_['tte_original'] = df_.tte # the code require the col name to be speficific things, so back up the original 
    df_['times_to_discrete'] = trans_time_to
    df_['times_discrete'] = df_.tte_discrete - df_.times_to_discrete
    df_.tte = df_.tte_discrete
    df_.times = df_.times_discrete
    
    num_eeg_feat = 72
    num_static_feat = 43
    if data_mode == 'eeg':
        feat_list = ['eeg_'+str(e+1) for e in range(num_eeg_feat)] + ['sta_'+str(s+1) for s in range(num_static_feat)]
    elif data_mode == 'sepsis':
        feat_list = ['ehr_'+str(e+1) for e in range(55)]
    df_       = df_[['id', 'tte', 'times', 'label', 'tte_original', 'time_left', 'times_original']+feat_list]
    df_org_            = df_.copy(deep=True)

    pat_info, data     = f_construct_dataset(df_, feat_list)
    _, data_org        = f_construct_dataset(df_org_, feat_list)

    data_mi                  = np.zeros(np.shape(data))
    data_mi[np.isnan(data)]  = 1
    data_org[np.isnan(data)] = 0
    data[np.isnan(data)]     = 0 

    x_dim           = np.shape(data)[2] # 1 + x_dim_cont + x_dim_bin (including delta)
    x_dim_cont      = 72 + 43 # these two number are not correct, but they sum up to the correct number x_dim
    x_dim_bin       = 0 # these two number are not correct, but they sum up to the correct number x_dim

    last_meas       = pat_info[:,[3]]  #pat_info[:, 3] contains age at the last measurement
    label           = pat_info[:,[2]]  #two competing risks
    time            = pat_info[:,[1]]  #time idx when event occurred (if using discretization)
    time_original   = pat_info[:,[5]]  #original time when event occurred
    time_to_last    = pat_info[:,[6]]  #original time when event occurred
    # num_Category    = int(np.max(pat_info[:, 1]) * 1.2) #or specifically define larger than the max tte
    num_Category    = int(np.max(pat_info[:, 1]) + 1)
    num_Event       = len(np.unique(label)) - 1

    if num_Event == 1:
        label[np.where(label!=0)] = 1 #make single risk

    mask1           = f_get_fc_mask1(last_meas, num_Event, num_Category)
    mask2           = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask3           = f_get_fc_mask3(time, -1, num_Category)

    DIM             = (x_dim, x_dim_cont, x_dim_bin)
    DATA            = (data, time, label, time_original, time_to_last)
    # DATA            = (data, data_org, time, label)
    MASK            = (mask1, mask2, mask3)

    return DIM, DATA, MASK, data_mi, trans_discrete_time