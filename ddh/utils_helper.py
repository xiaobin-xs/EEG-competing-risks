'''
Largely adapted from the original Tensorflow implementation of Dynamic DeepHit: https://github.com/chl8856/Dynamic-DeepHit
Changed code from Tensorflow into PyTorch version
'''

import numpy as np
import random
import torch
import torch.nn.functional as F

_EPSILON = 1e-08

##### USER-DEFINED FUNCTIONS
def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
        mask1 is required to get the contional probability (to calculate the denominator part)
        mask1 size is [N, num_Event, num_Category]. 1's until the last measurement time
    '''
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category]) # for denominator
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0]+1)] = 1 # last measurement time

    return mask


def f_get_minibatch(mb_size, x, x_mi, label, time, mask1, mask2, mask3):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb     = x[idx, :, :].astype(float)
    x_mi_mb  = x_mi[idx, :, :].astype(float)
    k_mb     = label[idx, :].astype(float) # censoring(0)/event(1,2,..) label
    t_mb     = time[idx, :].astype(float)
    m1_mb    = mask1[idx, :, :].astype(float) #fc_mask
    m2_mb    = mask2[idx, :, :].astype(float) #fc_mask
    m3_mb    = mask3[idx, :].astype(float) #fc_mask
    return x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb


def f_get_boosted_trainset(x, x_mi, time, label, mask1, mask2, mask3):
    _, num_Event, num_Category  = np.shape(mask1)  # dim of mask3: [subj, Num_Event, Num_Category]
    meas_time = np.concatenate([np.zeros([np.shape(x)[0], 1]), np.cumsum(x[:, :, 0], axis=1)[:, :-1]], axis=1)

    total_sample = 0
    for i in range(np.shape(x)[0]):
        total_sample += np.sum(np.sum(x[i], axis=1) != 0)

    new_label          = np.zeros([total_sample, np.shape(label)[1]])
    new_time           = np.zeros([total_sample, np.shape(time)[1]])
    new_x              = np.zeros([total_sample, np.shape(x)[1], np.shape(x)[2]])
    new_x_mi           = np.zeros([total_sample, np.shape(x_mi)[1], np.shape(x_mi)[2]])
    new_mask1          = np.zeros([total_sample, np.shape(mask1)[1], np.shape(mask1)[2]])
    new_mask2          = np.zeros([total_sample, np.shape(mask2)[1], np.shape(mask2)[2]])
    new_mask3          = np.zeros([total_sample, np.shape(mask3)[1]])

    tmp_idx = 0
    for i in range(np.shape(x)[0]):
        max_meas = np.sum(np.sum(x[i], axis=1) != 0)

        for t in range(max_meas):
            new_label[tmp_idx+t, 0] = label[i,0]
            new_time[tmp_idx+t, 0]  = time[i,0]

            new_x[tmp_idx+t,:(t+1), :] = x[i,:(t+1), :]
            new_x_mi[tmp_idx+t,:(t+1), :] = x_mi[i,:(t+1), :]

            new_mask1[tmp_idx+t, :, :] = f_get_fc_mask1(meas_time[i,t].reshape([-1,1]), num_Event, num_Category) #age at the measurement
            new_mask2[tmp_idx+t, :, :] = mask2[i, :, :]
            new_mask3[tmp_idx+t, :]    = mask3[i, :]

        tmp_idx += max_meas
        
    return(new_x, new_x_mi, new_time, new_label, new_mask1, new_mask2, new_mask3)


def loss_Log_Likelihood(out, k, fc_mask1, fc_mask2):
    '''
    Modified based on the corresponding part in the TensorFlow implementation
    out: output from the cause-specific network
    k: event indicator. 0 for censored, 1 for event 1, 2 for event 2, and so on
    '''    
    I_1 = torch.sign(k)
    denom = 1 - torch.sum(torch.sum(fc_mask1 * out, dim=2), dim=1, keepdim=True) # make subject specific denom.
    denom = torch.clamp(denom, _EPSILON, 1. - _EPSILON)
    
    #for uncenosred: log P(T=t,K=k|x,Y,t>t_M)
    tmp1 = torch.sum(torch.sum(fc_mask2 * out, dim=2), dim=1, keepdim=True)
    tmp1 = I_1 * torch.log(tmp1+_EPSILON / denom)

    #for censored: log \sum P(T>t|x,Y,t>t_M)
    tmp2 = torch.sum(torch.sum(fc_mask2 * out, dim=2), dim=1, keepdim=True)
    tmp2 = (1. - I_1) * torch.log(tmp2+_EPSILON / denom)

    loss1 = - torch.mean(tmp1 + tmp2)
    return loss1


def loss_Ranking(out, t, k, fc_mask3, num_Event, num_Category, sigma=0.1):
    '''
    Modified based on the corresponding part in the TensorFlow implementation
    '''
    if out.shape[0] == 1:
        return 0
    eta = []
    for e in range(num_Event):
        one_vector = torch.ones_like(t, dtype=torch.float32)
        I_2 = torch.eq(k, e+1).to(torch.float32) #indicator for event
        I_2 = torch.diag(torch.squeeze(I_2))
        tmp_e = out[:, e, :].view(-1, num_Category) #event specific joint prob.

        R = torch.matmul(tmp_e, fc_mask3.transpose(0, 1)) #no need to divide by each individual dominator
        # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

        diag_R = R.diag().reshape(-1, 1)
        R = torch.matmul(one_vector, diag_R.transpose(0, 1)) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
        R = R.transpose(0, 1)                                 # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

        T = F.relu(torch.sign(torch.matmul(one_vector, t.transpose(0, 1)) - torch.matmul(t, one_vector.transpose(0, 1))))
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

        T = torch.matmul(I_2, T) # only remains T_{ij}=1 when event occured for subject i

        tmp_eta = torch.mean(T * torch.exp(-R/sigma), dim=1, keepdim=True)

        eta.append(tmp_eta)
    eta = torch.stack(eta, dim=1) #stack referenced on subjects
    eta = torch.mean(eta.reshape(-1, num_Event), dim=1, keepdim=True)

    loss2 = torch.sum(eta) #sum over num_Events
    return loss2


def loss_RNN_Prediction(longitudinal_prediction, x, d=0):
    '''
    Modified based on the corresponding part in the PyTorch implementation
    '''
    length = (~(torch.abs(x).sum(2) == 0)).sum(axis = 1) - 1 # padded output are filled with zeros
    if x.is_cuda:
        device = x.get_device()
    else:
        device = torch.device("cpu")


    # Create a grid of the column index
    index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device) 

    # Select all predictions until the last observed
    prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))

    # Select all observations that can be predicted
    observation_mask = index <= length.unsqueeze(1).repeat(1, x.size(1))
    observation_mask[:, 0] = False # Remove first observation
    
    return torch.nn.MSELoss(reduction = 'mean')(longitudinal_prediction[prediction_mask], x[observation_mask])


def _f_get_pred(model, data, pred_horizon, device):
    '''
        predictions based on the prediction time.
        create new_data and new_mask2 that are available previous or equal to the prediction time (no future measurements are used)
    '''
    new_data  = np.zeros(np.shape(data))
    meas_time = np.concatenate([np.zeros([np.shape(data)[0], 1]), np.cumsum(data[:, :, 0], axis=1)[:, :-1]], axis=1)
    
    for i in range(data.shape[0]):
        last_meas = np.sum(meas_time[i, :] <= pred_horizon)
        new_data[i, :last_meas, :] = data[i, :last_meas, :]
    new_data = torch.from_numpy(new_data).float().to(device)
    
    model.eval()
    with torch.no_grad():
        _, out = model(new_data)
    out = torch.concat([o.unsqueeze(1) for o in out], 1) # (B, num_Event, num_Category)
    out = out.cpu().numpy()

    return out


def f_get_risk_predictions(model, data_, pred_time, eval_time, device):
    
    pred = _f_get_pred(model, data_[[0]], 0, device)
    _, num_Event, num_Category = pred.shape
       
    risk_all = {}
    for k in range(num_Event):
        risk_all[k] = np.zeros([data_.shape[0], len(pred_time), len(eval_time)])
            
    for p, p_time in enumerate(pred_time):
        ### PREDICTION
        pred_horizon = int(p_time)
        pred = _f_get_pred(model, data_, pred_horizon, device)


        for t, t_time in enumerate(eval_time):
            eval_horizon = int(t_time) + pred_horizon #if eval_horizon >= num_Category, output the maximum...

            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(pred[:,:,pred_horizon:(eval_horizon+1)], axis=2) #risk score until eval_time
            risk = risk / (np.sum(np.sum(pred[:,:,pred_horizon:], axis=2), axis=1, keepdims=True) +_EPSILON) #conditioniong on t > t_pred
            
            for k in range(num_Event):
                risk_all[k][:, p, t] = risk[:, k]
                
    return risk_all


def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)


def load_checkpoint(load_path, model, optimizer, device):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']