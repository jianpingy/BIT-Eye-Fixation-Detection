import torch
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import warnings

#The 0.975 quantile of the chi-squared distribution.
chi2q=torch.tensor([5.02389,7.37776,9.34840,11.1433,12.8325,14.4494,
       16.0128,17.5346,19.0228,20.4831,21.920,23.337,
       24.736,26.119,27.488,28.845,30.191,31.526,32.852,34.170,
       35.479,36.781,38.076,39.364,40.646,41.923,43.194,44.461,
       45.722,46.979,48.232,49.481,50.725,51.966,53.203,54.437,
       55.668,56.896,58.120,59.342,60.561,61.777,62.990,64.201,
       65.410,66.617,67.821,69.022,70.222,71.420])
 
#Median of the chi-squared distribution. 
chimed=torch.tensor([0.454937,1.38629,2.36597,3.35670,4.35146,
       5.34812,6.34581,7.34412,8.34283,9.34182,10.34,11.34,12.34,
       13.34,14.34,15.34,16.34,17.34,18.34,19.34,20.34,21.34,22.34,
       23.34,24.34,25.34,26.34,27.34,28.34,29.34,30.34,31.34,32.34,
       33.34,34.34,35.34,36.34,37.34,38.34,39.34,40.34,41.34,42.34,
       43.34,44.34,45.34,46.34,47.33,48.33,49.33])

def BIT(T,XL,YL,XR,YR,alpha=0.25,x_min=0,x_max=1281,y_min=0,y_max=769):
    print('Preprocessing...')
    Duration,Velocity,T,Z = Preprocessing(T,XL,YL,XR,YR,x_min,x_max,y_min,y_max)
    print(' ')
    
    print('Plot velocities of left and right eyes...')
    plot_velocity(Velocity)
    print(' ')
    
    print('FastMCD Algorithm...')
    n = Velocity.shape[0]
    data = torch.transpose(Velocity,0,1)
    p = 4
    h = math.ceil(n*(1-alpha))
    Avg,Cov,H,m,ind = FastMCD(data,p,h,n)
    print(' ')
    
    print('Plot velocities selected by FastMCD...')
    plot_fixations(Velocity,H)
    print(' ')
    print('Plot x and y-coordinates against time...')
    plot_x_with_time(Z,T,H,ind,figsize=(20,8))
    print(' ')
    print('Plot left eye velocities with approximated control threshold derived by FastMCD...')
    plot_scatter(Avg[0:2],Cov[0:2,0:2],data[0:2],H[0:2],m) ##Left
    print(' ')
    print('Plot right eye velocities with approximated control threshold derived by FastMCD...')
    plot_scatter(Avg[2:],Cov[2:,2:],data[2:],H[2:],m) ##Right
    print(' ')
    
    print('Now return the times of fixations, saccades and blinks respectively...')
    
    return (T,Fixation_Saccade_Blink(np.sqrt(0.001),Z,data,Avg,Cov,n))

def Preprocessing(T,XL,YL,XR,YR,x_min,x_max,y_min,y_max):
    Z = [XL,YL,XR,YR]
    
    #Cleaning empty values
    mask_empty = 0
    for i,item in enumerate(Z):
        if i == 0:
            mask_empty += item != ' '
        else:
            mask_empty *= item != ' '
    for i in range(len(Z)):
        Z[i] = Z[i][mask_empty==1].astype(float)
    Time = T[mask_empty==1].astype(float)
    
    #Find coordinates in range
    mask_valid = 0
    for i,item in enumerate(Z):
        if i == 0 or i == 2:
            if i == 0:
                mask_valid += (item <= x_max)*(item >= x_min) 
            else: mask_valid *= (item <= x_max)*(item >= x_min)
        else:
            mask_valid *= (item <= y_max)*(item >= y_min)
    for i in range(len(Z)):
        Z[i] = Z[i][mask_valid==1].tolist()
    Time = torch.tensor(Time[mask_valid==1].tolist())
    Z = torch.transpose(torch.tensor(Z),0,1)
    Velocity = Z[1:]-Z[0:-1]
    Duration = Time[1:]-Time[0:-1]
    return (Duration,Velocity,Time,Z)

def chi_to_dist_threshold(chi,Cov):
    return torch.sqrt(2*torch.log(chi*torch.sqrt(torch.det(2*torch.pi*Cov)))).item()

def multivariate_control(chi,X_data,mean,Cov,Cov_inv,n): #n is the total number of observations
    threshold = chi_to_dist_threshold(chi,Cov)
    d = dist(X_data,mean,Cov_inv,n)
    return (d <= threshold, d > threshold)

def Fixation_Saccade_Blink(chi,Z,X_data,Avg,Cov,n):
    (ind_fix,ind_unsure) = multivariate_control(chi,X_data,Avg,Cov,torch.linalg.inv(Cov),n)
    indx_all = torch.arange(n)
    indx_fix = (indx_all[ind_fix]+1).tolist()
    indx_unsure = indx_all[ind_unsure]+1
    flag = False
    if indx_unsure[-1] == n+1:
        flag = True
        indx_unsure = indx_unsure[:-1]
    delta2 = Z[indx_unsure+1]-Z[indx_unsure-1]
    (ind_blink,ind_sac) = multivariate_control(chi,torch.transpose(delta2,0,1),Avg,Cov,torch.linalg.inv(Cov),delta2.shape[0])
    indx_sac = (indx_unsure[ind_sac]).tolist()
    indx_blink = (indx_unsure[ind_blink]).tolist()
    if flag:
        indx_blink.append(n+1)
    return (indx_fix,indx_sac,indx_blink)

def plot_velocity(Velocity):
    fig,ax = plt.subplots(1,2)
    ax[0].plot(Velocity[:,0],Velocity[:,1])
    ax[0].set_title('Left Eye')
    ax[1].plot(Velocity[:,2],Velocity[:,3])
    ax[1].set_title('Right Eye')
    plt.show()

def plot_fixations(Velocity,selected_datapoints):
    fig,ax = plt.subplots(1,2)
    ax[0].scatter(Velocity[:,0],Velocity[:,1],s=1.5)
    ax[0].scatter(selected_datapoints[0,:],selected_datapoints[1,:],color='red',s=1.5)
    ax[0].set_title('Left Eye')
    ax[1].scatter(Velocity[:,2],Velocity[:,3],s=1.5)
    ax[1].scatter(selected_datapoints[2,:],selected_datapoints[3,:],color='red',s=1.5)
    ax[1].set_title('Right Eye')
    plt.show()
    
def plot_x_with_time(Z,Time,selected_datapoints,ind,figsize=(15,7)):
    fig,ax = plt.subplots(1,2,figsize=figsize)
    ax[0].plot(Time,Z[:,0],label='Left Eye')
    ax[0].plot(Time,Z[:,2],label='Rigt Eye',color='green')
    ax[0].scatter(Time[ind],(Z[ind,:][:,0]+Z[ind,:][:,2])/2,color='red',s=3)
    ax[0].set_title('Eye x-coordinate')
    ax[0].set(ylabel='x-coordinate (pixels)')
    ax[0].legend()
    
    ax[1].plot(Time,Z[:,1],label='Left Eye')
    ax[1].plot(Time,Z[:,3],label='Right Eye',color='green')
    ax[1].scatter(Time[ind],(Z[ind,:][:,1]+Z[ind,:][:,3])/2,color='red',s=3)
    ax[1].set_title('Eye y-coordinate')
    ax[1].set(ylabel='y-coordinate (pixels)')
    ax[1].legend()

def dist(X,T,S_inv,n): ##Mahalanobis distance
    return torch.sqrt(torch.diag(torch.transpose(X-T.repeat(1,n),0,1)@S_inv@(X-T.repeat(1,n))))

def T_and_S_inv(H,p,h):
    T = torch.mean(H,1,True)
    S = (H-T.repeat(1,h)).matmul(torch.transpose(H-T.repeat(1,h),0,1))/h
    det = torch.det(S)
    S_inv = torch.linalg.inv(S) if det>0 else None
    return (T,S_inv,det,S)

#H_old is a matrix of size p*h, p is the dimension, h is the number of points
def Cstep(H_old,data,p,h,n,initial=False):
    if initial:
        _,temp = H_old.shape
    elif H_old.shape[1]<h:
        temp = H_old.shape[1]
    else:
        temp = h
    T_old,S_old_inv,detmt,S = T_and_S_inv(H_old,p,temp)
    if S_old_inv is not None:
        d_old = dist(data,T_old,S_old_inv,n)
        d,ind = torch.sort(d_old)
        return (data[:,ind[0:h]],detmt,S,d[h-1],ind[0:h])
    else:
        warnings.warn("Warning: Majority of data points lie on a hyperplane.")
        return (H_old,torch.tensor([0]),None,None,None)

def Initial_H(X,p,h,n):
    ind = torch.randint(low=0,high=n,size=(p+1,)).tolist()

    H = X[:,ind]
    temp = p+1
    _,_,detmt,_ = T_and_S_inv(H,p,temp)
    counter = 0
    while detmt < 1e-10:
        counter += 1
        if counter >= h:
            break
        new = torch.randint(low=0,high=n,size=(1,)).item()
        while new in ind:
            new = torch.randint(low=0,high=n,size=(1,)).item()
        ind.append(new)
        H = X[:,ind]
        temp += 1
        _,_,detmt,_ = T_and_S_inv(H,p,temp)
    if counter < h:
        H,detmt,_,_,_ = Cstep(H,X,p,h,n,True)
    return H,detmt

def Initial_Screening(data,p,h,n):
    H,_ = Initial_H(data,p,h,n)
    H,_,_,_,_ = Cstep(H,data,p,h,n)
    H,detmt,_,_,_ = Cstep(H,data,p,h,n)
    return H,detmt

def FastMCD_select(data,p,h,n,repetition=500,best_num=10):
    detmt0 = []
    H0 = []
    for _ in range(repetition):
        H,detmt = Initial_Screening(data,p,h,n)
        if detmt.item() == 0:
            warnings.warn("Warning: Majority of data points lie on a hyperplane.")
            return (H,0)
        H0.append(H)
        detmt0.append(detmt.item())
    indx = np.argsort(detmt0)
    H0 = [H0[indx[k]] for k in range(best_num)]
    detmt0 = [detmt0[indx[k]] for k in range(best_num)]
    return (H0,detmt0)

def FastMCD_find_best(H0,detmt0,data,p,h,n):
    if detmt0 == 0:
        return (None,H0,None,None,None)
    else:
        max_iter = 200
        detmt_best = 1e200

        for i,H in enumerate(H0):
            H_new = H
            detmt_pre = detmt0[i]
            counter = 0
            while True:
                counter += 1
                if counter > max_iter:
                    break
                H_new,detmt,S,m,ind = Cstep(H_new,data,p,h,n)
                diff = detmt_pre - detmt
                if np.abs(diff) < 1e-5:
                    break
                det_mt_pre = detmt
            if detmt.item() < detmt_best:
                H_best = H_new
                S_best = S
                m_best = m
                ind_best = ind
        T_MCD = torch.mean(H_best,1,True)
        
        d = dist(data,T_MCD,torch.linalg.inv(S_best),n)
        d,_ = torch.sort(d)
        d_med = torch.median(d)
        S_MCD = d_med/chimed[p-1]*S_best
        return (T_MCD,S_MCD,H_best,m_best,ind_best)

def FastMCD(data,p,h,n):
    if n <= 600:
        H0,detmt0 = FastMCD_select(data,p,h,n,repetition=500,best_num=10)
        return FastMCD_find_best(H0,detmt0,data,p,h,n)
    else:
        size_sub = 300
        for i in range(4):
            if n/(i+2) < 300:
                size_sub = math.ceil(n/(i+2))
                break
        num_of_subsets = 0
        subsets = []
        indx = np.arange(n)
        np.random.shuffle(indx)
        
        #Subsets
        head = 0
        while num_of_subsets < 5:
            num_of_subsets += 1
            if head+size_sub >= n:
                subsets.append(data[:,indx[head:]])
                size_sub_last = n-head
                head = n
                break
            else:
                subsets.append(data[:,indx[head:(head+size_sub)]])
                head += size_sub
                if num_of_subsets == 5:
                    size_sub_last = size_sub
        data_merged = data[:,indx[0:head]]

        #For each subset, repeat 100 times
        rep = 100
        best_num = 10
        Hs_best = []
        detmts_best = []
        n_merged = head
        h_merged = math.ceil(n_merged*h/n)
        
        for i,data_sub in enumerate(subsets):
            if i == num_of_subsets-1:
                H_sub,detmt_sub = FastMCD_select(data_sub,p,math.ceil(size_sub_last*h/n),size_sub_last,repetition=rep,best_num=best_num)
            else:
                H_sub,detmt_sub = FastMCD_select(data_sub,p,math.ceil(size_sub*h/n),size_sub,best_num=best_num)
            if detmt_sub == 0:
                return (None,H_sub,None,None,None)
            else:
                Hs_best += H_sub
                detmts_best += detmt_sub
        
        H_merged = []
        detmt_merged = []
        for i,H in enumerate(Hs_best):
            H_out,_,_,_,_ = Cstep(H,data_merged,p,h_merged,n_merged)
            H_out,detmt_out,_,_,_ = Cstep(H_out,data_merged,p,h_merged,n_merged)
            H_merged.append(H_out)
            detmt_merged.append(detmt_out)
        indx = np.argsort(detmt_merged)
        H_merged = [H_merged[indx[k]] for k in range(best_num)]
        detmt_merged = [detmt_merged[indx[k]] for k in range(best_num)]
        return FastMCD_find_best(H_merged,detmt_merged,data,p,h,n)

def Final_One_Step_Update(T_MCD,S_MCD,data,n):
    d = dist(data,T_MCD,torch.linalg.inv(S_MCD),n)
    weights = d <= torch.sqrt(chi2q[p-1]).item()
    sum_w = torch.sum(weights)

    T1 = torch.mean(data[:,weights],1,True)
    S1 = (data[:,weights]-T.repeat(1,sum_w)).matmul(torch.transpose(data[:,weights]-T.repeat(1,sum_w),0,1))/(sum_w-1)
    return (T1,S1)

def plot_scatter(T,Cov,X,H,m,draw_ellipse=True,xlabel='x1',ylabel='x2'):
    fig,ax = plt.subplots(1)
    U,S,_ = np.linalg.svd(Cov,hermitian=True)
    ax.scatter(X[0,:],X[1,:])
    ax.scatter(H[0,:],H[1,:],color='red')
    Step = torch.tensor(np.linspace(0,2*np.pi,50)).reshape(1,50)
    hor = torch.cos(Step)*np.sqrt(S[0])*m
    ver = torch.sin(Step)*np.sqrt(S[1])*m
    if draw_ellipse:
        Ellipse_Pts = torch.tensor(U,dtype=torch.float64)@torch.cat((hor,ver),0)+T
        ax.plot(Ellipse_Pts[0,:],Ellipse_Pts[1,:],color='black')
    ax.set_title('Tolerance Ellipse (97.5%)')
    ax.set(xlabel=xlabel)
    ax.set(ylabel=ylabel)
    plt.show()
