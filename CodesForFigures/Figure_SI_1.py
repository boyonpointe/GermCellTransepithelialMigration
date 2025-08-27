import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import convolve2d
import pandas as pd
import os
from scipy.optimize import curve_fit
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as mpatches
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

cmap = plt.get_cmap('plasma')

Colors_35 = cmap(np.linspace(0,1,36))

cmap_seq = plt.get_cmap('managua')

Colors_seq = cmap_seq(np.linspace(0,1,8))

Colors_disc = ['#E69F00','#009E73','#0072B2','#2f4f4f','#8b4513','#000080']
#==================================================================#
def quadratic(x,a,b,c):
    y = a + b*x + c*x*x 
    return(y)

def fit_quadratic(x,y):
    popt,_ = curve_fit(quadratic,x,y)
    a,b,c = popt
    x_fit = np.linspace(np.min(x),np.max(x),1000)
    y_fit = quadratic(x_fit,a,b,c)
    return(x_fit,y_fit,popt)
#------------------------------ loading all the data ------------------------#

G_ECadVals = np.array(eval(open('FigData/C_G_values.txt','r').read()))
LambdaMs = np.array([4000,4500,5000,5500,6000,6500,7000])

D1 = np.load('Output/GC_TEM_4b_AdFlex_Exit_Array.npy')
D2 = np.load('Output/GC_TEM_4c_AdFlex_Exit_Array.npy')

D_lambda = np.zeros((7,3,50,1000))
D_lambda[:,0:2,:,:] = D1
D_lambda[:,2:3,:,:] = D2 * 10


D_success_mean = np.zeros((7,3,50))
D_success_std = np.zeros((7,3,50))

for i in range(7):
    for j in range(3):
        for k in range(50):
            data = D_lambda[i,j,k,:]
            data_ = data[data < 9990]
            if len(data_) > 0:
                D_success_mean[i,j,k] = np.nanmean(data_)
                D_success_std[i,j,k] = np.nanstd(data_)
            else:
                D_success_mean[i,j,k] = np.nan
                D_success_std[i,j,k] = np.nan


G_E = np.linspace(G_ECadVals[0],G_ECadVals[-1],100)

Sigma,Mu = np.nanstd(D_lambda[:,1,:,:],axis=2),np.nanmean(D_lambda[:,1,:,:],axis=2)

Sigma_fit = np.zeros((7,len(G_E)))
Mu_fit = np.zeros((7,len(G_E)))


Sigma_fit_success = np.zeros((7,len(G_E)))
Mu_fit_success = np.zeros((7,len(G_E)))


for i in range(7):
    kr = KernelReg(Mu[i,:],G_ECadVals,'c')
    Mu_fit[i,:] = kr.fit(G_E)[0]
    kr_ = KernelReg(Sigma[i,:],G_ECadVals,'c')
    Sigma_fit[i,:] = kr_.fit(G_E)[0]
    
    x = D_success_mean[i,1,:]
    y = D_success_std[i,1,:]
    idx_ = np.nonzero(np.isfinite(x))[0]
    
    x_ = x[idx_]
    y_ = y[idx_]
    z_ = G_ECadVals[idx_]
    if len(idx_) > 1:

        Kr  = KernelReg(x_,z_,'c')
        Kr_ = KernelReg(y_,z_,'c')
        Mu_fit_success[i,:] = Kr.fit(G_E)[0]
        Sigma_fit_success[i,:] = Kr_.fit(G_E)[0]
    else:
        Mu_fit_success[i,:] = np.nan
        Sigma_fit_success[i,:] = np.nan



fig,AX = plt.subplots(figsize=(4,6),tight_layout=True)
AX.axis(False)
axa = AX.inset_axes([0.2,0.0,0.8,0.48])
axb = AX.inset_axes([0.2,0.52,0.8,0.48])

# for i in range(7):
#     axa.plot(G_ECadVals,Sigma[i,:],'.',color=Colors_seq[i],markerfacecolor='None')
#     axa.plot(G_E,Sigma_fit[i,:],'-',color=Colors_seq[i])
#     axb.plot(G_ECadVals,Mu[i,:],'.',color=Colors_seq[i],markerfacecolor='None')
#     axb.plot(G_E,Mu_fit[i,:],'-',color=Colors_seq[i])
# axa.set_xticks([0,2,4,6,8,10])
# axb.set_xticks([0,2,4,6,8,10])
# axa.set_xticklabels([str(k) for k in [0,2,4,6,8,10] ])
# axb.set_xticklabels(['']*6)
# axa.set_ylabel(r'$\sigma_{\tau_e}$',fontsize=18,rotation=0,labelpad=10)
# axb.set_ylabel(r'$\tau_e$',fontsize=18,rotation=0,labelpad=10)
# axa.set_xlabel('$C_G$',fontsize=18)
# # plt.savefig('FigSI_1.png',bbox_inches='tight',dpi=600)
# # plt.savefig('FigSI_1.svg',bbox_inches='tight',dpi=600)
# plt.show()



for i in range(7):
    axa.plot(G_ECadVals,D_success_std[i,1,:],'.',color=Colors_seq[i],markerfacecolor='None')
    #axa.plot(G_E,Sigma_fit_success[i,:],'-',color=Colors_seq[i])
    axb.plot(G_ECadVals,D_success_mean[i,1,:],'.',color=Colors_seq[i],markerfacecolor='None')
    #axb.plot(G_E,Mu_fit_success[i,:],'-',color=Colors_seq[i])
axa.set_xticks([0,2,4,6,8,10])
axb.set_xticks([0,2,4,6,8,10])
axa.set_xticklabels([str(k) for k in [0,2,4,6,8,10] ])
axb.set_xticklabels(['']*6)
axa.set_ylabel(r'$\sigma_{\tau_e}$',fontsize=18,rotation=0,labelpad=10)
axb.set_ylabel(r'$\tau_e$',fontsize=18,rotation=0,labelpad=10)
axa.set_xlabel('$C_G$',fontsize=18)
# plt.savefig('FigSI_1.png',bbox_inches='tight',dpi=600)
# plt.savefig('FigSI_1.svg',bbox_inches='tight',dpi=600)
plt.show()











