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

G_ECadVals = eval(open('FigData/C_G_values.txt','r').read())
LambdaMs = np.array([4000,4500,5000,5500,6000,6500,7000])

D1 = np.load('Output/GC_TEM_4b_AdFlex_Exit_Array.npy')
D2 = np.load('Output/GC_TEM_4c_AdFlex_Exit_Array.npy')

D_lambda = np.zeros((7,3,50,1000))
D_lambda[:,0:2,:,:] = D1
D_lambda[:,2:3,:,:] = D2 * 10

P_Success_lambda = np.zeros(D_lambda.shape[:-1])
G_E = np.linspace(G_ECadVals[0],G_ECadVals[-1],100)

P_Success_lambda_fit = np.zeros((D_lambda.shape[0],D_lambda.shape[1],G_E.size))


for i in range(P_Success_lambda.shape[0]):
    for j in range(P_Success_lambda.shape[1]):
        for k in range(P_Success_lambda.shape[2]):
            Y = D_lambda[i,j,k,:] 
            denom = np.sum(~np.isnan(Y))
            P_Success_lambda[i,j,k] = np.nansum(Y < 9900)/denom
        kr = KernelReg(P_Success_lambda[i,j,:],G_ECadVals,'c')
        y_pred,y_std = kr.fit(G_E)
        P_Success_lambda_fit[i,j,:] = y_pred

#np.save('P_Success_lambda_fit_data.npy',P_Success_lambda_fit)



Traj = dict(np.load('Traj_cg_id_0_to_35.npz',allow_pickle=True))

time_points = np.linspace(0,1,Traj['0'].shape[1])

Traj_mean = np.zeros((35,1000))
for i in range(35):
    Traj_mean[i,:] = np.mean(Traj[str(i)],axis=0)
Traj_mean = (Traj_mean - 30)/30

tau_second_half = np.zeros(35)
for i in range(35):
    tau_second_half[i] = (1000 - np.sum(Traj_mean[i,:] <= 0.5))/1000

T_journey = np.nanmean(D_lambda,axis=3)

PS,TJ = P_Success_lambda.flatten(),T_journey.flatten()

x_fit,y_fit,params = fit_quadratic(PS,TJ)

CV = np.nanstd(D_lambda[:,1,:,:],axis=2)/np.nanmean(D_lambda[:,1,:,:],axis=2)

CV_fit = np.zeros((7,len(G_E)))
for i in range(7):
    kr = KernelReg(CV[i,:],G_ECadVals,'c')
    CV_fit[i,:] = kr.fit(G_E)[0]


GC_Pos = dict(np.load('GC_Pos_ee_10_ge_4.8979591836734695.npz',allow_pickle=True))

#=================================================================#
fig,AX = plt.subplots(figsize=(7,7),tight_layout=True)
AX.axis(False)
axa3 = AX.inset_axes([0.05, 0.0, 0.33, 0.3])
axa2 = AX.inset_axes([0.05, 0.3, 0.33, 0.3])
axa1 = AX.inset_axes([0.05, 0.6, 0.33, 0.3])

axb = AX.inset_axes([0.45,0.0,0.45,0.28])

axb_cb = AX.inset_axes([0.92,0.0,0.06,0.28])
#axb_cb.axis(False)

cmap_CG = mpl.colors.ListedColormap(Colors_35)
bounds_CG = np.linspace(0,7.2,36)
norm_CG = mpl.colors.BoundaryNorm(bounds_CG, cmap_CG.N)

CG_bar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_CG,norm=norm_CG),cax=axb_cb,orientation='vertical')

CG_bar.ax.set_yticks([0,3.5,7.0])
CG_bar.ax.set_yticklabels(['0.0','3.5','7.0'],rotation=0,fontsize=14)
CG_bar.ax.set_xlabel('$C_G$',fontsize=18)

axc1 = AX.inset_axes([0.4,0.3,0.38,0.29])




axc2 = AX.inset_axes([0.45,0.65,0.32,0.25])

axd1 = AX.inset_axes([0.8,0.35,0.2,0.24])
axd2 = AX.inset_axes([0.8,0.65,0.2,0.25])

#------- main colorbar --------#

axcb = AX.inset_axes([0.05,0.92,0.95,0.03])

cmap_lambda = mpl.colors.ListedColormap(Colors_seq)
bounds = np.arange(4000,8000,500)
norm = mpl.colors.BoundaryNorm(bounds, cmap_lambda.N+1)

lamb_bar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_lambda,norm=norm),cax=axcb,orientation='horizontal',aspect=200)
lamb_bar.ax.set_xticks([4250,4750,5250,5750,6250,6750,7250])
lamb_bar.ax.set_xticklabels(['4000','4500','5000','5500','6000','6500','7000'],rotation=0,fontsize=12)
lamb_bar.ax.xaxis.tick_top()
lamb_bar.ax.set_ylabel('$\lambda_C$',fontsize=18,rotation=0,labelpad=20)


#------------------------- mid right ------------------------------------#
axd1.plot(PS,TJ,'.',color='k',markerfacecolor='None')
axd1.plot(x_fit,y_fit,'-',color='#d46515',lw=2)
axd1.set_xticks([0,0.8])
axd1.set_xlabel('$p_{exit}$',fontsize=18,labelpad=-15)
axd1.set_yticks([6000,8000,10000])
axd1.set_yticklabels(['6','8','10'])
axd1.yaxis.tick_right()
axd1.set_ylabel('$t (10^3 mcs)$',rotation=270,fontsize=18,labelpad=15)
axd1.yaxis.set_label_position("right")

#------------------------- top right ------------------------------------#
for j in range(35):
    axd2.plot(G_ECadVals[j],tau_second_half[j],'o',markerfacecolor=Colors_35[j,:],markeredgecolor='k')
axd2.set_xticks([0,7])
axd2.set_xlabel('$C_G$',fontsize=18,labelpad=-15)
axd2.set_ylabel(r'$\tau_2$',fontsize=18,rotation=270,labelpad=0)
axd2.yaxis.set_label_position("right")
axd2.set_yticks([0.0,0.5])
axd2.yaxis.tick_right()
axd2.set_facecolor('#b1b2b3')

#------------------------ bottom right ---------------------------------#
axb.fill_between(time_points,0.5,1,color='#b1b2b3',alpha=1.0)

for j in range(35):
    axb.plot(time_points,Traj_mean[j,:],'-',color=Colors_35[j,:])

axb.set_yticks([0,0.5,1])
axb.set_yticklabels(['0','','1'])
axb.set_ylabel('$\mathsf{l}_e$',fontsize=18,labelpad=-10)
axb.set_xlim(0,1)
axb.set_ylim(0,1)
axb.set_xticks([0,0.5,1.0])
axb.set_xticklabels(['0','','1'])
axb.set_xlabel(r'$\tau$',fontsize=18,labelpad=-10)


#---------------------- mid top -----------------------------------------#
for i in range(7):
    axc2.plot(G_ECadVals,CV[i,:],'.',color=Colors_seq[i],markerfacecolor='None')
    axc2.plot(G_E,CV_fit[i,:],'-',color=Colors_seq[i])
axc2.set_xticks([0,5,10])
axc2.set_xticklabels(['0','','10'])
axc2.set_xlabel('$C_G$',fontsize=18,labelpad=-15)
axc2.set_yticks([0.0,0.1,0.2,0.3])
axc2.set_yticklabels(['0.0','','','0.3'])
axc2.set_ylabel('$CV_t$',fontsize=18,labelpad=-25)

#-------------------- left column ------------------#
fig_a_ax = [axa1,axa2,axa3]

for j in range(3):
    for i in range(7):
        fig_a_ax[j].plot(G_ECadVals,P_Success_lambda[i,j,:],'.',markerfacecolor='None',color=Colors_seq[i])
        fig_a_ax[j].plot(G_E,P_Success_lambda_fit[i,j,:],'-',color=Colors_seq[i],lw=3)
        
axa1.set_xticklabels([])
axa2.set_xticklabels([])

axa3.set_yticks([0.0,0.05,0.1])
axa2.set_yticks([0.0,0.25,0.5])
axa2.set_yticklabels(['0.0','','0.5'])
axa1.set_yticks([0.0,0.4,0.8])
axa2.set_ylabel(r'$p_{exit}$',fontsize=18,labelpad=-10)

axa3.set_xticks([0,3,5,7,10])
axa3.set_xticklabels(['0','3','','7','10'])
axa3.set_xlabel('$C_G$',fontsize=18,labelpad=-10)

#------- mid mid -------------#

n_tracks = len(GC_Pos)

for j in range(n_tracks):
    pos = GC_Pos[str(j)]
    x,y = pos[0,::10],pos[1,::10]
    if pos.shape[1] < 9990:
        axc1.plot(x,y,color='#C0C0C0',lw=0.3)
    else:
        axc1.plot(x,y,color='#B87333',lw=0.3)

r1,r2 = 30,45
thetas = np.linspace(-np.pi,np.pi)
cx,cy = 84,84

x1,y1 = cx + r1*np.cos(thetas),cy + r1*np.sin(thetas)
x2,y2 = cx + r2*np.cos(thetas),cy + r2*np.sin(thetas)
axc1.plot(x1,y1,'-',color='k')
axc1.plot(x2,y2,'-',color='k')

for j in range(20):
    theta1 = j * 2 * np.pi/20 
    xa,ya = cx + r1 * np.cos(theta1), cy + r1 * np.sin(theta1)
    xb,yb = cx + r2 * np.cos(theta1), cy + r2 * np.sin(theta1)
    axc1.plot([xa,xb],[ya,yb],'-',color='k',alpha=0.3)


axc1.set_xlim(35,135)
axc1.set_ylim(35,135)
axc1.set_xticks([])
axc1.set_yticks([])


axa1.text(0,0.72,r'($\text{a}$)',fontsize=18)
axa2.text(0,0.47,r'($\text{a}^\prime$)',fontsize=18)
axa3.text(0,0.1,r'($\text{a}^{\prime\prime}$)',fontsize=18)

axc2.text(-0.2,0.28,'(b)',fontsize=18)

axc1.text(35,120,'(c)',fontsize=18)

axb.text(0.0,0.85,'(d)',fontsize=18)

axd2.text(0.0,0.45,r'($\text{d}^\prime$)',fontsize=18)

axd1.text(0.5,9500,'(e)',fontsize=18)

#plt.savefig('Fig2_v0.png',bbox_inches='tight',dpi=300)
plt.show()
