import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
from scipy.signal import convolve2d
import pandas as pd
import os
from scipy.optimize import curve_fit
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from skimage.segmentation import find_boundaries
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

cmap = plt.get_cmap('plasma')

Colors_35 = cmap(np.linspace(0,1,36))

cmap_seq = plt.get_cmap('managua')

Colors_seq = cmap_seq(np.linspace(0,1,8))

Colors_disc = ['#E69F00','#009E73','#0072B2','#2f4f4f','#8b4513','#000080']
#==================================================================#
def get_cell_boundaries(mask_array):
    """
    Given a labeled mask array where each cell has a unique integer label,
    return a binary array where boundary pixels are 1 and others are 0.
    """
    boundaries = find_boundaries(mask_array, mode='outer').astype(np.uint8)
    return boundaries
#------------------------------------------------------------------#
def get_cell_and_perimeter(D):
    masks,Ctypes = D
    B = get_cell_boundaries(masks)
    print(np.min(B),np.max(B))
    Cells = np.array(Ctypes,dtype=int)
    Cells[B == 1] = -1 
    return(Cells)
#------------------------------------------------------------------#
values = [-1, 0, 1, 2, 3]
colors = ['black', 'white', 'white', '#7BB27E', '#FFC107']  # one for each value

# Create colormap and normalization
cmap_ = ListedColormap(colors)
norm_ = BoundaryNorm(boundaries=[-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], ncolors=len(colors))



D1 = np.load('GC_TEM_Frames/GC_TEM_lambda_7000_CE_10_CG_5_T_10_frame_1.npy')#100

D2 = np.load('GC_TEM_Frames/GC_TEM_lambda_7000_CE_10_CG_5_T_10_frame_2.npy')#200

D3 = np.load('GC_TEM_Frames/GC_TEM_lambda_7000_CE_10_CG_5_T_10_frame_7.npy')#700

D4 = np.load('GC_TEM_Frames/GC_TEM_lambda_7000_CE_10_CG_5_T_10_frame_8.npy')#800

Cells1 = get_cell_and_perimeter(D1)[30:140,30:140]
Cells2 = get_cell_and_perimeter(D2)[90:140,80:130]
Cells3 = get_cell_and_perimeter(D3)[90:140,90:140]
Cells4 = get_cell_and_perimeter(D4)[90:140,90:140]
#=================================================#

X  = np.load('Output/Exit_Array_GC_TEM_diffTE.npy')
TE = np.arange(0,56,2)
Lambdas = np.array([6000,7000,8000])

tau_exit = np.zeros((28,3))
p_exit   = np.zeros((28,3))

nvals = 1000
TE_fit       = np.linspace(0,54,nvals)
tau_exit_fit = np.zeros((nvals,3))
p_exit_fit   = np.zeros((nvals,3))

for j in range(3):
    for i in range(28):
        x_ = X[i,j,:]
        N  = np.sum(~np.isnan(x_))
        p_exit[i,j] = np.nansum(x_ < 9900)/N 
        tau_exit[i,j] = np.nanmean(x_)
    kr = KernelReg(p_exit[:,j],TE,'c')
    y_pred,y_std = kr.fit(TE_fit)
    p_exit_fit[:,j] = y_pred

    kr = KernelReg(tau_exit[:,j],TE,'c')
    y_pred,y_std = kr.fit(TE_fit)
    tau_exit_fit[:,j] = y_pred


#------------------------------------------------------#
fig,AX = plt.subplots(figsize=(7,3),tight_layout=True)
AX.axis(False)
axa = AX.inset_axes([0.0,0.1,0.42,0.9])
axb3 = AX.inset_axes([0.38,0.0,0.19,0.325])
axb2 = AX.inset_axes([0.38,0.335,0.19,0.325])
axb1 = AX.inset_axes([0.38,0.67,0.19,0.325])
axacb = AX.inset_axes([0.03,0.04,0.36,0.05])


axc2 = AX.inset_axes([0.67,0.10,0.33,0.36])
axc1 = AX.inset_axes([0.67,0.52,0.33,0.36])

axc_ = AX.inset_axes([0.67,0.9,0.33,0.1])
axc_.set_xticks([])
axc_.set_yticks([])


#----------------------------------------------#
CF = np.load('ChemicalField.npy')[30:140,30:140]

nY,nX = CF.shape 
x = np.arange(nX)
y = np.arange(nY)
X, Y = np.meshgrid(x, y)

step = 5

X,Y = X[::step,::step],Y[::step,::step]
CF = CF[::step,::step]

dCy, dCx = np.gradient(CF)   # note order: y first

axa.matshow(Cells1,origin='lower',cmap=cmap_,norm=norm_)

CMAP = plt.cm.magma_r
Q = axa.quiver(X, Y, dCx, dCy, CF, cmap=CMAP)
fig.colorbar(Q,cax=axacb,orientation="horizontal")
axacb.set_ylabel('M',fontsize=18,rotation=0)
axacb.yaxis.set_label_coords(-0.05, -0.4)
axacb.set_xticks([0,0.2,0.4,0.6])

axb1.matshow(Cells2,origin='lower',cmap=cmap_,norm=norm_)
axb2.matshow(Cells3,origin='lower',cmap=cmap_,norm=norm_)
axb3.matshow(Cells4,origin='lower',cmap=cmap_,norm=norm_)


axb1.text(30,40,'200',fontsize=10)
axb2.text(30,40,'700',fontsize=10)
axb3.text(30,40,'800',fontsize=10)

axa.text(65,102,'t = 100 mcs',fontsize=12)

for ax_ in [axa,axb1,axb2,axb3]:
    ax_.set_xticks([])
    ax_.set_yticks([])


mp1, = axc1.plot(TE,tau_exit[:,0],'o',markerfacecolor='None',markersize=4,color='k',label='6000')
mp2, = axc1.plot(TE,tau_exit[:,1],'s',markerfacecolor='None',markersize=4,color='k',label='7000')
mp3, = axc1.plot(TE,tau_exit[:,2],'d',markerfacecolor='None',markersize=4,color='k',label='8000')

line1, = axc1.plot(TE_fit,tau_exit_fit[:,0],'-',markerfacecolor='None',color='k',label='6000')
line2, = axc1.plot(TE_fit,tau_exit_fit[:,1],'-',markerfacecolor='None',color='k',label='7000')
line3, = axc1.plot(TE_fit,tau_exit_fit[:,2],'-',markerfacecolor='None',color='k',label='8000')


axc2.plot(TE,p_exit[:,0],'o',markerfacecolor='None',markersize=4,color='k')
axc2.plot(TE,p_exit[:,1],'s',markerfacecolor='None',markersize=4,color='k')
axc2.plot(TE,p_exit[:,2],'d',markerfacecolor='None',markersize=4,color='k')

axc2.plot(TE_fit,p_exit_fit[:,0],'-',markerfacecolor='None',color='k')
axc2.plot(TE_fit,p_exit_fit[:,1],'-',markerfacecolor='None',color='k')
axc2.plot(TE_fit,p_exit_fit[:,2],'-',markerfacecolor='None',color='k')

axc1.set_xticks([0,25,50])
axc2.set_xticks([0,25,50])

axc1.set_xticklabels(['','',''])
axc2.set_xticklabels(['0','25','50'])

axc1.set_yticks([4000,6000,8000,10000])
axc2.set_yticks([0.0,0.5,1.0])
axc1.set_yticklabels(['4000','','','10000'])

axc1.set_ylabel(r'$\tau_{e}$',fontsize=18)
axc2.set_ylabel('$p_{e}$',fontsize=18)
axc1.yaxis.set_label_coords(-0.18, 0.5)
axc2.yaxis.set_label_coords(-0.18, 0.5)


axc2.set_xlabel(r'$\text{T}_{E}$',fontsize=18)
axc2.xaxis.set_label_coords(0.7, -0.1)

axc_.legend(handles=[mp1, mp2, mp3], ncols=3,handletextpad=0.2,columnspacing=0.5,fontsize=10,frameon=False,loc='center')
axc_.set_ylabel('$\lambda_{M}$',fontsize=14,rotation=0)
axc_.yaxis.set_label_coords(-0.1, 0.3)


axc1.set_xlim(-8,54)
axc2.set_xlim(-8,54)

axc1.axvline(10,linestyle='-',color='silver',lw=5,alpha=0.5)
axc2.axvline(10,linestyle='-',color='silver',lw=5,alpha=0.5)

#-------- fig labels -----------#
axa.text(0,100,'(a)',fontsize=18)
axb1.text(-0.3,3,'(b)',fontsize=14)
axb2.text(-0.3,3,'(b$^{\prime}$)',fontsize=14)
axb3.text(-0.3,3,'(b$^{\prime\prime}$)',fontsize=14)

axc1.text(-8,8500,'(c)',fontsize=18)
axc2.text(-8,0.78,'(c$^{\prime}$)',fontsize=18)
# plt.savefig('Figure_2_v2.svg',bbox_inches='tight',dpi=600)
# plt.savefig('Figure_2_v2.png',bbox_inches='tight',dpi=600)
plt.show()




