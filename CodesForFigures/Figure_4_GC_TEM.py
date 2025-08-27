import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.image as mpimg
from scipy.signal import convolve2d
import pandas as pd
import os
from scipy.optimize import curve_fit
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as mpatches
from tifffile import imread
from PIL import Image, ImageOps

label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

dir_names = ['nos_w_control_5hrs/','nos_EcadOE_5hrs/','nos_ecadGFP_5hrs/','nos_nosEcadGFP_5hrs/']

#==================== Loading data ==================#
df = pd.read_csv('Exp/GC_Dist_and_Exit_Info_v2.csv')
fraction_out = []
Dists_In = []
Dists_Out = []

ExitPerEmbryo = []
MeanDistPerEmbryo = []
MedianDistPerEmbryo = []

for dirname in dir_names:
    df_ = df[df.dirname == dirname]
    fraction_out.append(np.sum(df_.Exit)/df_.shape[0])

    Exit = np.array(df_.Exit)
    Dist = np.array(df_.Dist)
    D1 = Dist[Exit == 0]
    D2 = Dist[Exit == 1]
    Dists_In.append(D1)
    Dists_Out.append(D2)
    #---------------------#
    Filenames = np.unique(df_.filename)

    X = []
    Y = []
    Z = []
    for filename in Filenames:
        dff = df_[df_.filename == filename]
        f_out = np.sum(dff.Exit)/dff.shape[0]
        mean_dist = np.mean(dff.Dist)
        median_dist = np.median(dff.Dist)
        X.append(f_out)
        Y.append(mean_dist)
        Z.append(median_dist)
    ExitPerEmbryo.append(X)
    MeanDistPerEmbryo.append(Y)
    MedianDistPerEmbryo.append(Z)
#----------------------------------------------------------#
Groups = ['nos_w_control_5hrs/','nos_EcadOE_5hrs/', 'nos_ecadGFP_5hrs/','nos_nosEcadGFP_5hrs/']

Dic = {}

for group in Groups:
    dic_ = {}
    df_ = df[df.dirname == group]
    Exit = np.array(df_.Exit) 
    Dist = np.array(df_.Dist)
    dic_['In'] = Dist[Exit == 0]
    dic_['Out'] = Dist[Exit == 1]
    Dic[group] = dic_ 



G_ECadVals = eval(open('FigData/C_G_values.txt','r').read())
TauExit = np.load('Tau_exit_lambda_fit_data.npy')[4,1,:]
G_E = np.linspace(G_ECadVals[0],G_ECadVals[-1],200)



legend_string = 'I: WT; II: Ubi-Ecad-OE-A; III: Ubi-Ecad-OE-B; IV: GC-Ecad-OE'

fig,AX = plt.subplots(figsize=(7,9))
AX.axis(False)

# ax0 = AX.inset_axes([0.0,0.965,1,0.035])

# ax0.set_xlim(0,1)
# ax0.set_ylim(0,1)

# ax0.text(0.05,0.3,legend_string,fontsize=12)
# ax0.set_xticks([])
# ax0.set_yticks([])

ax_top = AX.inset_axes([0,0.76,1.0,0.24])
ax_top.axis(False)

# axa = AX.inset_axes([0.01,0.76,0.24,0.2])
# axb = AX.inset_axes([0.26,0.76,0.24,0.2])
# axc = AX.inset_axes([0.51,0.76,0.24,0.2])
# axd = AX.inset_axes([0.76,0.76,0.24,0.2])
# for ax_ in [axa,axb,axc,axd]:
#     ax_.set_xticks([])
#     ax_.set_yticks([])


ax1 = AX.inset_axes([0.05,0.35,0.45,0.4]) # each gc dist
ax2 = AX.inset_axes([0.6,0.35,0.4,0.4]) #mean dist per embryo
ax3 = AX.inset_axes([0.05,0.0,0.45,0.3]) #fraction exit per embryo
ax4 = AX.inset_axes([0.6,0.0,0.4,0.3]) #schematic explanation 
ax4.set_xticks([])
ax4.set_yticks([])


GCDist = [np.concatenate([Dic[group]['In'],Dic[group]['Out']]) for group in Groups]


gcvp = ax1.violinplot(GCDist,showextrema=False,showmeans=False,widths=0.8)
cnt = 0
for pc in gcvp['bodies'] :
    pc.set_facecolor('silver')
    pc.set_edgecolor('k')
    pc.set_linewidth(1)
    cnt += 1

cnt = 1
for group in Groups:
    Y1,Y2 = Dic[group]['In'],Dic[group]['Out']
    X1,X2 = np.random.normal(cnt,0.1,len(Y1)),np.random.normal(cnt,0.1,len(Y2))
    
    
    #ax1.plot(X1,Y1,'.',markerfacecolor='None',color='#1A85FF',markersize=4,alpha=0.6)
    #ax1.plot(X2,Y2,'.',markerfacecolor='None',color='#D41159',markersize=4,alpha=0.6)

    
    ax1.plot(X2,Y2,'+',color='k',markersize=4,alpha=0.6)
    ax1.plot(X1,Y1,'.',markerfacecolor='None',color='#6e6d6d',markersize=4,alpha=0.6)

    y_mean = np.mean(np.concatenate([Y1,Y2]))
   
    ax1.plot([cnt-0.3,cnt+0.3],[y_mean]*2,'-',markerfacecolor='None',color='k',markersize=8)
    cnt += 1


ax1.plot(X2,Y2,'+',color='k',markersize=4,alpha=0.6,label='Out')
ax1.plot(X1,Y1,'.',markerfacecolor='None',color='#6e6d6d',markersize=4,alpha=0.6,label='In')

ax1.legend(loc='best',fontsize=14,frameon=False,handletextpad=-0.5)


mvp = ax2.violinplot(MeanDistPerEmbryo,showextrema=False,showmeans=False,widths=0.8)
cnt = 0
for pc in mvp['bodies'] :
    pc.set_facecolor('silver')
    pc.set_edgecolor('k')
    pc.set_linewidth(1)
    cnt += 1

for i in range(4):
    Y = MeanDistPerEmbryo[i]
    X = np.random.normal(i+1,0.1,len(Y))
    ax2.plot(X,Y,'.',markerfacecolor='None',color='k',markersize=8)
    ax2.plot([i+0.7,i+1.3],[np.mean(Y)]*2,'-',markerfacecolor='None',color='k',markersize=8)


evp = ax3.violinplot(ExitPerEmbryo,showextrema=False,showmeans=False,widths=0.8)

cnt = 0
for pc in evp['bodies'] :
    pc.set_facecolor('silver')
    pc.set_edgecolor('k')
    pc.set_linewidth(1)
    cnt += 1

for i in range(4):
    Y = ExitPerEmbryo[i]
    X = np.random.normal(i+1,0.1,len(Y))
    ax3.plot(X,Y,'.',markerfacecolor='None',color='k',markersize=8)
    ax3.plot([i+0.7,i+1.3],[np.mean(Y)]*2,'-',markerfacecolor='None',color='k',markersize=8)



ax4.plot(G_E,TauExit,'--',lw=3,color='k',alpha=0.5)
ax4.set_xticks(range(0,11,2))
ax4.set_xticklabels(['']*6)
# ax4.set_yticks([0,np.max(PExit)])
# ax4.set_yticklabels(['0','1'])
ax4.axvspan(2,2.5, facecolor='silver', alpha=0.3,label='WT')
# ax4.axvspan(3.5,4.0, facecolor='#05FE04', alpha=0.3,label=r'Ecad$\uparrow$')
# ax4.axvspan(0.5,1.0, facecolor='#D71B60', alpha=0.3,label=r'Ecad$\downarrow$')
ax4.axvspan(3.5,4.0, facecolor='#05FE04', alpha=0.3,label='More E-cad')
ax4.axvspan(0.5,1.0, facecolor='#D71B60', alpha=0.3,label='Less E-cad')

ax4.set_xlabel('$C_G$',fontsize=18,labelpad=-5)
ax4.set_ylabel(r'$\tau_{e}$',fontsize=18,labelpad=5)
leg = ax4.legend(fontsize=12,frameon=False,loc=(0.4,0.7),handletextpad=0.2,labelspacing=0.0)
for text in leg.get_texts():
    text.set_color('k')

# RAX = [axa,axb,axc,axd]
# RepImgs = [mpimg.imread('FigData/nos_w.tif'),mpimg.imread('FigData/nos_Ecad.tif'),mpimg.imread('FigData/nos_EcadGFP.tif'),mpimg.imread('FigData/nos_nosEcadGFP.tif')]
# for i in range(len(RAX)):
#     ax_ = RAX[i]
#     ax_.imshow(RepImgs[i])

top_panel = mpimg.imread('FigData/Fig4_top_SG.png')
ax_top.imshow(top_panel)




ax1.set_xticks([1,2,3,4])
ax1.set_xticklabels(['I','II','III','IV'])

ax2.set_xticks([1,2,3,4])
ax2.set_xticklabels(['I','II','III','IV'])

ax3.set_xticks([1,2,3,4])
ax3.set_xticklabels(['I','II','III','IV'])

ax1.set_yticks([0,40,80,120])
ax2.set_yticks([15,30,45,60])
ax3.set_yticks([0,0.5,1.0])

ax1.set_ylabel('$d_{gc}$',fontsize=18,labelpad=-10)

ax2.set_ylabel('$\overline{d_{gc}}$',fontsize=18,labelpad=-12)
ax2.yaxis.label.set_position((-0.1, 0.45)) # Adjust x and y coordinates
ax3.set_ylabel('$f_{e}$',fontsize=18,labelpad=-5)
ax3.yaxis.label.set_position((-0.1, 0.7)) # Adjust x and y coordinates


ax1.text(0.5,110,'(b)',fontsize=18)
ax2.text(0.5,58,'(c)',fontsize=18)
ax3.text(0.5,0.9,'(d)',fontsize=18)

ax4.text(0.03,9930,'(e)',fontsize=18)
plt.savefig('Fig3_v2.png',bbox_inches='tight',dpi=600)
plt.savefig('Fig3_v2.svg',bbox_inches='tight',dpi=1200)
plt.show()


