import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
#===========================================================#


dfgc = pd.read_csv('Exp/TimelapseTEM/Results_TEM_PGC.csv')
dfc = pd.read_csv('Exp/TimelapseTEM/Results_TEM1_center.csv')


TIDS = [1,9,16,21,26]#5,45,80,140,190

Distance_from_center = {} #germ cell distances during 5 different time points

for tid in TIDS:
    
    dfgc_ = dfgc[dfgc.Frame == tid]
    x_gc,y_gc = np.array(dfgc_.X_),np.array(dfgc_.Y_)

    dfc_ = dfc[dfc.Frame == tid]
    cent_x,cent_y = np.array(dfc_.X_)[0],np.array(dfc_.Y_)[0]
    
    Distance_from_center[str(tid)] = np.sqrt((x_gc - cent_x)**2 + (y_gc - cent_y)**2)
#----------------------------------------------------------------------#
LC = np.load('Exp/TimelapseTEM/Timelapse_lateral_center_pos.npy')
DC = np.load('Exp/TimelapseTEM/Timelapse_dorsal_center_pos.npy')

df_gclat = pd.read_csv('Exp/TimelapseTEM/Timelapse_lateral_gc_tracks.csv')

df_gcdor = pd.read_csv('Exp/TimelapseTEM/Timelapse_dorsal_gc_tracks.csv')

df_lat_state = pd.read_csv('Exp/TimelapseTEM/TL_TEM_lateral_inout.csv')

df_dor_state = pd.read_csv('Exp/TimelapseTEM/TL_TEM_dorsal_inout.csv')
        
CIDS_dorsal  = list(df_dor_state.cell)
CIDS_lateral = list(df_lat_state.cell)

Time_to_exist = np.array([6,7,6,6,5,5,6,5,7,6]) #time to exit violinplot 

df1 = pd.read_csv('FigData/Results_ROI_nospacer_1-5.csv')
df2 = pd.read_csv('FigData/Results_ROI_nospacer_7-10.csv')

Values = np.array(list(df1.Mean) + list(df2.Mean)) 

ECad_GCEC, ECad_ECEC_lat,ECad_ECEC_api = Values[0::3],Values[1::3],Values[2::3]
#====================================================================#
dflb = pd.read_csv('Exp/TimelapseTEM/Frames/Results_TEM_lumenboundary_v2.csv')
# Img = imread('Exp/TimelapseTEM/Frames/TEM_26_12.tif')
YLIMS = [[250,750],[220,720],[220,720]]

x = np.array(dflb[dflb.Frame == 26].X)
y = np.array(dflb[dflb.Frame == 26].Y)

# fig,ax = plt.subplots()
# ax.imshow(Img,cmap='gray')
# ax.plot(x,y,'.',markerfacecolor='None')
# ax.set_xlim(0,500)
# ax.set_ylim(200,700)
# plt.show()
#=============== plot axes ==========#
fig,AX = plt.subplots(figsize=(7,7),tight_layout=True)
AX.axis(False)
#---------------------------------------------------------#
ax0  = AX.inset_axes([0.0,0.79,1.0,0.21])
ax0.set_xticks([])
ax0.set_yticks([])
ax0.axis(False)
schem = mpimg.imread('FigData/Fig1_schematic.png')
ax0.imshow(schem)

ax0.text(20,200,r'$(\text{a})$',fontsize=18)
ax0.text(1600,200,r'$(\text{a}^\prime)$',fontsize=18)
ax0.text(3200,200,r'$(\text{a}^{\prime\prime})$',fontsize=18)
#----------------------------------------------------------#


ax1a = AX.inset_axes([0.0, 0.54, 0.13, 0.26])
ax1b = AX.inset_axes([0.0, 0.27, 0.13, 0.26])
ax1c = AX.inset_axes([0.0, 0.0, 0.13, 0.26])

ax2a = AX.inset_axes([0.19,0.24,0.47,0.21])
ax2b = AX.inset_axes([0.19,0.02,0.47,0.21])


ax3 = AX.inset_axes([0.19,0.55,0.47,0.25])

ax5 = AX.inset_axes([0.74,0.02,0.25,0.12])

ax6 = AX.inset_axes([0.68,0.15,0.32,0.32])
ax7 = AX.inset_axes([0.68,0.48,0.32,0.32])

ax6.set_xticks([])
ax6.set_yticks([])

ax7.set_xticks([])
ax7.set_yticks([])




#--------- Ecadherin levels ----------------------------------#
ECadLevels = [ECad_ECEC_api,ECad_ECEC_lat,ECad_GCEC]
VP  = ax5.violinplot(ECadLevels,orientation='horizontal',showextrema=False,showmeans=False,widths=0.8)
ax5.set_xticks([500,1000,1500,2000,2500])
ax5.set_xticklabels(['500','','1500','','2500'])

for pc in VP['bodies'] :
    #pc.set_facecolor(Colors[cnt])
    pc.set_facecolor('#a19f9f')
    pc.set_edgecolor('k')
    pc.set_linewidth(1)
for j in range(3):
    X = ECadLevels[j]
    Y = np.random.normal(j+1,0.1,len(X))
    ax5.plot(X,Y,'.',color='k',markerfacecolor='None')
# ax5.yaxis.tick_right()
# ax5.yaxis.set_label_position("right")
ax5.set_yticks([1,2,3])
ax5.set_yticklabels(['E$_{api}$','E$_{lat}$','EG'],fontsize=14)

for label in ax5.get_yticklabels():
    label.set_horizontalalignment('left')  # Left-align the text
    label.set_x(-0.2)  # Adjust position leftward (tweak value as needed)

ax5.set_xlabel('$\mathbf{I}_{Ecad}$',fontsize=16,labelpad=0)

#----------- distance with time -------------------------------#
DistFromCenter  = [Distance_from_center[str(j)] for j in TIDS]

vp = ax3.violinplot(DistFromCenter,showextrema=False,showmeans=False,widths=0.8)

cnt = 0
for pc in vp['bodies'] :
    #pc.set_facecolor(Colors[cnt])
    pc.set_facecolor('#a19f9f')
    pc.set_edgecolor('k')
    pc.set_linewidth(1)
    cnt += 1

for j in range(5):
    Y = DistFromCenter[j]
    X = np.random.normal(j+1,0.1,len(Y))
    ax3.plot(X,Y,'.',color='k',markerfacecolor='None')
ax3.set_xticks([1,2,3,4,5])
ax3.set_xticklabels(['5','45','80','140','190'])
ax3.set_yticks([0,25,50])
ax3.set_yticklabels(['0','','50'])
#ax3.yaxis.label.set_position((0.1,0.7)) # Adjust x and y coordinates
ax3.set_ylabel('$d_{gc}$',fontsize=18,rotation=90,labelpad=-20)
ax3.set_xlabel(r'$t\,(\text{min})$',fontsize=18,rotation=0,labelpad=2)
#ax3.yaxis.label.set_position((0.1,0.65)) # Adjust x and y coordinates
#------------------------ GCs in the midgut --------------------#
gcax = [ax1a,ax1b,ax1c]
IDS = [1,16,26]
dT = [5,90,190]
for m in range(len(IDS)):
    #Img = imread('Exp/TimelapseTEM/Frames/TEM_'+str(IDS[m])+'.tif')
    Img_ = Image.open('Exp/TimelapseTEM/Frames/TEM_'+str(IDS[m])+'.tif').convert('L')
    Img = ImageOps.invert(Img_)
    
    x = np.array(dflb[dflb.Frame == IDS[m]].X)
    y = np.array(dflb[dflb.Frame == IDS[m]].Y)
    Img = np.array(Img,dtype=np.uint16)
    print('shape = ',Img.shape)
    gcax[m].imshow(Img[:,0:512],cmap='gray')
    gcax[m].plot(x,y,'.',color='k',alpha=0.5,markeredgecolor='None',markersize=4)
    gcax[m].set_xticks([])
    gcax[m].set_yticks([])
    gcax[m].text(10,900,str(dT[m])+' min',fontsize=14)
gcax[0].plot([400,448],[950,950],lw=3,color='k')

#----------------------------- germ cells in the gut -----------------------#
img_ = mpimg.imread('FigData/Ecad_Vasa.tif')
img_ecad = mpimg.imread('FigData/Ecad_only_v2.tif')
ax6.imshow(img_ecad)
ax6.plot([450,512],[530,530],lw=3,color='k')
#--------------- distance from init position with time ---------------------#
ax7.imshow(img_)
ax7.plot([450,512],[530,530],lw=3,color='w')
ax7.text(0,540,'E-cad',color='m',fontsize=14)
ax7.text(150,540,'Vasa',color='#7ae82c',fontsize=14)
#---------------------------------------------------------------------------#

for cid in CIDS_dorsal:
    tmp = df_dor_state[df_dor_state.cell == cid]
    tin,tout = list(tmp.frame_in)[0],list(tmp.frame_out)[0]
    df_ = df_gcdor[df_gcdor.Label == cid]
    T  = np.array(df_.Frame)
    idx_ = np.argsort(T)
    X,Y,Z = np.array(df_.X_),np.array(df_.Y_),np.array(df_.Z_)
    T,X,Y,Z = T[idx_],X[idx_],Y[idx_],Z[idx_]
    tvals = []
    dvals = []
    ipos = [X[0],Y[0],Z[0]]
    
    for j in range(len(T)):
        t = T[j]
        cpos = DC[t,:]
        
        d = np.sqrt((ipos[0] - X[j])**2 + (ipos[1] - Y[j])**2 + (ipos[2] - Z[j])**2)
        tvals.append(t)
        dvals.append(d)
        if t < tin:
            #ax2a.plot(t,d,'o',alpha=0.7,color='#D81B60')
            mp1, = ax2a.plot(t,d,'o',alpha=0.7,color='k')
            t1 = t 
            d1 = d
        elif (~np.isnan(tout)) and (t >= tout) :
            #ax2a.plot(t,d,'o',alpha=0.7,color='#004D40')
            mp2, = ax2a.plot(t,d,'s',alpha=0.7,color='k')
            t2 = t 
            d2 = d
        else:
            #ax2a.plot(t,d,'o',alpha=0.7,color='#1E88E5')
            mp3, = ax2a.plot(t,d,'d',alpha=0.7,color='k')
            t3 = t 
            d3 = d
    
    ax2a.plot(tvals,dvals,'-',color='k',lw=0.5)
ax2a.plot(t1,d1,'o',alpha=0.7,color='k',label='in')
ax2a.plot(t2,d2,'s',alpha=0.7,color='k',label='out')
ax2a.plot(t3,d3,'d',alpha=0.7,color='k',label='transit')
ax2a.legend(loc='best',ncols=3,columnspacing=0.0,handletextpad=0.0,borderaxespad=0,frameon=False)
ax2a.set_ylabel('$\Delta d_{gc}$',fontsize=18,labelpad=-20)

#---------------------------------------#
for cid in CIDS_lateral:
    tmp = df_lat_state[df_lat_state.cell == cid]
    tin,tout = list(tmp.frame_in)[0],list(tmp.frame_out)[0]
    df_ = df_gclat[df_gclat.Label == cid]
    T  = np.array(df_.Frame)
    idx_ = np.argsort(T)
    X,Y,Z = np.array(df_.X_),np.array(df_.Y_),np.array(df_.Z_)
    T,X,Y,Z = T[idx_],X[idx_],Y[idx_],Z[idx_]
    tvals = []
    dvals = []
    ipos = [X[0],Y[0],Z[0]]
    
    for j in range(len(T)):
        t = T[j]

        cpos = LC[t,:]
        
        d = np.sqrt((ipos[0] - X[j])**2 + (ipos[1] - Y[j])**2 + (ipos[2] - Z[j])**2)
        tvals.append(t)
        dvals.append(d)
        print(t,tout)

        if t < tin:
            #ax2b.plot(t,d,'o',alpha=0.7,color='#D81B60')
            ax2b.plot(t,d,'o',alpha=0.7,color='k')
        elif (~np.isnan(tout)) and (t >= tout) :
            #ax2b.plot(t,d,'o',alpha=0.7,color='#004D40')
            ax2b.plot(t,d,'s',alpha=0.7,color='k')
        else:
            #ax2b.plot(t,d,'o',alpha=0.7,color='#1E88E5')
            ax2b.plot(t,d,'d',alpha=0.7,color='k')
    ax2b.plot(tvals,dvals,'-',color='k',lw=0.5)
ax2b.set_ylabel('$\Delta d_{gc}$',fontsize=18,labelpad=-20)
 

ax2a.set_xticks([0,4,8,12])
ax2b.set_xticks([0,4,8,12])
ax2a.set_yticks([0,30])
ax2b.set_yticks([0,25])
ax2a.set_xticklabels(['','','',''])
ax2b.set_xticklabels(['0','20','40','60']) #5mins
ax2a.set_yticklabels(['0','30'])
ax2b.set_yticklabels(['0','25'])



x_ = np.random.normal(1,0.2,len(Time_to_exist))

x_ = np.ones(len(Time_to_exist))

x_[Time_to_exist == 6] = np.linspace(0.8,1.2,np.sum(Time_to_exist == 6))
x_[Time_to_exist == 5] = np.linspace(0.9,1.1,np.sum(Time_to_exist == 5))
x_[Time_to_exist == 7] = np.linspace(0.9,1.1,np.sum(Time_to_exist == 7))


ax2b.set_xlabel(r'$t\,(\text{min})$',fontsize=18,labelpad=0)

#------fig labels ---------#

ax1a.text(0,150,r'$(\text{b})$',fontsize=18)
ax1b.text(0,150,r'$(\text{b}^\prime)$',fontsize=18)
ax1c.text(0,150,r'$(\text{b}^{\prime\prime})$',fontsize=18)


ax3.text(0.5,50,'(c)',fontsize=18)
ax2a.text(-0.5,28,'(d)',fontsize=18)
ax2b.text(-0.5,22,r'$(\text{d}^\prime)$',fontsize=18)

ax7.text(440,80,'(e)',fontsize=18,color='w')
ax6.text(440,80,r'$(\text{e}^\prime)$',fontsize=18)
ax5.text(2100,2.5,r'$(\text{e}^{\prime\prime})$',fontsize=18)

plt.savefig('Fig1_v2.png',bbox_inches='tight',dpi=600)
plt.savefig('Fig1_v2.svg',bbox_inches='tight',dpi=1200)
plt.show()



#Edit the midgut image to improve resolution, draw box around the 3 different interfaces
#color bars and channel labels
#Frames 1-14 is 5 mins and 15-27 is 10mins - relevant to (c)