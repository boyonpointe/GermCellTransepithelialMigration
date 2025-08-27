from cc3d.core.PySteppables import *
import numpy as np
import pandas as pd

class GC_TEM_AdFlexSteppable(SteppableBasePy):

    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self,frequency)
    
    def save_cellfield(self,fname):
        W,H = self.dim.x,self.dim.y 
        CF = np.zeros((2,W,H),dtype=np.uint8)
        for i in range(W):
            for j in range(H):
                cell = self.cellField[i,j,0]
                try:
                    CF[0,i,j] = int(cell.id)
                    CF[1,i,j] = int(cell.type)
                except:
                    print('')
        np.save(fname,CF)

    def start(self):
        """
        Called before MCS=0 while building the initial simulation
        """
        self.randseed = 5 #parameter 7
        self.chem_lambda = 4000 #parameter 2 1000-10000
        self.E_ECad = 10.0 #parameter 3 - fix it at 5
        self.G_ECad = 5.0 #parameter 4 - vary between 10^(-1,1)
        
        
        np.random.seed(self.randseed)
        
        #----- create a germ cell off center -------#
        germCell = self.new_cell(self.G)
        gc_offset = 10
        L2 = 85
        
        rand_theta = 2 * np.pi * np.random.rand()
        x_gc,y_gc = round(L2 + gc_offset*np.cos(rand_theta)), round(L2 + gc_offset*np.sin(rand_theta))
        self.cell_field[y_gc-5:y_gc+5,x_gc-5:x_gc+5,0] = germCell
        
        germCell.targetVolume = germCell.volume
        
        cd = self.chemotaxisPlugin.addChemotaxisData(germCell, "CF")
        cd.setLambda(self.chem_lambda)
        
        self.adhesionFlexPlugin.setAdhesionMoleculeDensity(germCell,"ECad",self.G_ECad)
        for cell in self.cell_list_by_type(self.E):
            self.adhesionFlexPlugin.setAdhesionMoleculeDensity(cell,"ECad",self.E_ECad)
            
        MAX = int(1e5)
        self.gc_wc_com = np.zeros((MAX,5),dtype=int)

    def step(self, mcs):
        """
        Called every frequency MCS while executing the simulation
        
        :param mcs: current Monte Carlo step
        """
        pos = [0,0,0,0,0]
        
        # if mcs%10 == 0:
            # self.save_cellfield('/Users/athena/Documents/GC_TEM/Output/SuccessFrames/frame_'+str(mcs)+'.npy')
            
        for cell in self.cell_list_by_type(self.G,self.W):
            if cell.type == self.G:
                pos[0] = cell.xCOM
                pos[1] = cell.yCOM
                try:
                    nbr = [gcnbr[0].type for gcnbr in self.get_cell_neighbor_data_list(cell)]
                    if 2 in nbr:
                        pos[4] = 1
                except:
                    pos[4] = 0
                    
            elif cell.type == self.W:
                pos[2] = cell.xCOM
                pos[3] = cell.yCOM
        #------- terminate when the gc has crossed the barrier ----#
        dist_from_start = np.sqrt((pos[2]-pos[0])**2 + (pos[3]-pos[1])**2)
        if dist_from_start > 70 :
            self.stop_simulation()
        #-----------------------------------------------------------#
        self.gc_wc_com[mcs,:] = pos     
        
            
        

        
    def finish(self):
        """
        Called after the last MCS to wrap up the simulation
        """
        POS = self.gc_wc_com[0:self.mcs,:]
        PD = pd.DataFrame({'gcX':POS[:,0],'gcY':POS[:,1],'wcX':POS[:,2],'wcY':POS[:,3],'nbrE':POS[:,4]})
        #PD.to_csv('/Users/athena/Desktop/GC_TEM_lambda_'+str(self.chem_lambda)+'_E_ECad_'+str(self.E_ECad)+'_G_ECad_'+str(self.G_ECad)+'_iter_'+str(self.randseed)+'.csv',sep=',')
        #PD.to_csv('/n/home03/chandrashekar/CC3DWorkspace/GC_TEM_AFlex/batch_IDX/GC_TEM_lambda_'+str(self.chem_lambda)+'_E_ECad_'+str(self.E_ECad)+'_G_ECad_'+str(self.G_ECad)+'_iter_'+str(self.randseed)+'.csv',sep=',')
        

    def on_stop(self):
        """
        Called if the simulation is stopped before the last MCS
        """
        POS = self.gc_wc_com[0:self.mcs,:]
        PD = pd.DataFrame({'gcX':POS[:,0],'gcY':POS[:,1],'wcX':POS[:,2],'wcY':POS[:,3],'nbrE':POS[:,4]})
        #PD.to_csv('/Users/athena/Desktop/GC_TEM_lambda_'+str(self.chem_lambda)+'_E_ECad_'+str(self.E_ECad)+'_G_ECad_'+str(self.G_ECad)+'_iter_'+str(self.randseed)+'.csv',sep=',')
        #PD.to_csv('/n/home03/chandrashekar/CC3DWorkspace/GC_TEM_AFlex/batch_IDX/GC_TEM_lambda_'+str(self.chem_lambda)+'_E_ECad_'+str(self.E_ECad)+'_G_ECad_'+str(self.G_ECad)+'_iter_'+str(self.randseed)+'.csv',sep=',')