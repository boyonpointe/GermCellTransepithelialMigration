
from cc3d import CompuCellSetup
        

from GC_TEM_AdFlexSteppables import GC_TEM_AdFlexSteppable

CompuCellSetup.register_steppable(steppable=GC_TEM_AdFlexSteppable(frequency=1))


CompuCellSetup.run()
