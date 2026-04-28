import os
import sys

# Add OpenSim bin directory so DLLs can be found
opensim_bin = r"D:\Program Files\OpenSim\OpenSim 4.5\bin"
os.add_dll_directory(opensim_bin)
os.environ['PATH'] = opensim_bin + ';' + os.environ['PATH']

# Add the Python SDK to the module search path
sys.path.insert(0, r"D:\Program Files\OpenSim\OpenSim 4.5\sdk\Python")

import opensim as osim

model = osim.Model(r"D:\Program Files\Opensim-using_DT_Paper\Models\gait2392_simbody.osim")
print(f"Model: {model.getName()}")
print(f"Bodies: {model.getBodySet().getSize()}")
print(f"Muscles: {model.getMuscles().getSize()}")
print("OpenSim connected successfully!")