import pyNN.hardware.spikey as pynn
import workstation_control
import numpy
import time
import sys
import os
import shutil



startTime = time.time()

if len(sys.argv) > 1:
    workstation = sys.argv[1]
else:
    workstation = None
if len(sys.argv) > 2:
    filenameRawData = sys.argv[2]
else:
    filenameRawData = './voutRaw.dat'

# necessary setup
pynn.setup(timestep=1.0, useUsbAdc=False, workStationName=workstation,
           calibOutputPins=False, calibIcb=False, calibTauMem=False, calibSynDrivers=False, calibVthresh=False, rng_seeds=[1234])


####################################################################
                   #  experiment parameters #
####################################################################

neuron = pynn.Population(1, pynn.IF_facets_hardware1)
neuron.record()
pynn.run(1000.0)

pynn.hardware.hwa.sp.autocalib(pynn.hardware.hwa.cfg, filenameRawData)

endTime = time.time()
print 'vouts calibration took', (endTime-startTime)/60.,'minutes'
print 'please commit results in "*/spikeyhal/spikeycalib.xml" (ignore the below warning that this file does not exist)'
