import calibratorTauRef as calibrator
import numpy as np
import sys, time


########################################### calibration parameters ########################################
neuronIDs = range(192, 384)                 # neurons we want to observe
icb_range = np.arange(0.001, 0.6, 0.02)     # icb range to be measured
tau_mem = 2.0                               # membrane time constant. As small as possible for good results
runs = 50                                   # number of runs to be averaged for each neuron and icb value
overwriteData = True                        # replace possibly existing data?
debugPlot = False                           # plot measurements of tau_ref for debugging? 

if len(sys.argv) < 4:
    print 'please provide [station] [filename results] [filename raw data]'
    exit()

workstation = sys.argv[1]
filenameResult = sys.argv[2]
filenameRawData = sys.argv[3]

# initialize calibrator object for given neurons and parameters
clb = calibrator.CalibratorTauRef(neuronIDs=neuronIDs, workstation=workstation, icb_range=icb_range, tau_mem=tau_mem, debugPlot=debugPlot, filename=filenameResult, filenameRawData=filenameRawData, runs=runs, overwriteData=overwriteData)
    
# run calibration
start = time.time()
clb.calib()
clb.fit_data()

print 'calibrating neurons {0} to {1} took {2} minutes'.format(min(neuronIDs), max(neuronIDs), round((time.time()-start)/60., 2))
