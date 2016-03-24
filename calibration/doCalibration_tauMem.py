import calibratorTauMem as calibrator
import os, sys, time

################################## set parameters for calibration ###########################

neuronIDs = range(384)  # neurons to be calibrated
numValues = 7           # number of iLeak steps
iLeakRange = [0.2, 1.2] # value range for iLeak

# overwrite possibly existing data?
overwrite_data = True

if len(sys.argv) < 4:
    raise Exception('provide arguments [station] [filename results] [filename raw data]')

workstation = sys.argv[1]
filenameResult = sys.argv[2]
filenameRawData = sys.argv[3]

# TODO: add report to JSON file for raw data; up to then store in same folder as raw data
filenameReport = os.path.join(os.path.splitext(filenameRawData)[0], 'report.dat')

import pyNN.hardware.spikey as pynn
pynn.setup(workStationName=workstation)
chipVersion = pynn.getChipVersion()
pynn.end()
if chipVersion == 4:
    neuronIDs = range(192, 384)

##############################################################################################

# initiatlize calibrator object for given neurons and parameters
clb = calibrator.Calibrator_TauMem(neuronIDs=neuronIDs, workstation=workstation, numValues=numValues, iLeakRange=iLeakRange, filenameResult=filenameResult, filenameRawData=filenameRawData, overwrite_data=overwrite_data, chipVersion=chipVersion, reportFile=filenameReport)

startTime = time.time()

# run calibration
clb.calib()
# fit tau_mem vs iLeak
clb.fitResults()
#clb.plotResults(filenameFig='delme.png')

endTime = time.time()
print 'Calibration of tau_mem for neurons {0} to {1} took {2} minutes'.format(min(neuronIDs), max(neuronIDs), round((endTime-startTime)/60., 1))
