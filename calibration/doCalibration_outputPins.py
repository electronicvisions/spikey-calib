import calibratorOutputPins as calibrator
import sys, numpy, time, pickle

#######################################################################
numPinBlocks = 48 # number of neurons to be tested per pin
numVsteps    = 10 # number of voltage steps to be swept through
numRuns      = 5  # number of runs per voltage, per pin and per neuron
#######################################################################

timeStart = time.time()

if len(sys.argv) < 4:
    raise Exception('provide arguments [station] [filename results] [filename raw data]')

workstation = sys.argv[1]
filename = sys.argv[2]
filenameRawData = sys.argv[3]

calib = calibrator.CalibratorOutputPins(workstation=workstation, numPinBlocks=numPinBlocks, numVsteps=numVsteps, numRuns=numRuns)

calibResult, rawData = calib.calib()

numpy.savetxt(filenameRawData, rawData)
pickleFile = open(filename, 'wb')
pickle.dump(calibResult, pickleFile)
pickleFile.close()

print 'Output pin calibration took', round((time.time() - timeStart) / 60., 1), 'minutes'
