import calibratorSynDriver as calibrator
import sys, time

blocks = range(2)
synapseDriverList = range(256)

timeStart = time.time()

if len(sys.argv) < 5:
    raise Exception('provide arguments [station] [filename results exc] [filename results inh] [filename raw data]')

workstation = sys.argv[1]
# TODO: replace result files for excitatory and inhibitory drivers with single JSON file
filenameExc = sys.argv[2]
filenameInh = sys.argv[3]
filenameRawData = sys.argv[4]

calib = calibrator.CalibratorSynDriver_tpfeil(workstation=workstation, filenameRawData=filenameRawData, synapseDriverList=synapseDriverList)

for block in blocks:
    calib.calib(block)
calib.writeResults(filenameExc, filenameInh)

print 'Synapse driver calibration took', round((time.time() - timeStart) / 60., 1), 'minutes'
