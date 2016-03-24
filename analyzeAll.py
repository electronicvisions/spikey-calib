# Master to analyze all calibration data

import sys, os, time, shutil

# Define what to analyze
############################
calibAdc = True
calibVouts = True
calibOutputPins = True
calibTauMem = True
calibSynDrivers = True
############################
#TODO: add analysis for calibration of refractory period, see also analyze_calib_tauRef.py

if os.environ.has_key('PYNN_HW_PATH'):
    pynnhw_path = os.environ['PYNN_HW_PATH']
    sys.path.insert(0, os.path.join(pynnhw_path, "tools"))
else:
    raise Exception("Variable PYNN_HW_PATH not set!")
if os.environ.has_key('SPIKEYHALPATH'):
    spikeyhal_path = os.environ['SPIKEYHALPATH']
else:
    raise Exception("Variable SPIKEYHALPATH not set!")



import workstation_control

if len(sys.argv) > 1:
    workstation = sys.argv[1]
else:
    workstation = workstation_control.getDefaultWorkstationName()
    workstation = workstation.replace('\n', '')

print 'INFO: Using workstation', workstation

# get system info
myStationDict = workstation_control.getWorkstation(workStationName=workstation)
basePath = workstation_control.basePath
spikeyNr = myStationDict['spikeyNr']

# make folders
folderRawData = './results/rawData'
folderPlots = './results/plots'

if not os.path.isdir(folderRawData):
    os.makedirs(folderRawData)
if not os.path.isdir(folderPlots):
    os.makedirs(folderPlots)

# backup calibration files
filenamePrefix = 'station'
dataExtension = '.dat'
figExtension = '.png'
timestring = time.strftime("%Y_%m_%d-%H_%M_%S_")
def backupFile(filename):
    if os.path.isfile(filename):
        pathStr, fileStr = os.path.split(filename)
        shutil.copy(filename, os.path.join(pathStr, timestring + fileStr))
def genFilenameRaw(ident, noExt=False):
    filename = filenamePrefix + str(spikeyNr) + '_' + ident
    if not noExt:
        filename += dataExtension
    return os.path.join(folderRawData, filename)

calibfileAdcRaw        = genFilenameRaw('adc')
calibfileVoutsRaw      = genFilenameRaw('voutRaw')
calibfileOutputPinsRaw = genFilenameRaw('outputPinsRaw')
calibfileTauMemRaw     = genFilenameRaw('tauMemRaw')
calibfileSynDriverRaw  = genFilenameRaw('synDriver', noExt=True)
calibfileOutputPins    = os.path.join(basePath, myStationDict['calibfileOutputPins'])
calibfileTauMem        = os.path.join(basePath, myStationDict['calibfileTauMem'])
calibfileSynDriverExc  = os.path.join(basePath, myStationDict['calibfileSynDriverExc'])
calibfileSynDriverInh  = os.path.join(basePath, myStationDict['calibfileSynDriverInh'])

def genFilenameFig(ident):
    return os.path.join(folderPlots, filenamePrefix + str(spikeyNr) + '_' + ident + figExtension)

figfileAdc        = genFilenameFig('adc')
figfileVout       = genFilenameFig('vout')
figfileOutputPins = genFilenameFig('outputPins')
figfileNeuronMems = genFilenameFig('neuronMems')
figfileTauMem     = genFilenameFig('tauMem')
figfileSynDriver  = genFilenameFig('synDriver')

if calibAdc: backupFile(figfileAdc)
if calibVouts: backupFile(figfileVout)
if calibOutputPins: backupFile(figfileOutputPins)
if calibOutputPins: backupFile(figfileNeuronMems)
if calibTauMem: backupFile(figfileTauMem)
######################################

if calibAdc:
    os.system('python analysis/analyze_calib_adc.py ' + calibfileAdcRaw + ' ' + figfileAdc)

if calibVouts:
    filenameXML = os.path.join(spikeyhal_path, 'spikeycalib.xml')
    os.system('python analysis/analyze_calib_vout.py ' + filenameXML + ' ' + calibfileVoutsRaw + ' ' + figfileVout)

if calibOutputPins:
    os.system('python analysis/analyze_calib_outputPins.py ' + calibfileOutputPins + ' ' + calibfileOutputPinsRaw + ' ' + figfileOutputPins + ' ' + figfileNeuronMems)

if calibTauMem:
    os.system('python analysis/analyze_calib_tauMem.py ' + calibfileTauMem + ' ' + calibfileTauMemRaw + ' ' + figfileTauMem)

if calibSynDrivers:
    os.system('python analysis/analyze_calib_synDrivers.py ' + calibfileSynDriverRaw + ' ' + figfileSynDriver)
