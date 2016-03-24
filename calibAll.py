# Master calibration script for the Spikey hardware system (version 4 and 5)

# This script executes all available calibration routines for the Spikey hardware system.
# The parameters specifying the system are taken from the file 'workstations.xml' which should be
# located in PYNN_HW_PATH/config.

import os
import sys

if os.environ.has_key('PYNN_HW_PATH'):
    pynnhw_path = os.environ['PYNN_HW_PATH']
    sys.path.insert(0, os.path.join(pynnhw_path, "tools"))
else:
    raise Exception("Variable PYNN_HW_PATH not set!")

import time
import shutil
import workstation_control



# Define what to calibrate (the order DOES matter, and it's strongly recommended to keep to the one proposed below)
############################
calibVouts = True
calibOutputPins = True
calibTauMem = True
calibIcb = False
calibSynDrivers = True
############################

startTime = time.time()

if len(sys.argv) > 1:
    workstation = sys.argv[1]
else:
    workstation = workstation_control.getDefaultWorkstationName()
    workstation = workstation.replace('\n', '')

print 'INFO: Using workstation', workstation

# get system info
myStationDict = workstation_control.getWorkstation(workStationName=workstation)
basePath = workstation_control.basePath

# make folders
folderRawData = './results/rawData'
folderPlots = './results/plots'

if not os.path.isdir(folderRawData):
    os.makedirs(folderRawData)
if not os.path.isdir(folderPlots):
    os.makedirs(folderPlots)

# backup calibration files
dataExtension = '.dat'
timestring = time.strftime("%Y_%m_%d-%H_%M_%S")
def backupFile(filename):
    if os.path.isfile(filename):
        print 'backup', filename
        shutil.copy(filename, filename + '.bak.' + timestring)
        os.remove(filename)
    #TODO: the following is obsolete if calib files unified
    folder = os.path.splitext(filename)[0]
    if os.path.isdir(folder):
        print 'backup', folder
        shutil.copytree(folder, folder + '.bak.' + timestring)
        shutil.rmtree(folder)
def genFilenameRaw(ident):
    return os.path.join(folderRawData, workstation + '_' + ident + dataExtension)

calibfileVoutsRaw      = genFilenameRaw('voutRaw')
calibfileOutputPinsRaw = genFilenameRaw('outputPinsRaw')
calibfileTauMemRaw     = genFilenameRaw('tauMemRaw')
calibfileIcbRaw        = genFilenameRaw('icbRaw')
calibfileSynDriverRaw  = genFilenameRaw('synDriver')
calibfileOutputPins    = os.path.join(basePath, myStationDict['calibfileOutputPins'])
calibfileTauMem        = os.path.join(basePath, myStationDict['calibfileTauMem'])
calibfileIcb           = os.path.join(basePath, myStationDict['calibfileIcb'])
calibfileSynDriverExc  = os.path.join(basePath, myStationDict['calibfileSynDriverExc'])
calibfileSynDriverInh  = os.path.join(basePath, myStationDict['calibfileSynDriverInh'])

if calibVouts:      backupFile(calibfileVoutsRaw)
if calibOutputPins: backupFile(calibfileOutputPins)
if calibOutputPins: backupFile(calibfileOutputPinsRaw)
if calibTauMem:     backupFile(calibfileTauMem)
if calibTauMem:     backupFile(calibfileTauMemRaw)
if calibIcb:        backupFile(calibfileIcb)
if calibIcb:        backupFile(calibfileIcbRaw)
if calibSynDrivers: backupFile(calibfileSynDriverExc)
if calibSynDrivers: backupFile(calibfileSynDriverInh)
if calibSynDrivers: backupFile(calibfileSynDriverRaw)





# from here, calibration starts...
status = 0
######################################################################################################

if calibVouts:

    status = os.system('python calibration/doCalibration_vouts.py ' + workstation + ' ' + os.path.abspath(calibfileVoutsRaw))

######################################################################################################

if calibOutputPins and status==0:

    status = os.system('python calibration/doCalibration_outputPins.py ' + workstation + ' ' + os.path.abspath(calibfileOutputPins) + ' ' + os.path.abspath(calibfileOutputPinsRaw))

######################################################################################################

if calibTauMem and status==0:
    status = os.system('cd calibration; '
                       + 'python doCalibration_tauMem.py ' + workstation + ' ' + os.path.abspath(calibfileTauMem) + ' ' + os.path.abspath(calibfileTauMemRaw)
                       + '; cd ..')

######################################################################################################

if calibIcb and status==0:
    status = os.system('cd calibration; '
                       + 'python doCalibration_tauRef.py ' + workstation + ' ' + os.path.abspath(calibfileIcb) + ' ' + os.path.abspath(calibfileIcbRaw)
                       + '; cd ..')

######################################################################################################

if calibSynDrivers and status==0:
    status = os.system('cd calibration; '
                       'python doCalibration_synDrivers.py ' + workstation + ' ' + os.path.abspath(calibfileSynDriverExc) + ' ' + os.path.abspath(calibfileSynDriverInh) + ' ' + os.path.abspath(calibfileSynDriverRaw)
                       + '; cd ..')

######################################################################################################

if status != 0:
    print '\ncalibration FAILED!!!'
else:
    print "\ncalibration successful"
