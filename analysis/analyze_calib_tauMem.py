import sys
sys.path.append('calibration') #TODO: TP: fix this
import calibratorTauMem as calibrator
import matplotlib.pyplot as plt

# neurons to plot
neuronIDs = range(384)
# save plot with calibration curves?
saveFig = True

if len(sys.argv) == 4:
    filename = sys.argv[1]
    filenameRaw = sys.argv[2]
    filenameFig = sys.argv[3]
else:
    print 'please provide filename for calibration, raw data and plot'
    exit()

# initialize calibrator object for given neurons and data files
clb = calibrator.Calibrator_TauMem(neuronIDs=neuronIDs, filenameResult=filename, filenameRawData=filenameRaw, reportFile=None)

# fit and plot tau_mem vs iLeak
fig = plt.figure()
clb.plotResults(fig=fig, saveFig=saveFig, filenameFig=filenameFig)
