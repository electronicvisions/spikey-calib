import sys
sys.path.append('../calibration') #TODO: TP: fix this
import calibratorTauRef as calibrator
import matplotlib.pyplot as plt

# neurons to plot
neuronIDs = range(192, 384)

# save plot with calibration curves?
saveFig = True

if len(sys.argv) == 4:
    filename = sys.argv[1]
    filenameRaw = sys.argv[2]
    filenameFig = sys.argv[3]
else:
    print 'please provide filename for calibration, raw data and plot'
    exit()

# initiatlize calibrator object for given neurons and data files
clb = calibrator.CalibratorTauRef(neuronIDs=neuronIDs, filename=filename, filenameRawData=filenameRaw, reportFile=None)

# fit and plot tau_ref vs iLeak
fig = plt.figure()
clb.fit_data()
clb.plot_data(fig=fig, saveFig=saveFig, filenameFig=filenameFig)
plt.show()

