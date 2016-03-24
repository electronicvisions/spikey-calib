import sys, pickle
import numpy as np
import matplotlib.pyplot as plt

withAlternativeFitPoly = False

if len(sys.argv) == 5:
    filename = sys.argv[1]
    filenameRaw = sys.argv[2]
    filenameFig1 = sys.argv[3]
    filenameFig2 = sys.argv[4]
else:
    print 'please provide filename for calibration, raw data and plot'
    exit()

pinRange = range(4)
pinBlocksRange = range(48)

dataRaw = np.loadtxt(filenameRaw)
xMin = np.min(dataRaw[:,1])
xMax = np.max(dataRaw[:,1])
yMin = np.min(dataRaw[:,2])
yMax = np.max(dataRaw[:,2])
index = np.linspace(yMin * 0.9, yMax * 1.1, 1000)

pickleFile = open(filename, 'rb')
data = pickle.load(pickleFile)
pickleFile.close()

def format_plot(filename, titlePre):
    plt.xlim(xMin, xMax)
    plt.title(titlePre + ' calibration and fit')
    plt.xlabel('applied voltage [V]')
    plt.ylabel('measured voltage [V]')
    plt.savefig(filename)

def plot_fit(key0, key1):
    fit = data[key0][key1]
    evaluate = np.polyval(fit, index)
    plt.plot(evaluate, index, lw=2, c='r')

def own_fit(dataOwnFit):
    limLow = 0.4 #V
    limHigh = 1.4 #V
    plt.axvline(limLow, c='k', ls='--')
    plt.axvline(limHigh, c='k', ls='--')

    dataToFit = dataOwnFit[dataOwnFit[:,0] >= limLow]
    dataToFit = dataToFit[dataToFit[:,0] <= limHigh]
    print 'discarded', (len(dataOwnFit) - len(dataToFit)), 'data points'

    for degree in range(3, 4):
        fit, res, rank, singular_values, rcond = np.polyfit(dataToFit[:,1], dataToFit[:,0], degree, full=True)
        evaluate = np.polyval(fit, index)
        plt.plot(evaluate, index, lw=2, c='g')
        print 'degree of polynomial fit:', degree, '-> residual:', res
        print 'coefficients:', fit

plt.figure()
for block in range(2):
    for pin in pinRange:
        #plot raw data output pins
        dataOnePin = dataRaw[np.array(dataRaw[:,0] / 192, np.int) * 4 + dataRaw[:,0] % 4 == block * 4 + pin]
        if len(dataOnePin) == 0: #for spikey version 4
            print 'data for pin', pin, 'does not exist'
            continue
        vouts = np.unique(dataOnePin[:,1])
        mean = []
        std = []
        for vout in vouts:
            mean.append(np.mean(dataOnePin[dataOnePin[:,1] == vout][:,2]))
            std.append(np.std(dataOnePin[dataOnePin[:,1] == vout][:,2]))
        plt.errorbar(vouts, mean, yerr=std, c='b')

        #plot fit
        plot_fit('polyFitOutputPins', 'pin' + str(block * 4 + pin))

        #plot alternative fits
        if withAlternativeFitPoly:
            dataOnePin = np.vstack((vouts, mean, std)).T
            own_fit(dataOnePin)
format_plot(filenameFig1, 'Output pins')

plt.figure()
for block in range(2):
    for pin in pinRange:
        for pinBlock in pinBlocksRange:
            neuron = block * 192 + pinBlock * 4 + pin
            #plot raw data each neuron
            dataOneNeuron = dataRaw[dataRaw[:,0] == neuron]
            if len(dataOneNeuron) == 0: #for spikey version 4
                print 'data for neuron', neuron, 'does not exist'
                continue
            plt.errorbar(dataOneNeuron[:,1], dataOneNeuron[:,2], yerr=dataOneNeuron[:,3], c='b')

            #plot fits
            plot_fit('polyFitNeuronMems', 'neuron' + str(neuron).zfill(3))

            #plot alternative fits
            if withAlternativeFitPoly:
                own_fit(dataOneNeuron[:,1:4])
format_plot(filenameFig2, 'Neuron membrane')
