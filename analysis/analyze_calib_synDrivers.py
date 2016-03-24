#!/usr/bin/python

import sys

if len(sys.argv) == 3:
    filenameRaw = sys.argv[1]
    filenameFig = sys.argv[2]
else:
    print 'please provide filename for raw data and plot'
    exit()

import numpy as np
import matplotlib.pyplot as plt
import os
fmt = 'png'

fallExcData = np.loadtxt(os.path.join(filenameRaw, 'fall_exc.dat'))
outExcData = np.loadtxt(os.path.join(filenameRaw, 'out_exc.dat'))
fallInhData = np.loadtxt(os.path.join(filenameRaw, 'fall_inh.dat'))
outInhData = np.loadtxt(os.path.join(filenameRaw, 'out_inh.dat'))

def plotRateHist(ax, rate, label=''):
    xlim = (rate.min(), rate.max())
    ax.hist(rate, range=xlim, bins=20, label=label)
    ax.axvline(x=np.mean(rate), color='r', lw=2)
    ax.set_xlim(xlim)
    ax.set_xlabel('r [Hz]')
    ax.set_ylabel('# synapse drivers')
    ax.legend()

def plotParamHist(ax, param, xlabel=''):
    xlim = (0, 2.5)
    ax.hist(param, range=xlim, bins=25)
    ax.axvline(x=np.mean(param), color='r', lw=2)
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel + ' [muA]')
    ax.set_ylabel('# synapse drivers')

def crossPlot(xData, yData, xError=None, yError=None, title='', filename=None):
    xlim = (0, 2.5)
    plt.figure()
    if xError != None and yError != None:
        plt.errorbar(xData, yData, xError, yError, 'ro', mew=1, mec='r', ecolor='gray')
    else:
        plt.plot(xData, yData, 'xr')
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.xlabel('drvifall [muA]')
    plt.ylabel('drviout [muA]')
    plt.title(title)
    if filename != None:
        plt.savefig(filename + '.' + fmt)

def plotParamOverDriver(index, rate, ylabel='rate [Hz]', title='', filename=None):
    plt.figure()
    plt.plot(index, rate, 'x-r')
    plt.xlabel('driver index')
    plt.ylabel(ylabel)
    plt.title(title)
    if filename != None:
        plt.savefig(filename + '.' + fmt)

fig = plt.figure()
fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.25, hspace=0.25)

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

plotRateHist(ax1, fallExcData[:,1], label='after calib drvifallExc')
plotRateHist(ax2, fallInhData[:,1], label='after calib drvifallInh')
plotRateHist(ax3, outExcData[:,1], label='after calib drvioutExc')
plotRateHist(ax4, outInhData[:,1], label='after calib drvioutInh')

plt.savefig(filenameFig + '_rates.' + fmt)

fig = plt.figure()
fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.25, hspace=0.25)

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

plotParamHist(ax1, fallExcData[:,0], xlabel='drvifallExc')
plotParamHist(ax2, outExcData[:,0], xlabel='drvioutExc')
plotParamHist(ax3, fallInhData[:,0], xlabel='drvifallInh')
plotParamHist(ax4, outInhData[:,0], xlabel='drvioutInh')

plt.savefig(filenameFig + '_params.' + fmt)

#plotParamOverDriver(range(len(fallExcData[:,1])), fallExcData[:,1], title='exc calib fall', filename=os.path.join(filenameRaw, 'driver_rate_fall_exc'))
#plotParamOverDriver(range(len(fallInhData[:,1])), fallInhData[:,1], title='inh calib fall', filename=os.path.join(filenameRaw, 'driver_rate_fall_inh'))
#plotParamOverDriver(range(len(outExcData[:,1])), outExcData[:,1], title='exc calib out', filename=os.path.join(filenameRaw, 'driver_rate_out_exc'))
#plotParamOverDriver(range(len(outInhData[:,1])), outInhData[:,1], title='inh calib out', filename=os.path.join(filenameRaw, 'driver_rate_out_inh'))

#plotParamOverDriver(range(len(fallExcData[:,0])), fallExcData[:,0], ylabel='drvifallExc', title='exc calib fall', filename=os.path.join(filenameRaw, 'driver_param_fall_exc'))
#plotParamOverDriver(range(len(fallInhData[:,0])), fallInhData[:,0], ylabel='drvifallInh', title='inh calib fall', filename=os.path.join(filenameRaw, 'driver_param_fall_inh'))
#plotParamOverDriver(range(len(outExcData[:,0])), outExcData[:,0], ylabel='drvioutExc', title='exc calib out', filename=os.path.join(filenameRaw, 'driver_param_out_exc'))
#plotParamOverDriver(range(len(outInhData[:,0])), outInhData[:,0], ylabel='drvioutInh', title='inh calib out', filename=os.path.join(filenameRaw, 'driver_param_out_inh'))

#crossPlot(fallExcData[:,0], outExcData[:,0], title='exc calib', filename=os.path.join(filenameRaw, 'cross_exc'))
#crossPlot(fallInhData[:,0], outInhData[:,0], title='inh calib', filename=os.path.join(filenameRaw, 'cross_inh'))
