import numpy as np
import os, sys
import time
import pyNN.hardware.spikey as p
import utilities_calib_tauMem as utils
import hwconfig_default_s1v2 as default
from datetime import datetime
from copy import deepcopy

class Calibrator_TauMem(object):

    ### initialize calibrator object with given parameters ###
    def __init__(self, neuronIDs, workstation=None, numValues=5, iLeakRange=[0.2, 2.5], filenameResult='./tauMemFit.dat', filenameRawData='', overwrite_data=True, chipVersion=5, reportFile=None):

        self.workstation = workstation
        self.neuronIDs = neuronIDs
        self.numValues = numValues
        self.iLeakRange = iLeakRange
        self.filenameResult = filenameResult
        self.filenameRawData = filenameRawData
        self.overwrite_data = overwrite_data
        self.chipVersion = chipVersion
        self.reportFile = reportFile
        self.trials = 5        # trials to average tau_mem across
        self.targetSpikes = 20 # minimum number of spikes (target rate)
        self.vRestStep = 1.0   # increase of resting potential if too less spikes

        # v_thresh as close as possible to v_rest such that converging region appears, for better fitting
        self.neuronParams = {
                'v_thresh'  :   p.IF_facets_hardware1.default_parameters['v_thresh'],
                'v_rest'    :   p.IF_facets_hardware1.default_parameters['v_thresh'] + 3.0,
                'v_reset'   :   p.IF_facets_hardware1.default_parameters['v_reset'],
                }

        self.maxVRest = self.neuronParams['v_rest'] + 10.0

        rawDataFolder = os.path.splitext(filenameRawData)[0] #TODO: no folder, but file
        if not os.path.isdir(rawDataFolder):
            os.mkdir(rawDataFolder)
        self.prefix_rawData = os.path.join(rawDataFolder, 'taum_vs_iLeak')


    ### record dependency of tau_mem on iLeak ###
    def record_tau(self, neuronNr, iLeak, v_rest=None):
        print 'now at neuron number', neuronNr

        # linear dependency of simulation time on 1/iLeak
        duration = 5.0*1000.0/float(iLeak)
        duration = np.min([duration, 50000.0])

        # initialize pyNN
        mappingOffset = neuronNr
        if self.chipVersion == 4:
            mappingOffset = neuronNr - 192
        p.setup(useUsbAdc=True, calibTauMem=False, calibVthresh=False, calibSynDrivers=False, calibIcb=False, mappingOffset=mappingOffset, workStationName=self.workstation)

        # set g_leak such that iLeak is the desired value
        iLeak_base = default.iLeak_base
        g_leak = float(iLeak)/iLeak_base

        # determine tau_mem, v_rest and v_reset all at once
        trials = 0
        params = deepcopy(self.neuronParams)
        params['g_leak'] = g_leak
        if v_rest != None:
            params['v_rest'] = v_rest

        neuron = p.Population(1, p.IF_facets_hardware1, params)
        neuron.record()
        p.record_v(neuron[0], '')

        crossedTargetRate = False
        while params['v_rest'] < self.maxVRest:
            print 'now at trial', trials, '/ v_rest =', params['v_rest']
            p.run(duration)
            trace = p.membraneOutput
            dig_spikes = neuron.getSpikes()[:,1]
            memtime = p.timeMembraneOutput
            timestep = memtime[1] - memtime[0]

            # if neuron spikes with too low rate, try again with higher resting potential
            if len(dig_spikes) < self.targetSpikes:
                params['v_rest'] = params['v_rest'] + self.vRestStep
                neuron.set(params)
                print 'Neuron spiked with too low rate, trying again with parameters', params
                trials += 1
            else: # proper spiking
                crossedTargetRate = True
                break

        if not crossedTargetRate:
            utils.report('Could not find parameters for which neuron {0} spikes. Will return nan tau_mem'.format(neuronNr), self.reportFile)
            return np.concatenate(([iLeak], [np.nan] * 6, [params['v_rest']]))

        p.end()

        # determine tau_mem from measurements
        result = utils.fit_tau_mem(trace, memtime, dig_spikes, timestep=timestep, reportFile=self.reportFile)
        if result == None: # fit failed
            utils.report('Fit of membrane time constant for neuron {0} failed (iLeak = {1})'.format(neuronNr, iLeak), self.reportFile)
            return np.concatenate(([iLeak], [np.nan] * 6, [params['v_rest']]))
        return np.concatenate(([iLeak], result, [params['v_rest']]))


    ### run calibration ###
    def calib(self):
        utils.report('Calibrator_TauMem.calib() called on '+ str(datetime.now()), self.reportFile, append=False)
        # iterate over every neuron in neuronIDs
        for neuron in self.neuronIDs:
            utils.report('MEASUREMENTS FOR NEURON {0}'.format(neuron), self.reportFile)
            filename_data = self.prefix_rawData + '_neuron'+str(neuron).zfill(3) + '.dat' #TODO: save to one file
            if os.path.isfile(filename_data):
                if not self.overwrite_data:
                    utils.report('File {0} already exists, will proceed to next neuron'.format(filename_data), self.reportFile)
                    continue
                else:
                    # remove file since we want a new one
                    os.remove(filename_data)

            # determine iLeak values at which to record tau_mem
            iLeakValues = []
            # use evenly spaced INVERSE iLeak values
            inv_step = (1./min(self.iLeakRange) - 1./max(self.iLeakRange))/(self.numValues-1)
            inv_values = np.arange(1./max(self.iLeakRange), 1./min(self.iLeakRange) + inv_step / 2.0, inv_step)
            iLeakValues = 1./inv_values

            # iterate over every value of given variable
            for iLeak in iLeakValues:
                utils.report('NOW AT ILEAK {0}'.format(iLeak), self.reportFile)
                # record dependency of given tau on iLeak values
                resultCollector = []
                v_rest = None
                for i in range(self.trials):
                    resultCollector.append(self.record_tau(neuronNr=neuron, iLeak=iLeak, v_rest=v_rest))
                    v_rest = resultCollector[-1][-1]
                median = np.median(resultCollector, axis=0)
                std = np.std(resultCollector, axis=0)
                result = np.concatenate((median[:-1], std[1:-1], [median[-1]]))
                utils.save_to_columns(result, filename_data)


    ### fit dependency of tau_mem on iLeak ###
    def fitResults(self):
        utils.fit_dependency(neuronIDs=self.neuronIDs, prefix_rawData=self.prefix_rawData, filename_result=self.filenameResult, overwrite_data=self.overwrite_data, reportFile=self.reportFile)

    ### plot results of this measurement ###
    def plotResults(self, fig=None, saveFig=True, filenameFig=None):
        import matplotlib.pyplot as plt
        if fig == None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        for neuron in self.neuronIDs:
            #fig.clf()
            #ax = fig.add_subplot(111)
            utils.plot_dependency(ax=ax, neuron=neuron, prefix_rawData=self.prefix_rawData, filename_result=self.filenameResult)
            #fig.savefig(os.path.splitext(filenameFig)[0] + '_' + str(neuron).zfill(3) + '.png')
        ax.set_xlim(1.0 / np.max(self.iLeakRange) * 0.8, 1.0 / np.min(self.iLeakRange) * 1.2)
        if saveFig and filenameFig != None:
            plt.savefig(filenameFig)
        #plt.show()
