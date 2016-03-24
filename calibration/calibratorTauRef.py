import pyNN.hardware.spikey as p
import numpy as np
import matplotlib.pyplot as plt
import utilities_calib_tauMem as utils
import pickle, os
from scipy.optimize import curve_fit
from subprocess import call


class CalibratorTauRef(object):
    
    def __init__(self, neuronIDs=[0], workstation=None, icb_range=[0.01, 0.1, 2.5], tau_mem=2.0, debugPlot=False, filename='calibTauRef.dat', filenameRawData='rawData.dat', runs=30, overwriteData=False, reportFile=None, minIcb=0.037):
        self.neuronIDs = neuronIDs
        self.workstation = workstation
        self.icb_range = icb_range
        self.tau_mem = tau_mem
        self.debugPlot = debugPlot
        self.filename = filename
        self.filenameRaw = filenameRawData
        self.runs = runs
        self.overwriteData = overwriteData
        self.reportFile = reportFile
        self.minIcb = minIcb

        self.meanISI = 300.                         # ms
        self.duration = (self.runs+1)*self.meanISI  # ms
        self.initial_weight = 0.003                 # uS

        self.inputParameters =  {
                                'numInputs'     :   10,                 # number of spike sources connected to observed neuron
                                'weight'        :   self.initial_weight,# uS
                                'duration'      :   self.duration,      # ms
                                'inputSpikes'   :   { 'spike_times' :   np.arange(10, self.duration, self.meanISI)}, # ms
                                }

        self.neuronParams = {
                            'v_reset'    : -90.0, # mV 
                            'e_rev_I'    : -90.0, # mV
                            'v_rest'     : -65.0, # mV
                            'v_thresh'   : -55.0, # mV
                            'g_leak'     :  0.2/(self.tau_mem/1000.), # nS
                            }

        assert len(self.inputParameters['inputSpikes']['spike_times']) >= 30, 'too few input spikes for representative data, use at least 30 spikes'

        self.rawDataFolder = os.path.splitext(filenameRawData)[0]
        if not os.path.isdir(self.rawDataFolder):
            os.makedirs(self.rawDataFolder)
        #self.prefix_rawData = os.path.join(self.rawDataFolder, 'tauRef_vs_icb')

    # exponential dependency of tau_ref on icb
    def tau_vs_icb(self, icb, a, b):
        return a*np.exp(-b*icb)


    def recordTauRef(self, neuronNr, icb):
        # necessary hardware setup
        p.setup(useUsbAdc=True, mappingOffset=neuronNr-192, calibTauMem=True, calibVthresh=False, calibSynDrivers=False, calibIcb=False, workStationName=self.workstation)
        p.hardware.hwa.setIcb(icb)
        # observed neuron
        neuron = p.Population(1, p.IF_facets_hardware1, self.neuronParams)
        # stimulating population
        input = p.Population(self.inputParameters['numInputs'], p.SpikeSourceArray, self.inputParameters['inputSpikes'])
        # connect input and neuron
        conn = p.AllToAllConnector(allow_self_connections=False, weights=self.inputParameters['weight'])
        proj = p.Projection(input, neuron, conn, synapse_dynamics=None, target='excitatory')

        # record spikes and membrane potential
        neuron.record()
        p.record_v(neuron[0],'')

        # run experiment
        p.run(self.duration)

        # evaluate results
        spikesDig = neuron.getSpikes()[:,1]
        membrane = p.membraneOutput
        time = p.timeMembraneOutput
        # clean up
        p.end()

        # determine sampling bins
        timestep = time[1] - time[0]

        # detect analog spikes
        spikesAna, isiAna = utils.find_spikes(membrane, time, spikesDig, reportFile=self.reportFile)

        # determine refractory period from measurement of analog spikes
        tau_ref, tau_ref_err, doubles_spikes = utils.fit_tau_refrac(membrane, timestep, spikesAna, isiAna, noDigSpikes=len(spikesDig), reportFile=self.reportFile, debugPlot=self.debugPlot)
       
        return tau_ref, tau_ref_err, doubles_spikes, spikesDig


    def calib(self):
        max_tries = 5
        for neuronNr in self.neuronIDs:
            tau_ref_list = []
            err_list = []
            doubles_at = []
            self.inputParameters['weight'] = self.initial_weight

            for icb in self.icb_range:
                temp_tau = []
                temp_err = []
              
                ###
                tau_ref, tau_ref_err, double_spikes, spikesDig = self.recordTauRef(neuronNr, icb)
                tries = 0
                # increase weight if there are no or too few spikes
                while tries < max_tries and len(spikesDig) < self.duration/self.meanISI - 5:
                    self.inputParameters['weight'] *= 2
                    print "too few spikes, increasing weight to", self.inputParameters['weight']
                    tau_ref, tau_ref_err, double_spikes, spikesDig = self.recordTauRef(neuronNr, icb)
                    tries += 1
                # decrease weight if bursts occur
                tries = 0
                while tries <= max_tries and double_spikes == True:
                    if icb not in doubles_at:
                        doubles_at.append(icb)
                    self.inputParameters['weight'] -= 0.2*self.inputParameters['weight'] # was 0.5
                    print "avoiding bursts, setting weight to", self.inputParameters['weight']
                    tau_ref, tau_ref_err, double_spikes, spikesDig = self.recordTauRef(neuronNr, icb)
                    tries += 1
                # valid tau_ref?
                if tau_ref != -1:
                    temp_tau.append(tau_ref)
                    temp_err.append(tau_ref_err)
                ###

                tau_ref_list.append(np.mean(temp_tau))
                err_list.append(np.mean(temp_err))

            # check if possibly existing data should be replaced
            if self.overwriteData == False and os.path.isfile(self.rawDataFolder+'/neuron{0}.txt'.format(neuronNr)):
                print 'Data for neuron {0} already exists, will proceed to next neuron'.format(neuronNr)
                continue

            with file(self.rawDataFolder+'/neuron{0}.txt'.format(neuronNr), 'w') as outfile:
                pickle.dump(self.icb_range, outfile)
                pickle.dump(tau_ref_list, outfile)
                pickle.dump(err_list, outfile)
                pickle.dump(doubles_at, outfile)
                outfile.close()
    

    def fit_data(self):
        def save_zero(neuron, append=True):
            to_save = np.concatenate((np.array([neuron]), np.zeros(4)))
            utils.save_to_columns(to_save, self.filename, append=append)

        failed_fits = []
        no_data = []

        for neuron in range(384):
            if neuron not in self.neuronIDs:
                save_zero(neuron, append=(neuron!=0))
                continue
            if not os.path.isfile(self.rawDataFolder+'/neuron{0}.txt'.format(neuron)):
                no_data.append(neuron)
                save_zero(neuron)
                continue

            data = open(self.rawDataFolder+'/neuron{0}.txt'.format(neuron))
            icb_range = np.array(pickle.load(data))
            tau_ref_list = np.array(pickle.load(data))
            err_list = np.array(pickle.load(data))
            doubles_at = pickle.load(data)

            # exclude values that could not be measured from fit
            valid = np.where(np.isfinite(tau_ref_list))[0]
            icb_range = icb_range[valid]
            tau_ref_list = tau_ref_list[valid]
            err_list = err_list[valid]

            data.close()

            if neuron == self.neuronIDs[0]:
                print 'fitting data...'

            fit_failed = False
            params = np.zeros(2)
            step_index = 0
            if (icb_range <= self.minIcb).any():
                step_index = min(np.where(icb_range >= self.minIcb))[0]
            try:
                params, cov = curve_fit(self.tau_vs_icb, icb_range[step_index:], tau_ref_list[step_index:])
            except:
                failed_fits.append(neuron)

            if neuron in failed_fits:
                save_zero(neuron, append=(neuron!=0))
            else:
                rangeBorders = [min(icb_range), max(icb_range)]
                to_save = np.concatenate((np.array([neuron]), params, rangeBorders)) 
                utils.save_to_columns(to_save, filename=self.filename, append=(neuron!=0))
        print 'successful fits: {0}/{1}'.format(len(self.neuronIDs)-len(failed_fits)-len(no_data), len(self.neuronIDs)-len(no_data))
        if len(failed_fits) > 0:
            print 'fit failed for the following neurons:', failed_fits
        if len(no_data) > 0:
            print 'no data exists for the following neurons:', no_data


    def plot_data(self, fig=None, saveFig=True, filenameFig=None):
        if fig == None:
            fig = plt.figure()
        ax = fig.add_subplot(111)

        for neuron in self.neuronIDs:
            # if no raw data exists, continue
            if not os.path.isfile(self.rawDataFolder+'/neuron{0}.txt'.format(neuron)):
                continue
            else:
                data = open(self.rawDataFolder+'/neuron{0}.txt'.format(neuron))
                icb_range = pickle.load(data)
                tau_ref_list = pickle.load(data)
                err_list = pickle.load(data)
                doubles_at = pickle.load(data)
                data.close()

                if neuron == self.neuronIDs[0]:
                    print 'plotting data and fit...'
                
                # choose random but same color for data and fit
                color = np.random.rand(3,)
                # plot measured data
                ax.errorbar(icb_range, tau_ref_list, yerr=err_list, fmt='o', color=color, label='neuron{0}'.format(neuron))
            # plot exponential fit
            if os.path.isfile(self.filename):
                fitdata = np.loadtxt(self.filename, delimiter='\t')
                params = fitdata[neuron][1:3]
                if self.minIcb <= min(icb_range):
                    icb_values = np.arange(min(icb_range), max(icb_range), (max(icb_range)-min(icb_range))/1000.)
                else:
                    icb_values = np.arange(self.minIcb, max(icb_range), (max(icb_range)-self.minIcb)/1000.)
                ax.plot(icb_values, self.tau_vs_icb(icb_values, params[0], params[1]), color=color)
            else:
                print 'no file with name {0} exists'.format(self.filename)
        ax.set_xlabel('icb')
        ax.set_ylabel('tau_ref [ms]')
        ax.axvline(x=self.minIcb, color='k', lw=3)
        ax.set_ylim(0, 10)
        #plt.legend()
        if saveFig and filenameFig != None:
            plt.savefig(filenameFig)
        return fig
