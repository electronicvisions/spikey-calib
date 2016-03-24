#!/usr/bin/python

import os, sys, time, shutil, logging
import numpy as np
from multiprocessing import Process, Queue
import pyNN.hardware.spikey as pynn
import hwconfig_default_s1v2 as default

class CalibratorSynDriver_tpfeil(object):
    def __init__(self, workstation=None, randomSeed=0, filenameRawData='delme', synapseDriverList=range(256)):
        np.random.seed(randomSeed) #TODO: if multiple runs, seed should be changed for each run
        self.workstation    = workstation

        self.targetWeight   = 7.0 * 1e-9 # target conductance for excitatory synapses
        self.ratioInhExc    = 4.0        # ratio between inhibitory and excitatory conductances
        self.targetRate     = 5.0        # target firing rate
        self.hardwareWeight = 7.0        # to be calibrated to targetWeight => synaptic weight = targetWeight / hardwareWeight
        self.stimParams     = None
        self.neuronParams   = {'v_rest'     : -75.0,  # resting membrane potential in mV
                               'v_reset'    : -80.0,  # reset potential after a spike in mV
                               'g_leak'     :  20.0,  # g=c/tau in nS
                                                      # TODO: 40nS is better, because can be reached by more neurons
                               'e_rev_I'    : -80.0,  # reversal potential for inhibitory input in mV
                               'v_thresh'   : -55.0   # spike threshold in mV
                              }

        self.neuronList        = []
        self.neuronListBlocks  = [range(192), range(192, 384)] # for left and right block
        self.synapseDriverList = synapseDriverList             # starts from 0 for left and right block

        self.simulationTrials  = 10                            # for software simulations
        self.emulationTrials   = 3
        self.runtime           = 21.0 * 1000.0                 # in ms
                                                               # TODO: investigate with trial-to-trial variability over runtime
        self.cutfirst          = 1000.0                        # discard spikes at beginning (in ms)
        self.currentMax        = default.currentMax
        self.currentMin        = default.currentMin
        self.currentDiffLimit  = 0.01

        #setup folder
        #TODO: replace multiple files in folder with single file
        self.folder = os.path.splitext(filenameRawData)[0]
        self.filenameDrvifallExc = 'fall_exc.dat'
        self.filenameDrvioutExc  = 'out_exc.dat'
        self.filenameDrvifallInh = 'fall_inh.dat'
        self.filenameDrvioutInh  = 'out_inh.dat'

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        #setup logging
        self.logfile = 'syn_calib_log.txt'
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG)
        fileHandler = logging.FileHandler(filename=os.path.join(self.folder, self.logfile), mode='w')
        logFormatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%d %b %Y %H:%M:%S")
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)

        assert len(self.synapseDriverList) >= 2, 'no synapse drivers must be >= 2'

        #input rates obtained from software simulations
        #7nS 5Hz
        self.nuExcNoInh = 91.2
        self.nuExc = 136.8
        self.nuInh = 97.8

        #2nS 3Hz LowTh
        #self.nuExcNoInh = 160.2
        #self.nuExc = 320.4
        #self.nuInh = 298.4



    def __del__(self):
        logging.shutdown()



    #def getStimulus(self, self.targetWeight):
        #'''Software simulations to obtain firing rate of stimulus.'''
        #from theory import theory
        #collector = []
        #for i in range(self.simulationTrials):
            #mytheory = theory(useNest=True, targetRate=self.targetRate, seed=i, neuronParams=self.neuronParams)
            #nuExcNoInh, nuExc, nuInh = mytheory.findFrequencies(self.targetWeight)
            #collector.append([nuExcNoInh, nuExc, nuInh])
            ##mytheory.simCheck(self.targetWeight, nuExc, nuInh)
        #collector = np.array(collector)
        #nuExcNoInh, nuExc, nuInh = np.mean(collector, axis=0)
        #nuExcNoInhErr, nuExcErr, nuInhErr = np.std(collector, axis=0)
        #np.savetxt(os.path.join(self.folder, 'ref_sims.txt'), [nuExcNoInh, nuExc, nuInh, nuExcNoInhErr, nuExcErr, nuInhErr])
        #self.logger.info('found nuExcNoInh/nuExc/nuInh: ' + str(nuExcNoInh) + '+-' + str(nuExcNoInhErr) + '/' + str(nuExc) + '+-' + str(nuExcErr) + '/' + str(nuInh) + '+-' + str(nuInhErr))
        #return nuExcNoInh, nuExc, nuInh, nuExcNoInhErr, nuExcErr, nuInhErr



    def setStimulus(self, nuExc, nuInh):
        '''Set firing rate of excitatory and inhibitory stimulus.'''
        return {'excitatory':{'duration': self.runtime, 'start': 0, 'rate': nuExc},
                'inhibitory':{'duration': self.runtime, 'start': 0, 'rate': nuInh}}



    def emulateMemLeakSafe(self, *args, **kwargs):
        '''deprecated TODO: remove'''
        def processEmulate(*args, **kwargs):
            queue = kwargs.pop('results')
            rate = self.emulate(*args, **kwargs)
            queue.put(rate)

        results = Queue()
        kwargs['results'] = results
        proc = Process(target=processEmulate, args=args, kwargs=kwargs)
        proc.start()
        proc.join()
        rate = results.get()
        if proc.exitcode != 0:
            print 'ERROR: emulate() crashed'
            exit()

        return rate



    def emulate(self, driverIndexExc, driverIndexInh=None, drvirise=None, drvifallExc=None, drvifallInh=None, drvioutExc=None, drvioutInh=None, filename=None, calibSynDrivers=False):
        '''Run emulations on hardware.'''
        assert self.stimParams != None, 'specifiy stimulus first'
        pynn.setup(calibTauMem=True,
                   calibSynDrivers=calibSynDrivers,
                   calibVthresh=False,
                   calibIcb=False,
                   workStationName=self.workstation,
                   writeConfigToFile=False)

        #create neuron
        self.neuronList.sort()
        neuronCollector = []
        currentIndex = 0
        for neuronIndex in self.neuronList:
            if neuronIndex > currentIndex: #insert dummy neurons if neuronList not continuous
                dummyPopSize = neuronIndex - currentIndex
                dummy = pynn.Population(dummyPopSize, pynn.IF_facets_hardware1, self.neuronParams)
                currentIndex += dummyPopSize
                self.logger.debug('inserted ' + str(dummyPopSize) + ' dummy neurons')
            if neuronIndex == currentIndex:
                neuron = pynn.Population(1, pynn.IF_facets_hardware1, self.neuronParams)
                currentIndex += 1
                neuron.record()
                neuronCollector.append(neuron)
            else:
                raise Exception('Could not create all neurons')

        #create input and connect to neuron
        synDrivers = [[driverIndexExc, 'excitatory'], [driverIndexInh, 'inhibitory']]
        synDrivers.sort()

        currentIndex = 0
        for synDriver in synDrivers:
            toInsertIndex = synDriver[0]
            targetType = synDriver[1]
            if toInsertIndex == None:
                self.logger.debug('skipping ' + targetType + ' stimulation')
                continue
            if toInsertIndex > currentIndex:
                dummyPopSize = toInsertIndex - currentIndex
                dummy = pynn.Population(dummyPopSize, pynn.SpikeSourcePoisson)
                currentIndex += dummyPopSize
                self.logger.debug('inserted ' + str(dummyPopSize) + ' dummy synapse drivers')
            stim = pynn.Population(1, pynn.SpikeSourcePoisson, self.stimParams[targetType])
            self.logger.debug('inserted 1 stimulus of type ' + targetType + ' with rate ' + str(self.stimParams[targetType]['rate']))

            if targetType == 'excitatory':
                hardwareWeightTemp = self.hardwareWeight * pynn.minExcWeight()
            elif targetType == 'inhibitory':
                hardwareWeightTemp = self.hardwareWeight * pynn.minInhWeight()
            else:
                raise Exception('Synapse type not supported!')

            for neuron in neuronCollector:
                pynn.Projection(stim, neuron, method=pynn.AllToAllConnector(weights=hardwareWeightTemp), target=targetType)
            currentIndex += 1

        #set custom parameters
        if drvirise != None:
            pynn.hardware.hwa.drvirise = drvirise
        else:
            pynn.hardware.hwa.drvirise = default.drvirise
        if drvifallExc != None:
            pynn.hardware.hwa.drvifall_base['exc'] = drvifallExc
        else:
            pynn.hardware.hwa.drvifall_base['exc'] = default.drvifall_base['exc']
        if drvifallInh != None:
            pynn.hardware.hwa.drvifall_base['inh'] = drvifallInh
        else:
            pynn.hardware.hwa.drvifall_base['inh'] = default.drvifall_base['inh']
        if drvioutExc != None:
            pynn.hardware.hwa.drviout_base['exc'] = drvioutExc
        else:
            pynn.hardware.hwa.drviout_base['exc'] = default.drviout_base['exc']
        if drvioutInh != None:
            pynn.hardware.hwa.drviout_base['inh'] = drvioutInh
        else:
            pynn.hardware.hwa.drviout_base['inh'] = default.drviout_base['inh']

        #run
        pynn.run(self.runtime)
        #temperature = pynn.hardware.hwa.getTemperature()
        #self.logger.debug('temperature ' + str(temperature) + ' degree Celsius')

        #obtain firing rate
        spikes = None
        neuronCount = 0
        for neuron in neuronCollector:
            spikesNeuron = neuron.getSpikes()
            neuronCount += neuron.size
            if spikes == None:
                spikes = spikesNeuron
            else:
                spikes = np.concatenate((spikes, spikesNeuron))
        if len(spikes) > 0:
            spikes = spikes[spikes[:,1] > self.cutfirst]
            if len(spikes) > 0:
                spikes = spikes[spikes[:,1] <= self.runtime]
        if filename != None:
            np.savetxt(os.path.join(self.folder, filename), spikes)
        spikesPerNeuron = float(len(spikes)) / neuronCount
        pynn.end()
        rate =  spikesPerNeuron * 1e3 / (self.runtime - self.cutfirst)

        return rate



    def devideAndConquer(self, inverse, currentType, driverIndexExc, **args):
        '''
        Devide and conquer algorithm to find current values for synapse drivers.
        Set argument "inverse" to True, if higher current implies higher firing rate.
        '''
        rate = 0
        currentLow = self.currentMin
        currentHigh = self.currentMax

        #while difference between upper and lower current value too big
        while True:
            if currentHigh - currentLow < self.currentDiffLimit:
                self.logger.info('found current ' + str(current) + ' with rate ' + str(rate))
                return current, rate
            #devide
            current = currentLow + (currentHigh - currentLow) / 2.0
            args[currentType] = current

            self.logger.debug('current (is/high/low) ' + str(current) + '/' + str(currentHigh) + '/' + str(currentLow) + ' with args ' + str(args))
            rateList = []
            #take median of several runs
            for i in range(self.emulationTrials):
                self.logger.debug('run ' + str(i))
                rateList.append(self.emulate(driverIndexExc, **args))
            rate = np.median(rateList)
            self.logger.debug('rate list: ' + str(rateList) + ' rate: ' + str(rate) + 'Hz')

            #update boundaries
            if inverse:
                rateTemp = -rate
                targetRateTemp = -self.targetRate
            else:
                rateTemp = rate
                targetRateTemp = self.targetRate
            if rateTemp > targetRateTemp:
                currentHigh = current
            else:
                currentLow = current



    def calibDriverExcMemLeakSafe(self, *args, **kwargs):
        '''Calibration of excitatory drivers.'''
        def processCalibDriverExc(*args, **kwargs):
            queue = kwargs.pop('results')
            resultTemp = self.calibDriverExc(*args, **kwargs)
            queue.put(resultTemp)

        results = Queue()
        kwargs['results'] = results
        proc = Process(target=processCalibDriverExc, args=args, kwargs=kwargs)
        proc.start()
        proc.join()
        resultTemp = results.get()
        if proc.exitcode != 0:
            print 'ERROR: calibDriverExc() crashed'
            exit()

        return resultTemp



    def calibDriverExc(self, driverIndexExc):
        #drvifall
        drvifallExc, rateDrvifallExc = self.devideAndConquer(True, 'drvifallExc', driverIndexExc)
        #drviout
        drvioutExc, rateDrvioutExc = self.devideAndConquer(False, 'drvioutExc', driverIndexExc, drvifallExc=drvifallExc)
        self.logger.info('found drvifall/drviout (with rate) ' + str(drvifallExc) + '/' + str(drvioutExc) + ' (' + str(rateDrvifallExc) + '/' + str(rateDrvioutExc) + ')')
        return drvifallExc, drvioutExc, rateDrvifallExc, rateDrvioutExc



    def calibDriverInhMemLeakSafe(self, *args, **kwargs):
        '''Calibration of inhibitory drivers.'''
        def processCalibDriverInh(*args, **kwargs):
            queue = kwargs.pop('results')
            resultTemp = self.calibDriverInh(*args, **kwargs)
            queue.put(resultTemp)

        results = Queue()
        kwargs['results'] = results
        proc = Process(target=processCalibDriverInh, args=args, kwargs=kwargs)
        proc.start()
        proc.join()
        resultTemp = results.get()
        if proc.exitcode != 0:
            print 'ERROR: calibDriverInh() crashed'
            exit()

        return resultTemp



    def calibDriverInh(self, driverIndexExc, driverIndexInh, drvifallExc, drvioutExc):
        #drvifall
        drvifallInh, rateDrvifallInh = self.devideAndConquer(False, 'drvifallInh', driverIndexExc, driverIndexInh=driverIndexInh, drvifallExc=drvifallExc, drvioutExc=drvioutExc)
        #drviout
        drvioutInh, rateDrvioutInh = self.devideAndConquer(True, 'drvioutInh', driverIndexExc, driverIndexInh=driverIndexInh, drvifallInh=drvifallInh, drvifallExc=drvifallExc, drvioutExc=drvioutExc)
        self.logger.info('found drvifall/drviout (with rate) ' + str(drvifallInh) + '/' + str(drvioutInh) + ' (' + str(rateDrvifallInh) + '/' + str(rateDrvioutInh) + ')')
        return drvifallInh, drvioutInh, rateDrvifallInh, rateDrvioutInh



    def stimIndex2calibIndex(self, block, driverIndex):
        return (block + 1) * default.numPresyns - driverIndex - 1



    def calib(self, block):
        '''Complete calibration procedure.'''
        drvifallExc = None
        drvioutExc = None
        drvifallExcCollector = np.ones((512, 2)) / default.drvifall_base['exc']
        drvifallExcCollector[:,1] = 0
        drvioutExcCollector = np.ones((512, 2)) / default.drviout_base['exc']
        drvioutExcCollector[:,1] = 0
        drvifallInhCollector = np.ones((512, 2)) / default.drvifall_base['inh']
        drvifallInhCollector[:,1] = 0
        drvioutInhCollector = np.ones((512, 2)) / default.drviout_base['inh']
        drvioutInhCollector[:,1] = 0

        self.neuronList = self.neuronListBlocks[block]

        #only excitatory stimulus
        self.stimParams = self.setStimulus(self.nuExcNoInh, 0)
        for driverIndex in self.synapseDriverList:
            self.logger.info('calibrating excitatory driver ' + str(driverIndex) + ' in block ' + str(block))
            drvifallExc, drvioutExc, rateDrvifallExc, rateDrvioutExc = self.calibDriverExcMemLeakSafe(driverIndex)
            calibIndex = self.stimIndex2calibIndex(block, driverIndex)
            drvifallExcCollector[calibIndex][0] = drvifallExc / default.drvifall_base['exc']
            drvifallExcCollector[calibIndex][1] = rateDrvifallExc
            drvioutExcCollector[calibIndex][0] = drvioutExc / default.drviout_base['exc']
            drvioutExcCollector[calibIndex][1] = rateDrvioutExc
        #save results to file
        filenameExcFall = os.path.join(self.folder, self.filenameDrvifallExc)
        filenameExcOut = os.path.join(self.folder, self.filenameDrvioutExc)
        if block > 0:
            drvifallExcOld = np.loadtxt(filenameExcFall)
            drvioutExcOld = np.loadtxt(filenameExcOut)
            indexStart = (1 - block) * default.numPresyns
            indexEnd = (2 - block) * default.numPresyns
            drvifallExcCollector[indexStart:indexEnd] = drvifallExcOld[indexStart:indexEnd]
            drvioutExcCollector[indexStart:indexEnd] = drvioutExcOld[indexStart:indexEnd]
        np.savetxt(filenameExcFall, drvifallExcCollector)
        np.savetxt(filenameExcOut, drvioutExcCollector)

        #both excitatory and inhibitory stimulus
        self.stimParams = self.setStimulus(self.nuExc, self.nuInh)
        for driverIndexInh in self.synapseDriverList:
            #choose excitatory synapse driver randomly
            indexList = np.array(self.synapseDriverList)
            indexInh = np.where(indexList == driverIndexInh)[0][0]
            indexExc = indexInh
            while indexExc == indexInh:
                shuffleList = range(len(indexList))
                np.random.shuffle(shuffleList)
                indexExc = shuffleList[0]
            driverIndexExc = indexList[indexExc]

            calibIndex = self.stimIndex2calibIndex(block, driverIndexInh)
            calibIndexExc = self.stimIndex2calibIndex(block, driverIndexExc)
            self.logger.info('calibrating inhibitory driver ' + str(driverIndexInh) + ' with excitatory driver ' + str(driverIndexExc) + ' in block ' + str(block))
            drvifallInh, drvioutInh, rateDrvifallInh, rateDrvioutInh = self.calibDriverInhMemLeakSafe(driverIndexExc, driverIndexInh, drvifallExcCollector[calibIndexExc][0] * default.drvifall_base['exc'], drvioutExcCollector[calibIndexExc][0] * default.drviout_base['exc'])
            drvifallInhCollector[calibIndex][0] = drvifallInh / default.drvifall_base['inh']
            drvifallInhCollector[calibIndex][1] = rateDrvifallInh
            drvioutInhCollector[calibIndex][0] = drvioutInh / default.drviout_base['inh']
            drvioutInhCollector[calibIndex][1] = rateDrvioutInh
        #save results to file
        filenameInhFall = os.path.join(self.folder, self.filenameDrvifallInh)
        filenameInhOut = os.path.join(self.folder, self.filenameDrvioutInh)
        if block > 0:
            drvifallInhOld = np.loadtxt(filenameInhFall)
            drvioutInhOld = np.loadtxt(filenameInhOut)
            indexStart = (1 - block) * default.numPresyns
            indexEnd = (2 - block) * default.numPresyns
            drvifallInhCollector[indexStart:indexEnd] = drvifallInhOld[indexStart:indexEnd]
            drvioutInhCollector[indexStart:indexEnd] = drvioutInhOld[indexStart:indexEnd]
        np.savetxt(os.path.join(self.folder, self.filenameDrvifallInh), drvifallInhCollector)
        np.savetxt(os.path.join(self.folder, self.filenameDrvioutInh), drvioutInhCollector)



    def adjustRateNoCalib(self, inverse, currentType, **args):
        '''
        Adjust current by "devide and conquer algorithm",
        but here, firing rates are averaged over all synapse drivers.
        Used for control measurement with similar target firing rate.
        '''
        rate = 0
        rateCollector = []
        currentLow = self.currentMin
        currentHigh = self.currentMax
        while True:
            if currentHigh - currentLow < self.currentDiffLimit:
                self.logger.info('found current ' + str(current) + ' with rate ' + str(rate))
                return current, rateCollector
            current = currentLow + (currentHigh - currentLow) / 2.0
            args[currentType] = current

            self.logger.debug('current (is/high/low) ' + str(current) + '/' + str(currentHigh) + '/' + str(currentLow) + ' with args ' + str(args))
            rateCollector = []
            for driverIndexExc in self.synapseDriverList:
                rateList = []
                for i in range(self.emulationTrials):
                    rateList.append(self.emulate(driverIndexExc, **args))
                rate = np.median(rateList)
                rateCollector.append(rate)
            rate = np.mean(rateCollector)
            self.logger.debug('rate: ' + str(rate) + 'Hz')

            if inverse:
                rateTemp = -rate
                targetRateTemp = -self.targetRate
            else:
                rateTemp = rate
                targetRateTemp = self.targetRate
            if rateTemp > targetRateTemp:
                currentHigh = current
            else:
                currentLow = current



    def measureNoCalib(self):
        '''Perform control measurements tuned to similar target rate.'''
        self.stimParams = self.setStimulus(self.nuExcNoInh, 0)
        drvifallExc = self.adjustRateNoCalib(True, 'drvifallExc')
        #self.stimParams = self.setStimulus(self.nuExc, self.nuInh)
        #self.adjustRateNoCalib(...)

        self.logger.info('drvifallExc = ' + str(drvifallExc[0]))
        np.savetxt(os.path.join(self.folder, 'fall_exc.txt'), drvifallExc[1])



    def measureStandard(self, calibSynDrivers=True):
        '''Perform control measurements with standard parameters.'''
        self.stimParams = self.setStimulus(self.nuExcNoInh, 0)

        rateCollector = []
        for driverIndexExc in self.synapseDriverList:
            rateList = []
            for i in range(self.emulationTrials):
                rateList.append(self.emulate(driverIndexExc, calibSynDrivers=calibSynDrivers))
            rate = np.median(rateList)
            self.logger.debug('synapse driver ' + str(driverIndexExc) + ' with rate: ' + str(rate) + 'Hz')
            rateCollector.append(rate)
        rate = np.mean(rateCollector)
        np.savetxt(os.path.join(self.folder, 'fall_exc.txt'), rateCollector)



    def writeResults(self, filenameExc, filenameInh):
        filenameTmp = {}
        filenameTmp['exc'] = os.path.join(self.folder, 'delme_exc.dat')
        filenameTmp['inh'] = os.path.join(self.folder, 'delme_inh.dat')
        filenameResult = {'exc': filenameExc, 'inh': filenameInh}
        drvifallExcCalib = np.loadtxt(os.path.join(self.folder, self.filenameDrvifallExc))[:,0]
        drvioutExcCalib = np.loadtxt(os.path.join(self.folder, self.filenameDrvioutExc))[:,0]
        drvifallInhCalib = np.loadtxt(os.path.join(self.folder, self.filenameDrvifallInh))[:,0]
        drvioutInhCalib = np.loadtxt(os.path.join(self.folder, self.filenameDrvioutInh))[:,0]
        calibExc = np.concatenate((drvioutExcCalib, drvifallExcCalib))
        calibInh = np.concatenate((drvioutInhCalib, drvifallInhCalib))
        assert(len(calibExc) == default.numPresyns * len(self.neuronListBlocks) * 2)
        assert(len(calibInh) == default.numPresyns * len(self.neuronListBlocks) * 2)
        np.savetxt(filenameTmp['exc'], calibExc)
        np.savetxt(filenameTmp['inh'], calibInh)

        #write header
        weightPerBit = self.targetWeight / self.hardwareWeight
        def generateHeader(synapseType):
            header = '# lsbPerExcitatoryMicroSiemens:\n' \
                     + str(1e-6 / weightPerBit) \
                     + '\n# lsbPerInhibitoryMicroSiemens:\n' \
                     + str(1e-6 / weightPerBit / self.ratioInhExc) \
                     + '\n# drviout_base:\n' \
                     + str(default.drviout_base[synapseType]) \
                     + '\n# drvifall_base:\n' \
                     + str(default.drvifall_base[synapseType]) \
                     + '\n# calibration factors:\n'
            return header

        for synapseType in ['exc', 'inh']:
            with open(os.path.join(self.folder, filenameTmp[synapseType]), 'r') as fileRead:
                newContent = generateHeader(synapseType) + fileRead.read()
                with open(filenameResult[synapseType], 'w') as fileWrite:
                    fileWrite.write(newContent)
