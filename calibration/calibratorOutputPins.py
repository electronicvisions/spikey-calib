import pyNN.hardware.spikey as pynn
import os, sys, numpy, time, pwd
import datetime as dt

class CalibratorOutputPins(object):

    def __init__(self, workstation=None, numPinBlocks=1, numVsteps=5, numRuns=3, randomSeed=[3210]):

        self.workstation = workstation
        self.seeds = randomSeed
        self.numPinBlocks = numPinBlocks
        self.numVsteps = numVsteps
        self.numRuns = numRuns

        self.voltageLimLow = 0.4 #V
        self.voltageLimHigh = 1.4 #V
        self.polyDegree = 3
        self.limMemStd = 0.01 #mV for checking whether occasionally occuring digital spikes are "ghost spikes"

        ####################################################################
                           #  experiment parameters #
        ####################################################################

        self.duration = 10000.0 # msec

        self.neuronParams = {
                        'v_reset'    : -80.0, # mV
                        'e_rev_I'    : -80.0, # mV
                        'v_rest'     : -75.0, # mV
                        'v_thresh'   : -55.0, # mV
                        'g_leak'     :  250.0 # nS large leakage conductance to pull strongly to rest potential
        }

        ####################################################################
                      #  procedural experiment description #
        ####################################################################

        self.result = {}
        self.result['polyFitOutputPins'] = {}
        self.result['polyFitNeuronMems'] = {}
        self.rawData = None
        self.noSpikesTotal = 0
        self.chipVersion = None


    def calib(self):

        self.result['datetime'] = dt.datetime.now()
        self.result['temperature'] = 'TODO'
        self.result['person'] = pwd.getpwuid(os.getuid()).pw_name

        # one setup is necessary in order to determine spikey version
        pynn.setup(workStationName=self.workstation, calibOutputPins=False, calibNeuronMems=False, calibTauMem=False, calibSynDrivers=False, calibVthresh=False, calibBioDynrange=False)
        self.chipVersion = pynn.hardware.chipVersion()

        for block in range(2):
            if not self.chipVersion == 4 or block == 1:
                for pin in range(4):

                    lower = -90.
                    upper = -40.
                    step = (upper - lower) / self.numVsteps
                    for vrest in numpy.arange(lower, upper+step/2., step):

                        pin_in = numpy.nan
                        pin_out = numpy.nan

                        for pinBlock in range(self.numPinBlocks):

                            neuron = block * 192 + pinBlock * 4 + pin

                            # necessary setup
                            mappingOffset = neuron
                            if self.chipVersion == 4:
                                mappingOffset = neuron % 192
                            pynn.setup(useUsbAdc=True, workStationName=self.workstation, mappingOffset=mappingOffset, rng_seeds=self.seeds, avoidSpikes=True, \
                                       calibOutputPins=False, calibNeuronMems=False, calibTauMem=False, calibSynDrivers=False, calibVthresh=False)

                            self.neuronParams['v_rest'] = vrest

                            # set up network
                            n = pynn.create(pynn.IF_facets_hardware1,self.neuronParams,n=1)

                            pynn.record_v(n, '')
                            pynn.record(n, '')

                            #http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
                            traceSum = 0.0
                            traceSumSqr = 0.0
                            # execute the experiment in a loop
                            for i in range(self.numRuns):
                                pynn.run(self.duration, translateToBioVoltage=False)
                                if i==0:
                                    pin_in = pynn.hardware.hwa.vouts[neuron/192,neuron%2 + 2]
                                else:
                                    assert pin_in == pynn.hardware.hwa.vouts[neuron/192,neuron%2 + 2], 'vout should not differ'
                                mem = pynn.membraneOutput
                                memMean = mem.mean()
                                traceSum += memMean
                                traceSumSqr += numpy.power(memMean, 2)
                                noSpikes = len(pynn.spikeOutput[1])
                                if not float(noSpikes) / self.duration * 1e3 == 0:
                                    self.noSpikesTotal += noSpikes
                                    print 'there are', noSpikes, 'spikes on the membrane (most likely / hopefully ghost spikes)'
                                    assert mem.std() < self.limMemStd, 'digital spikes and spikes on the membrane found!'
                            pin_out = traceSum / self.numRuns
                            pin_out_std = (traceSumSqr - (numpy.power(traceSum, 2) / self.numRuns)) / (self.numRuns - 1)
                            pynn.end()

                            print 'For neuron',neuron,'the written voltage',pin_in,'appeared on the scope as',pin_out,'/2'

                            #save raw data
                            newData = numpy.vstack((neuron, pin_in, pin_out, numpy.sqrt(pin_out_std))).T
                            if self.rawData == None:
                                self.rawData = newData
                            else:
                                self.rawData = numpy.vstack((self.rawData, newData))



        def filter_and_fit(dataset):
            #filter data
            dataset = numpy.atleast_2d(dataset)
            dataToFit = numpy.atleast_2d(dataset[dataset[:,1] >= self.voltageLimLow])
            dataToFit = numpy.atleast_2d(dataToFit[dataToFit[:,1] <= self.voltageLimHigh])
            noPins = len(numpy.unique(numpy.array(dataset[:,0] / 192, numpy.int) * 4 + dataset[:,0] % 4))
            assert (len(dataset) - len(dataToFit)) % noPins == 0, 'discarding data failed'
            print 'discarded', (len(dataset) - len(dataToFit)) / noPins, 'data points'

            #fit polynomial
            return numpy.polyfit(dataToFit[:,2], dataToFit[:,1], self.polyDegree)

        for block in range(2):
            if self.chipVersion == 4 and block == 0:
                continue

            for pin in range(4):
                #data for output pin calibration
                dataOnePin = self.rawData[numpy.array(self.rawData[:,0] / 192, numpy.int) * 4 + self.rawData[:,0] % 4 == block * 4 + pin]

                #calculate mean over neurons with same pin
                vouts = numpy.unique(dataOnePin[:,1])
                mean = []
                std = []
                for vout in vouts:
                    mean.append(numpy.mean(dataOnePin[dataOnePin[:,1] == vout][:,2]))
                    std.append(numpy.std(dataOnePin[dataOnePin[:,1] == vout][:,2]))
                dataOnePinMean = numpy.vstack((numpy.zeros_like(vouts), vouts, mean, std)).T

                self.result['polyFitOutputPins']['pin' + str(block * 4 + pin)] = filter_and_fit(dataOnePinMean)

                for pinBlock in range(self.numPinBlocks):
                    neuron = block * 192 + pinBlock * 4 + pin
                    #data for membrane calibration of single neurons
                    dataOneNeuron = self.rawData[self.rawData[:,0] == neuron]
                    self.result['polyFitNeuronMems']['neuron' + str(neuron).zfill(3)] = filter_and_fit(dataOneNeuron)



        print 'total number of spikes', self.noSpikesTotal, 'in', len(numpy.unique(numpy.array(self.rawData[:,0] / 192, numpy.int) * 4 + self.rawData[:,0] % 4)) * (self.numVsteps + 1) * self.numPinBlocks * self.numRuns, 'runs'
        return self.result, self.rawData
