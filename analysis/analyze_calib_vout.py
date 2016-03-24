import sys
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et

if len(sys.argv) == 4:
    filenameXML = sys.argv[1]
    filename = sys.argv[2]
    filenameFig = sys.argv[3]
    spikeyNr = ''.join(x for x in filename if x.isdigit())
    spikeyNr = int(spikeyNr)
    assert spikeyNr < 1000, 'there should not be any other digits than spikey number in filename'
else:
    print 'please provide filename for calibration, raw data (spikeycalib.XML) and plot'
    exit()

#prepare measurement points
data = np.loadtxt(filename)
voutList = np.unique(data[:,0])
voutList = np.array(voutList, dtype=np.int)
assert len(voutList) == 20 or len(voutList) == 40, 'wrong data' #for spikey version 4 and 5

plt.figure()
#plot measurement points
for voutNr in voutList:
    dataThisVout = data[data[:,0] == voutNr]
    plt.errorbar(dataThisVout[:,2], dataThisVout[:,3], yerr=dataThisVout[:,4], c='r')

#prepare fits
slope = []
offset = []
validMin = []
validMax = []
dataXML = et.parse(filenameXML)
root = dataXML.getroot()
spikeyList = root.findall('spikey')
for spikey in spikeyList:
    if int(spikey.attrib['nr']) == spikeyNr:
        slope = spikey.find('slope').text.strip().split(';')
        offset = spikey.find('offset').text.strip().split(';')
        validMin = spikey.find('validMin').text.strip().split(';')
        validMax = spikey.find('validMax').text.strip().split(';')

#plot fits
for i in voutList:
    indexDac = np.arange(0,2.5,0.01)
    plt.plot(indexDac, indexDac * float(slope[i]) + float(offset[i]), c='gray')
    plt.axvline(float(validMin[i]), c='b')
    plt.axvline(float(validMax[i]), c='b')

plt.title('vout calibration and fit')
plt.xlabel('written [V]')
plt.ylabel('measured [V]')
plt.savefig(filenameFig)
