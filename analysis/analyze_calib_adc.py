import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) == 3:
    filenameRaw = sys.argv[1]
    filenameFig = sys.argv[2]
else:
    print 'please provide filename for raw data and plot'
    exit()

data = np.loadtxt(filenameRaw)

slope, offset = np.polyfit(data[:,0], data[:,1], 1)

print 'fit of ADC calibration:', slope, 'x +', offset
print 'please enter these results into workstations.xml!'

plt.figure()
plt.errorbar(data[:,0], data[:,1], yerr=data[:,2], c='b')
index = np.array([np.min(data[:,0]), np.max(data[:,0])])
plt.plot(index, index * slope + offset, c='r')
plt.title('fast ADC calibration and fit')
plt.xlabel('applied voltage [mV]')
plt.ylabel('measured ADC value')
plt.xlim(0, 1000)
plt.savefig(filenameFig)
