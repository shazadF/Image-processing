import os
import re
import matplotlib.pyplot as plt
import csv
import numpy as np
import plotly
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

def making_x_plots ():
	clean_lines = []
	os.chdir(r'/home/ulab/Desktop/Git/bit/Plot_Image/Graph')
   	file = open('forhad-output3.txt')
	lines = file.readlines()
	#file.close()
	strToSearch=''
	for line in lines:
		strToSearch += line

	patFinder = re.compile('val_acc: \d\.\d\d\d\d', re.IGNORECASE)
	findPat_x = re.findall(patFinder,strToSearch)

    	findPat_x = [i.split(':')[1] for i in findPat_x]

	for i in range(len(findPat_x)):
   		 findPat_x[i] = float(findPat_x[i])

	return findPat_x
def making_y_plots ():
	os.chdir(r'/home/ulab/Desktop/Git/bit/Plot_Image/Graph')
	file = open('forhad-output3.txt')
	lines = file.readlines()
	strToSearch=''
	for line in lines:
		strToSearch += line

	patFinder = re.compile(' acc: \d\.\d\d\d\d', re.IGNORECASE)
	findPat_y = re.findall(patFinder,strToSearch)
	findPat_y = [i.split(':')[1] for i in findPat_y]

	for i in range(len(findPat_y)):
   		 findPat_y[i] = float(findPat_y[i])

	return findPat_y

print making_x_plots ()

print making_y_plots ()

y1 = making_y_plots ()
x1 = np.arange(100)
x2 = np.arange(100)
# print len(x1)
y2 = making_x_plots ()

f = plt.figure()
f.suptitle("Fig for accuracy", fontsize=16)
ax1 = f.add_subplot(211)
ax1.set_title("Per epoch accuracy")
ax1.plot(x1,y1)

#f = plt.figure()
#f.suptitle("Fig for validation accuracy", fontsize=16)
ax2 = f.add_subplot(212)
ax2.set_title("Per epoch validation accuracy")
ax2.plot(x2,y2)

plt.show()
