"""
	Takes log files of multiple char-rnn models and plots the train and validation losses
	X axis denotes number of epochs. Additionally the program outputs training and 
	validation loss values at the end of the log file.
	
	Parameters:
		windowVal - moving average window size for validation
		windowTrain - moving average size for train
		iterationsPerEpoch - number of iterations in one epoch
		maxEpoch - number of epochs

	Usage:
		python plot_loss.py [output_file_name] [model_log_file]*
		

"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os

"""
	Returns moving average of 'loss' array, where 'window' is the size of the moving window
	Assuming that 'loss' will have at least 'window' elements
"""
def movingAverage(loss, window):
	mas = []
	for i in range(len(loss)):
		j = i - window + 1
		if (j < 0):
			j = 0
		sum = 0.0
		for k in range(window):
			sum += loss[j + k]
		mas.append(sum / window)
	return mas

plotname = sys.argv[1]
while (plotname[:3] == '../'):
	plotname = plotname[-(len(plotname) - 3):]
plotname = plotname + '.png'

windowVal = 1
windowTrain = 200
iterationsPerEpoch = 171 # this should be manually configured
maxEpoch = 50

minv = 1e8
maxv = -1e8

""" 
	Plots train and validation losses for a single model
	'filename' is filename of model's log file
	'index' is used to choose the plot color
"""
def plotTrainVal(filename, index, plotLabel):
	global minv
	global maxv
	
	os.system(" egrep '*train_loss = [0-9|.]*' " + filename + " >tmpLossTrain.txt")
	os.system(" egrep '*saving checkpoint*' " + filename + " >tmpLossVal.txt")	
	tmpLossVal = open('tmpLossVal.txt', 'r')
	tmpLossTrain = open('tmpLossTrain.txt', 'r')

	valx = []
	valy = []
	for st in tmpLossVal.readlines():
		valx.append(min(float(len(valy)+1) * 1000 / iterationsPerEpoch, maxEpoch))
		valy.append(float(st.split(' ')[3].split('/')[1].split('_')[3].split('.t7')[0]))
	print "\t Last val y: ", float(st.split(' ')[3].split('/')[1].split('_')[3].split('.t7')[0])

	
	trainx = []
	trainy = []
	for st in tmpLossTrain.readlines():
	    if len(st.split(',')) > 1 :
	        if st.split(',')[1].find('train_loss')!=-1:
	            trainx.append(float(st.split(',')[0].split('m')[1].split('/')[0]) / iterationsPerEpoch)
	            trainy.append(float(st.split(',')[1].split(' ')[3]))
	            
	print "\t Last train y: ", float(st.split(',')[1].split(' ')[3])

	os.remove('tmpLossVal.txt')
	os.remove('tmpLossTrain.txt')
	
	wndVal = min(windowVal, int(0.8 * len(valy)))
	wndTrain = min(windowTrain, int(0.8 * len(trainy)))
	
	print "Train length: ", len(trainy), " \t\t window: ", wndTrain
	print "Val length: ", len(valy), " \t\t window: ", wndVal
	
	valy = movingAverage(valy, wndVal)
	trainy = movingAverage(trainy, wndTrain)
	valx = valx[:len(valy)]
	trainx = trainx[:len(trainy)]
	
	greenDiff = 50
	redBlueDiff = 50
	plt.plot(trainx, trainy, '#00' + hex(index * greenDiff)[2:] 
			+ hex(256 - index * redBlueDiff)[2:],
			label=plotLabel + " train")
	plt.hold(True)
	plt.plot(valx, valy, '#' + hex(256 - index * redBlueDiff)[2:] 
			+ hex(index * greenDiff)[2:] + '00',
			label=plotLabel + " validation")
	plt.hold(True)
	
	minv = min(minv, min(trainy))
	maxv = max(maxv, max(trainy))
	minv = min(minv, min(valy))
	maxv = max(maxv, max(valy))
	

for i in range(2, len(sys.argv)):
	plotTrainVal(sys.argv[i], i-1, sys.argv[i][4:])

minv = minv * 0.8
maxv = maxv * 1.2
#plt.gca().set_yticks(np.linspace(minv, maxv, int((maxv - minv) * 20)), minor=True)
plt.legend(loc='upper right', fontsize='x-small')
plt.gcf().savefig(plotname)
