from memory_profiler import profile

import logging

import random

import collections
from collections import OrderedDict

import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import socket

import network
import genetic
import parallel
import island

# import unittest
# from randomdict import RandomDict
# import warnings
# warnings.filterwarnings("always")


		

@profile
def INNTS():
	"""Give parameters"""
	gen = 30
	dataset = 'cifar10'
	# Probability of mutation being 2%
	mutationChance = 2 
	

	param = collections.OrderedDict({
		'nbNeurons': {1:32, 2:64, 3:128, 4:256, 5:512, 6:768, 7:1024},
		'nbLayers': {1:1, 2:3, 3:6, 4:9, 5:12, 6:15, 7:20},
		'activation': {1:'sigmoid', 2:'elu', 3:'selu', 4:'relu', 5:'tanh', 6:'hard_sigmoid', 7:'linear'}, 
		'optimizer': {1:'sgd', 2:'rmsprop', 3:'adagrad', 4:'adadelta', 5:'adam', 6:'adamax', 7:'nadam'},
		'dropout': {1:0.1, 2:0.15, 3:0.2, 4:0.25, 5:0.3, 6:0.4, 7:0.5}
	})

	# Initializing the MPI and testing if it's been initialized
	MPI.Init()
	print(MPI.Is_initialized())
	print(MPI.Is_finalized())
	
	# Initlialize the MPI class functions
	pd = parallel.parallelDistributed(MPI, param)
	groupSize = 7
	imc = island.islandModelCommunicators(MPI, groupSize, param)
	# print 'sg', imc.subGroup
	im = island.Island(imc.subGroup)

	# logging.basicConfig(filename='log_file')
	filename = 'output{}.log'.format(socket.gethostname())
	logger = logging.getLogger()
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.DEBUG)
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.DEBUG)

	# initialize the networks
	# one random network at every processor
	logger.debug('imc subrank: %d, %d', imc.subRank, imc.rank)
	net = network.Network(param)
	data = net.initNetwork()
	data['activation'] = im.getActivation()
	#logger.debug('imc subrank: %d, %d', imc.subRank, imc.rank)

	# MPI.COMM_WORLD.Barrier()

	bestFitness = -1
	ga = genetic.geneticAlgorithm(param)

	#print 'getranks', imc.getRanks()
	newData = {}	
	newdataFit = {}
	ndk = 0 #new datakey
	for g in range(gen):
		if bestFitness < 100:
			mum=data
			logger.debug('Initialized network @rank/subrank= %d/%d, %s', pd.rank, imc.subRank, mum)
			#Natural Selection, part I of Genetic Algorithm
			dad = imc.interIslandExchange(data)
			logger.debug('for data after exchange rank = %d, %s', pd.rank, data)
			logger.debug('for dad after exchange rank = %d, %s', pd.rank, dad)
			# now the parameters of the prev network are stored in dad
			# the parameters of the current network are stored in mum
		
			# crossover of parents
			child = ga.crossover(mum, dad)
			logger.debug('Crossover Done, generation = %d, rank = %d, child = %s', g, pd.rank, child)

			#Mutation of child
			ga.mutation(child, mutationChance)	
			logger.debug('Mutation Done, generation = %d, rank = %d, network = %s', g, pd.rank, child)

			# Every processor trains and evaluate the accuracy/fitness of the networks: parent and child
			# Set the dictionaries to network for evaluation
			fitnessMum =ga.getFitness(mum, dataset)
			fitnessChild = ga.getFitness(child, dataset)
			logger.debug('Training Done, generation = %d, rank = %d, parentFitness = %.4f, childFitness = %.4f', g, pd.rank, fitnessMum, fitnessChild)

			# if evolved child is better than the parent then change the data for next generation to consider
			if fitnessChild > fitnessMum:
				data = child
				bestFitness = fitnessChild
			else:
				data = mum
				bestFitness = fitnessChild

			if g%10 == 0:
				ndk = imc.subGroup #== pd.rank // groupSize == 0:
				newdata[ndk] = {pd.gatherData(data): pd.gatherData(bestFitness)}
				MPI.Comm.Barrier()
				logger.debug('newData = %s', newData)
				#SORTING
				for key,value in sorted(newData[ndk].iterkeys(), key=lambda (k,v): (v,k)):
					logger.debug('newData = %s, fitness = %d', key, newData[ndk][key])
					# if data is best of the group, best of three => parallel sort
					data = newData[ndk][key][0]
				# getBestRanks()
				MPI.Comm.Barrier()
				# alltoall or gather onto this processor, 7 ranks
				#Intracommunicator
				logger.debug('Intracommunicator')
				if pd.rank // groupSize == 0:
					imc.intraIslandExchange(data, groupSize)
			
			'''
			Memory Management
			Only data and bestFitness is taken, rest all are deleted
			'''
			fitnessChild = None
			fitnessMum = None
			child = None
			dad = None
			mum = None
			ndk = None
			newdata = None
	

		else:
			# Broadcast the best results to all the processors
			pd.broadcast(data)
			pd.broadcast(bestFitness)
			print data
			print bestFitness
			# And halt
			MPI.Finalize()
		

			
	MPI.Finalize()

	del gen
	del dataset
	del mutationChance 
	del param
	del pd
	del imc
	del im
	del groupSize
	del net
	del ga
	del data
	del mum
	del bestFitness
	del newData	
	del newdataFit
	del ndk


if __name__ == '__main__':
	INNTS()
		



