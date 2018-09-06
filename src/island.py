import random

import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

import parallel



class Island():
	
	def __init__(self, sg):
		self.subGroup = sg
	
	'activation function of the networks in the island'
	def getActivation(self):
		
		activation = None
		if self.subGroup == 0:
			activation = 'sigmoid'
		elif self.subGroup == 1:
			activation = 'elu'
		elif self.subGroup == 2:
			activation = 'selu'
		elif self.subGroup == 3:
			activation = 'relu'
		elif self.subGroup == 4:
			activation = 'tanh'
		elif self.subGroup == 5:
			activation = 'hard_sigmoid'
		elif self.subGroup == 6:
			activation = 'linear'
		return activation

	'optimizer function of the networks in the island'
	def getOptimizer(self):
		
		optimizer = None
		if self.subGroup == 0:
			optimizer = 'sgd'
		elif self.subGroup == 1:
			optimizer = 'rmsprop'
		elif self.subGroup == 2:
			optimizer = 'adagrad'
		elif self.subGroup == 3:
			optimizer = 'adadelta'
		elif self.subGroup == 4:
			optimizer = 'adam'
		elif self.subGroup == 5:
			optimizer = 'adamax'
		elif self.subGroup == 6:
			optimizer = 'nadam'
		return optimizer

	'neurons of the networks in the island'
	def getNeurons(self):
		
		neurons = None
		if self.subGroup == 0:
			neurons = 32
		elif self.subGroup == 1:
			neurons = 64
		elif self.subGroup == 2:
			neurons = 128
		elif self.subGroup == 3:
			neurons = 256
		elif self.subGroup == 4:
			neurons = 512
		elif self.subGroup == 5:
			neurons = 768
		elif self.subGroup == 6:
			neurons = 1024
		return neurons		
		
	'Layers of the networks in the island'
	def getLayers(self):
		
		Layers = None
		if self.subGroup == 0:
			Layers = 1
		elif self.subGroup == 1:
			Layers = 3
		elif self.subGroup == 2:
			Layers = 6
		elif self.subGroup == 3:
			Layers = 9
		elif self.subGroup == 4:
			Layers = 12
		elif self.subGroup == 5:
			Layers = 15
		elif self.subGroup == 6:
			Layers = 20
		return Layers		
		
	'Dropout of the networks in the island'
	def getDropout(self):
		
		Dropout = None
		if self.subGroup == 0:
			Dropout = 0.1
		elif self.subGroup == 1:
			Dropout = 0.15
		elif self.subGroup == 2:
			Dropout = 0.2
		elif self.subGroup == 3:
			Dropout = 0.25
		elif self.subGroup == 4:
			Dropout = 0.3
		elif self.subGroup == 5:
			Dropout = 0.4
		elif self.subGroup == 6:
			Dropout = 0.5
		return Dropout		
		
		
		
		
class islandModelCommunicators(parallel.parallelDistributed):
	
	
	def __init__(self, MPI, groupSize, param=None):

		parallel.parallelDistributed.__init__(self, MPI, param=None)

		self.group=self.comm.Get_group()

		# split the communicator in groups of three
		# taking N = 21
		# 7 islands
		self.subGroup = self.rank // groupSize
		self.subComm = MPI.Comm.Split(self.comm, self.subGroup, self.rank)

		self.subSize, self.subRank = self.subComm.Get_size(), self.subComm.Get_rank()

	# get all the ranks of a best fitness networks within each of the seven subgroup
	def getBestRanks(self):
		lst = []
		start = self.subGroup
		lst.extend(str(start))
		for _ in range(self.subGroup):
			lst.extend(str(start+3))
		return lst

	# Every Island knows : their rank, subrank & subgroup
	def interIslandExchange(self,data):
		# logging.warning('running using isend and irecv~~~~~~~~~~~`')
		reqSend1 = self.comm.isend(data, dest=((self.size+self.rank+3)%self.size), tag=self.rank)
		reqRecv2 = self.comm.irecv(source=((self.size+self.rank-3)%self.size), tag=self.rank-1)
		dataPrev = reqRecv2.wait()
		reqSend1.wait()
		return dataPrev
	
	'''
	The island to which the netwrok is to be sent
	'''
	def getIsland():
		rand = random.randint(0, self.subSize)
		if rand == self.subGroup:
			getIsland()
		return rand

	'''
	The Network of the Island which is to be sent
	'''

	def getCurrIsland():
		return self.subGroup

	'''
	get the network to be sorted
	'''
	def getNetSort():	
		boolFlag = None
		sg = self.getCurrIsland()
		

	def getNetwork(groupSize): 
		isl = self.getIsland()
		net = isl*groupSize
		#while(self.subRank != net//groupSize):
			
		'''
		for the list
			if isl == (in that list) // groupSize
			return in that list
		'''
		



	# Every Island knows : their rank, subrank & subgroup
	def intraIslandExchange(self,data, groupSize):
		# logging.warning('running using isend and irecv~~~~~~~~~~~`')
		sendTo = self.getNetwork(groupSize)
		recvFrom = self.getNetwork(groupSize)
		reqSend1 = self.comm.isend(data, dest=(sendTo), tag=ANY_TAG)
		reqRecv2 = self.comm.irecv(source=(recvFrom), tag=ANY_TAG)
		dataPrev = reqRecv2.wait()
		reqSend1.wait()
		return dataPrev


