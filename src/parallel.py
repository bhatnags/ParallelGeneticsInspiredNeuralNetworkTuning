import mpi4py
#from mpi4py import MPI
#from mpi4py.MPI import ANY_SOURCE
mpi4py.rc(initialize=False, finalize=False)


'''
Class for parallel functions:
	Initializes the MPI features
		Non-Blocking Exchange
		Broadcast
'''
class parallelDistributed():

	def __init__(self, MPI, param=None):
		self.comm=MPI.COMM_WORLD
		self.size = self.comm.Get_size()
		# assert comm.size > 1
		self.rank = self.comm.Get_rank()
		self.name = MPI.Get_processor_name()



	'''
	Non-Blocking Exchange
		Send the params to the next network
		Receive params of the previous network
		(size + rank - 1)%size //previous // recv from prev
		(size + rank + 1)%size //next // send to next
		Returns the data of the previous network
	'''
	def nonBlockingExchange(self,data):
		reqSend1 = self.comm.isend(data, dest=((self.size+self.rank+1)%self.size), tag=self.rank)
		reqRecv2 = self.comm.irecv(source=((self.size+self.rank-1)%self.size), tag=self.rank-1)
		dataPrev = reqRecv2.wait()
		reqSend1.wait()
		return dataPrev


	'''
	Broadcast:
		Data Broadcasted from the root to all the other sockets/nodes
	'''
	def broadcast(self, data):
		data = self.comm.bcast(data, root = 0)


		
	'''
	Get all the data on the root
	'''
	def collateData(self, fitData):
		recvdata = self.comm.Gather(fitData, root = 0)
