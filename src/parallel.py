import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE



class parallelDistributed():
	# send the params to the next network
	# recv params of the prev network
	# nonBlocking
	# (size + rank - 1)%size //previous // recv from prev
	# (size + rank + 1)%size //next // send to next
	def __init__(self, MPI, param=None):
		self.comm=MPI.COMM_WORLD
		self.size = self.comm.Get_size()
		# assert comm.size > 1
		self.rank = self.comm.Get_rank()
		self.name = MPI.Get_processor_name()
	
	def nonBlockingExchange(self,data):
		# logging.warning('running using isend and irecv~~~~~~~~~~~`')
		reqSend1 = self.comm.isend(data, dest=((self.size+self.rank+1)%self.size), tag=self.rank)
		reqRecv2 = self.comm.irecv(source=((self.size+self.rank-1)%self.size), tag=self.rank-1)
		dataPrev = reqRecv2.wait()
		reqSend1.wait()
		return dataPrev

	'''
	def collateData(self, data):
		recvdata = self.comm.Gather(data, root = 0)
	'''

	def broadcast(self, data):
		data = self.comm.bcast(data, root = 0)


		
