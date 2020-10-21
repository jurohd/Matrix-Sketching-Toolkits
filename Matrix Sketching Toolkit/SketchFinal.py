import numpy as np
import math
import scipy
from numpy import linalg as LA
from scipy.sparse import csc_matrix
from scipy.linalg import hadamard
from scipy.linalg import fractional_matrix_power
import networkx as nx
def SparseGaussianGen(n, c, s):
	'''
	helper function that generates sparse gaussian
	'''
	n=s
	sigma = 1
	prob = math.log10(n)**c/n
	length = s*n
	G = np.zeros(length)
	for i in range(length):
		p = np.random.uniform(0,1)
		if p<prob:
			G[i] = np.random.normal(0,sigma)
	G = G.reshape(s,n)*math.sqrt(1/prob/s)
	return G

def realfft(matrixA):
	n = matrixA.shape[0]
	matFFT = np.fft.fft(matrixA, n=None, axis=0) / np.sqrt(n)
	if n % 2 == 1:
		tmp = int((n+1) / 2)
		idxReal = list(range(1, tmp))
		idxImag = list(range(tmp, n))
	else:
		tmp = int(n/2)
		idxReal = list(range(1, tmp))
		idxImag = list(range(tmp+1, n))
	matRealFFT = matFFT.real
	matRealFFT[idxReal, :] = matFFT[idxReal, :].real * np.sqrt(2)
	matRealFFT[idxImag, :] = matFFT[idxImag, :].imag * np.sqrt(2)
	return matRealFFT

def get_pinv_half(A):
	m, n = A.shape
	_, s, Vt = np.linalg.svd(A,full_matrices=False)
	epslon = np.finfo(float).eps
	tol = max(m, n)*epslon*s[0]
#	print(tol)
#	print(s)
	count = 0
	for i in range(len(s)):
		if s[i] > tol:
#			print(i,'no')
			s[i] = 1/s[i]
			count += 1
		else:
#			print(i,'yes')
			s[i]=0
#	print(s)
	ResultMatrix = Vt.T*s@Vt
	return ResultMatrix, Vt.T[:,:count]
	
def get_rel_error(sketchsize, matrixA, matrixC):
	x = np.random.normal(0,1,sketchsize)
	normalized_x = x/LA.norm(x)
	Ax = LA.norm(matrixA@normalized_x,2)
	SAx = LA.norm(matrixC@normalized_x,2)
	err = abs(Ax-SAx)
	rel_err = err/Ax
	return rel_err
	
def get_distortion(matrixA, matrixC):
	ATA = matrixA.T@matrixA
	CTC = matrixC.T@matrixC
	ATA_pinvhalf,V = get_pinv_half(matrixA)  #SVD,Truncate small singular values
	expression = ATA_pinvhalf@CTC@ATA_pinvhalf
#		I = np.eye(self.n)
	P_ATA = V@V.T
	distortion = LA.norm(P_ATA-expression,2)
	return distortion

#Computes an r-rank approximation of a matrix
def low_rank_approx(A, r): 

	U, s, V  = LA.svd(A, full_matrices=False)
	return U[:,:r]*s[:r]@V[:r,:]

def get_LR_error(A, C, k):
	
	Q, R = LA.qr(C.T, mode='reduced')
#	print(Q.shape)
#	print(R.shape)
#	print(A.shape)
	B = A@Q
	U, s, V  = LA.svd(B, full_matrices=False)
	Vk = V[:k,:]@Q.T
	A_til = A@Vk.T@Vk
#	print(A_til)
	error = LA.norm(A-A_til,'fro')
	print('LR_error',error)
	return error
	
class SketchClass:
	
	def __init__( self, inputdim, sketchdim ):
		self.inputdim 	= int(inputdim)
		self.sketchdim 	= int(sketchdim)
		self.randstate 	= np.random.get_state()
#		self.randnum  	= np.random.rand() 		#for testing rand state
		np.random.seed()
#		self.matrixA		= matrixA
#		self.m, self.n 		= matrixA.shape
#		self.s				= s				#skeching size
#		self.Appliedleft 	= Appliedleft
#		self.GHRT_c			= GHRT_c
#		self.state			= np.random.get_state()
#		if self.Appliedleft:
#			self.matrixS = np.zeros((self.s,self.m))
#			self.matrixC = np.zeros((self.s,self.n))
#		else:
#			self.matrixS = np.zeros((self.n,self.s))
#			self.matrixC = np.zeros((self.m,self.s))
	def Apply( self, matrixA, applyLeft = True):
#		np.random.set_state(self.randstate)
		print(np.random.rand())
		return matrixA
		
	def Matricize( self, applyLeft = True):
		temp_state = np.random.get_state()
		np.random.set_state(self.randstate)
		I = np.eye(self.inputdim)
		matrixS = self.Apply(I, applyLeft)
		np.random.set_state(temp_state)
		return matrixS

		
class GaussianSketch(SketchClass):

	def Apply( self, matrixA, applyLeft = True ):
		temp_state = np.random.get_state()
		np.random.set_state(self.randstate)
		if applyLeft:
			matrixS = np.random.normal(0, 1, (self.sketchdim,self.inputdim))/math.sqrt(self.sketchdim)
			np.random.set_state(temp_state)
			return matrixS @ matrixA
		else:
			matrixS = np.random.normal(0, 1, (self.inputdim,self.sketchdim))/math.sqrt(self.sketchdim)
			np.random.set_state(temp_state)
			return matrixA @ matrixS
		np.random.set_state(temp_state)
		
#	def Matricize( self, applyLeft = True ):
#		temp_state = np.random.get_state()
#		np.random.set_state(self.randstate)
#		if applyLeft:
#			matrixS = np.random.normal(0, 1, (self.sketchdim,self.inputdim))/math.sqrt(self.sketchdim)
#		else:
#			matrixS = np.random.normal(0, 1, (self.inputdim,self.sketchdim))/math.sqrt(self.sketchdim)
#		np.random.set_state(temp_state)
#		return matrixS
#	
#	def Sanitycheck(self,applyLeft = True):
#		temp_state = np.random.get_state()
#		np.random.set_state(self.randstate)
#		I = np.eye(self.inputdim)
#		matrixS = self.Apply(I, False)
#		matrixS_real = np.random.normal(0, 1, (self.inputdim,self.sketchdim))/math.sqrt(self.sketchdim)
#		print(matrixS)
#		print(matrixS_real)
#		print(matrixS-matrixS_real)
#		np.random.set_state(temp_state)

class countSketch(SketchClass):
	'''
	Input:
		original data matrix 				: m by n matrix A
		target skeching dimention for rows	: s
		indicator if sketch matrix is needed: boolean returnSketchMatrix
		
	Output:
		result matrix 						: s by n matrix C
		skeching matrix(if required)		: s by m matrix S
	'''
	
	def Apply( self, matrixA, applyLeft = True ):
		m, n = matrixA.shape
		temp_state = np.random.get_state()
		np.random.set_state(self.randstate)
		if applyLeft:
			matrixC = np.zeros((self.sketchdim, n))
			hashedIndices = np.random.choice(self.sketchdim, self.inputdim, replace=True)
		#		print(hashedIndices)
			randSigns = np.random.choice(2, self.inputdim, replace=True) * 2 - 1 # a m-by-1{+1, -1} vector
			A_flip = randSigns.reshape(self.inputdim, 1) * matrixA # flip the signs 
			for i in range(self.sketchdim):
				idx = (hashedIndices == i)
				matrixC[i, :] = np.sum(A_flip[idx, :], 0)	
			
		else: # sanity check needed
			matrixC = np.zeros((m, self.sketchdim))
			hashedIndices = np.random.choice(self.s, self.n, replace=True)
		#		print(hashedIndices)
			randSigns = np.random.choice(2, self.n, replace=True) * 2 - 1 # a m-by-1{+1, -1} vector
			A_flip = self.matrixA * randSigns.reshape(self.n, 1)# flip the signs 
			for i in range(self.s):
				idx = (hashedIndices == i)
				matrixC[:, i] = np.sum(A_flip[:, idx], 1)
		np.random.set_state(temp_state)
		return matrixC

		
class SRHT(SketchClass):
	'''
	Param:
		Input matrix A (m by n)
		target skeching dimention for rows	: s
		indicator if sketch matrix is needed: boolean returnSketchMatrix
	
	Output:
		result matrix 						: s by n matrix C
		skeching matrix(if required)		: s by m matrix S
	'''
	
	def Apply( self, matrixA, applyLeft = True ):
		m, n = matrixA.shape
		temp_state = np.random.get_state()
		np.random.set_state(self.randstate) 
		if applyLeft:
			#augmented dimension d(must be power of two)
			d = 1 << int(np.ceil(np.log2(m))) 
			
			H = hadamard(d)
			
			# Diagonal sign matrix
			signs = 2*np.random.binomial(n=1, p=.5, size=d) - 1  # Random signs
			D = np.diag(signs)
			
			# Uniform sampling matrix
			# Each column has a single 1; each row has at most one 1
			inds = np.random.choice(a=d, size=self.sketchdim, replace=False)
			P = csc_matrix((np.ones(self.sketchdim), (inds, range(self.sketchdim))),
						   shape=(d, self.sketchdim))
#			print(P.T.shape, H.shape, D.shape)
			matrixS = P.T@H@D*math.sqrt(1/self.sketchdim)
			print(matrixS.shape)
			print(matrixA.shape)
			return matrixS @ matrixA
		else: # sanity check needed
			#augmented dimension d(must be power of two)
			d = 1 << int(np.ceil(np.log2(n))) 
			
			H = hadamard(d)
			
			# Diagonal sign matrix
			signs = 2*np.random.binomial(n=1, p=.5, size=d) - 1  # Random signs
			D = np.diag(signs)
			
			# Uniform sampling matrix
			# Each column has a single 1; each row has at most one 1
			inds = np.random.choice(a=d, size=self.sketchdim, replace=False)
			P = csc_matrix((np.ones(self.sketchdim), (inds, range(self.sketchdim))),
						   shape=(d, self.sketchdim))
			matrixS = P.T@H@D*math.sqrt(1/self.sketchdim)
			matrixC = matrixA @ matrixS
		
		np.random.set_state(temp_state)

class GRHT(SRHT):
	'''
	Param:
		Input matrix A (m by n)
		target skeching dimention for rows	: s
		indicator if sketch matrix is needed: boolean returnSketchMatrix
	
	Output:
		result matrix 						: s by n matrix C
		skeching matrix(if required)		: s by m matrix S
	'''
	def Apply( self, matrixA, applyLeft = True, constant = 6.4 ):
		m, n = matrixA.shape
		temp_state = np.random.get_state()
		np.random.set_state(self.randstate) 
		matrixC = SRHT.Apply(self, matrixA, applyLeft)
		G = SparseGaussianGen(n, constant, self.sketchdim)
		np.random.set_state(temp_state)
		return G @ matrixC
			
class SRFT(SketchClass):
	'''
	The Subsampled Randomized Fourier Transform
	Param:
		Input matrix A (m by n)
		target skeching dimention for rows	: s
		indicator if sketch matrix is needed: boolean returnSketchMatrix
	
	Output:
		result matrix 						: s by n matrix C
		skeching matrix(if required)		: s by m matrix S
	'''
	def Apply( self, matrixA, applyLeft = True ):
		m, n = matrixA.shape
		temp_state = np.random.get_state()
		np.random.set_state(self.randstate) 
		if applyLeft:
			randSigns = np.random.choice(2, m) * 2 - 1
			randIndices = np.random.choice(m, self.sketchdim, replace=False)
			matrixA_flip = matrixA * randSigns.reshape(m, 1)
			matrixA_flip = realfft(matrixA_flip)
			matrixC = matrixA[randIndices, :] * np.sqrt(m/self.sketchdim)
		else: # sanity check needed
			randSigns = np.random.choice(2, n) * 2 - 1
			randIndices = np.random.choice(n, self.sketchdim, replace=False)
			matrixA_flip = matrixA.T * randSigns.reshape(n, 1)
			matrixA_flip = realfft(matrixA_flip)
			matrixC = matrixA[randIndices, :].T * np.sqrt(n/self.sketchdim)
			matrixC = matrixC.T
		np.random.set_state(temp_state)
		return matrixC
			
class MagicGraph(SketchClass):
	'''
	|X|=m, |Y|=s, G(X,Y;E)
	
	Param:
		Input matrix A (m by n)
		target skeching dimention for rows	: s
		indicator if sketch matrix is needed: boolean returnSketchMatrix
	
	Output:
		result rank preserving matrix 		: s by n matrix C
		skeching matrix(if required)		: s by m matrix S
	'''
		
	def Apply( self, matrixA, applyLeft = True, num_randmatch = 2):
		m, n = matrixA.shape
		temp_state = np.random.get_state()
		np.random.set_state(self.randstate)
		if applyLeft: #sketching matrix apply on left
			matrixC = np.zeros((self.sketchdim, n))
			# k m-by-1 vector showing which row to match
			rand_perm = []
			for i in range (num_randmatch):
				rand_perm.append(np.random.permutation(m).reshape(m,1))
			match 		= np.concatenate(rand_perm,axis = 1)
			multiple 	= m//self.sketchdim
			
#			scaling 	= 1/2/multiple #scaling factor
			
			if self.sketchdim*multiple != m:
				dim 		= self.sketchdim*(multiple+1)
				diff 		= dim-m 
				rc = []
				for i in range (num_randmatch):
					rc.append(np.random.choice(m, diff,replace=False).reshape(diff,1))
				dilation = np.concatenate(rc,axis = 1)
#				dilation 	= np.full((diff,num_randmatch), -1)
				match 		= np.concatenate((match,dilation))
				np.random.shuffle(match)
		#			print(match)
			# divide one side into s groups
			match = np.split(match,self.sketchdim)
			
			length = len(match)
			for i in range(length):
				idx = match[i].flatten()
		#			print('before',idx.shape)
				idx = idx[idx>=0]
		#			print('after',idx.shape)
				rand_coeff = np.random.choice(2, (idx.shape[0], 1), replace=True) * 2 - 1
				matrixC[i, :] = np.sum(rand_coeff/math.sqrt(2)*matrixA[idx, :], 0)
			return matrixC
			
		else: # sanity check needed
			matrixC = np.zeros((m, self.sketchdim))
			# 2 m-by-1 vector showing which row to match
			rand_perm1 	= np.random.permutation(n).reshape(n,1)
			rand_perm2 	= np.random.permutation(n).reshape(n,1)
			match 		= np.concatenate((rand_perm1,rand_perm2),axis = 1)
			multiple 	= n//self.sketchdim
			
			if self.sketchdim*multiple != n:
				dim 		= self.sketchdim*(multiple+1)
				diff 		= dim-n
				dilation 	= np.full((diff,2), -1)
				match 		= np.concatenate((match,dilation))
				np.random.shuffle(match)
		#			print(match)
			# divide one side into s groups
			match = np.split(match,self.sketchdim)
			
			length = len(match)
			CT = matrixC.T
			for i in range(length):
				idx = match[i].flatten()
		#			print('before',idx.shape)
				idx = idx[idx>=0]
		#			print('after',idx.shape)
				rand_coeff = np.random.uniform(0,1,(idx.shape[0], 1))
				CT[i, :] = np.sum(rand_coeff*matrixA.T[idx, :], 0)
			
			return CT
		
	

class ExpanderGraph(SketchClass):
	'''
	|X|=m, |Y|=s, G(X,Y;E)
	
	Param:
		Input matrix A (m by n)
		target skeching dimention for rows	: s
		indicator if sketch matrix is needed: boolean returnSketchMatrix
	
	Output:
		result rank preserving matrix 		: s by n matrix C
		skeching matrix(if required)		: s by m matrix S
	'''
	def Apply( self, matrixA, applyLeft = True, sparsity = 2):
		m, n = matrixA.shape
		temp_state = np.random.get_state()
		np.random.set_state(self.randstate)
		if applyLeft: #sketching matrix apply on left
			matrixS = np.zeros((self.sketchdim, m))
			for i in range(m):
				randperm = np.random.permutation(self.sketchdim)
				mapped_rows = randperm[:sparsity]
				randSigns = np.random.choice(2, sparsity, replace=True) * 2 - 1
#				print(sparsity, mapped_rows, randSigns)
				matrixS[mapped_rows,i]=randSigns/math.sqrt(sparsity)
#			print(matrixS)
#			print('sketchsize=',self.sketchdim,'NNZ=',np.count_nonzero(matrixS))
			return matrixS@matrixA