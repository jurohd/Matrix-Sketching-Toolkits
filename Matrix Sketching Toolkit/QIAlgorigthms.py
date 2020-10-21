from SketchFinal import MagicGraph
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

def RankEstBounded(A,s):
	m,n = A.shape
	c = 1
#	z = c * min(s, 2)
	z = c * min(s, math.sqrt(np.count_nonzero(A)/math.log(m)))
	MG_L = MagicGraph(m,z)
	MG_R = MagicGraph(z,n)
#	MG_L.apply(A)
#	MG_R.apply(A)
	S1 = MG_L.Matricize(applyLeft=True)
	R1 = MG_R.Matricize(applyLeft=False)
	#print(S1,R1)
#	print(S1.shape,R1.shape)
	K1 = S1@A@R1
	#print(K1)
	k1 = LA.matrix_rank(K1)
	print('k1 =',k1)
	if k1<z/c or (k1==z/c and z/c==s):
		return k1
	MG_L = MagicGraph(m,c*s)
	MG_R = MagicGraph(c*s,n)
	
	S2 = MG_L.Matricize(applyLeft=True)
	R2 = MG_R.Matricize(applyLeft=False)
	K2 = S2@matrixA@R2
	k2 = LA.matrix_rank(K2)
	print('k2 =',k2)
	return min(s,k2)

def RankEst(A, lbd):
	t = 0
	m,n = A.shape
	while 1:
		print('t=',t)
		st = m**(0.5+t/lbd)
		print('st=',st)
		k0 = RankEstBounded(A, st)
		if k0<st:
			return k0
		t+=1

m 			= 1024
n 			= 600
rank 		= 80
s		 	= math.sqrt(m)

matrixA = np.random.normal(0, 1**2, (m,n))
U, S, V = np.linalg.svd(matrixA,full_matrices=False)
matrixA = np.dot(U[:,:rank]*S[:rank],V[:rank,:])
print('The matrix A has', matrixA.shape[0], 'rows and', matrixA.shape[1], 'columns.')
print('rank =',LA.matrix_rank(matrixA))
rankbounded = RankEstBounded(matrixA,s)
print('rankbounded =',rankbounded)
rankest = RankEst(matrixA, lbd=10)
print('rankest =',rankest)