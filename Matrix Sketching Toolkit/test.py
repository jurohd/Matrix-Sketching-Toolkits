from sketch import SketchClass
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as LA

m 			= 2048
n 			= 1200
rank 		= 60
sketchsize	= 30
increment 	= 200


matrixA = np.random.normal(0, 1**2, (m,n))
U, s, V = np.linalg.svd(matrixA,full_matrices=False)
matrixA = np.dot(U[:,:rank]*s[:rank],V[:rank,:])
print('The matrix A has', matrixA.shape[0], 'rows and', matrixA.shape[1], 'columns.')
print('rank =',LA.matrix_rank(matrixA))
colNormsA = np.sqrt(np.sum(np.square(matrixA), 0))
print(colNormsA)
## compare the l2 norm of each col of A and C
#colNormsA = np.sqrt(np.sum(np.square(Sclass.matrixA), 0))
#print(colNormsA)

Gsketch 	= []
Gsketch_std = []
Csketch 	= []
Csketch_std = []
SRHTsketch 	= []
SRHT_std	= []
SRFTsketch 	= []
SRFT_std	= []
GRHTsketch 	= []
GRHT_std	= []
Magicsketch = []
MS_std		= []

repeat = 10
sindex = []
M = m
while sketchsize <= M:
	print(sketchsize)
	sindex.append(sketchsize)
	Sclass 		= SketchClass(matrixA, sketchsize, 3)
	Gerror 		= []
	Cerror 		= []	
	SRHTerror 	= []
	SRFTerror 	= []
	GRHTerror 	= []
	Magic_error = []
	for i in range (repeat):
		print('repeat =',i)
		Sclass.GaussianSketch()
		Gerror.append(Sclass.get_rel_error()) 
#		print('Gaussina done')
		Sclass.countSketch()
		Cerror.append(Sclass.get_rel_error()) 
#		print('Countsketch done')
		Sclass.SRHTsketch()
		SRHTerror.append(Sclass.get_rel_error()) 
#		print('SRHT done')
		Sclass.SRFTsketch()
		SRFTerror.append(Sclass.get_rel_error()) 
#		print('SRFT done')
		Sclass.GRHTsketch()
		GRHTerror.append(Sclass.get_rel_error()) 
#		print('GRHT done')
		Sclass.MagicGraphSketch()
		Magic_error.append(Sclass.get_rel_error()) 
#		print('Magicgraph done')
	Gsketch.append(np.mean(Gerror))
	Gsketch_std.append(np.std(Gerror))
	Csketch.append(np.mean(Cerror))
	Csketch_std.append(np.std(Cerror))
	SRHTsketch.append(np.mean(SRHTerror))
	SRHT_std.append(np.std(SRHTerror))
	SRFTsketch.append(np.mean(SRFTerror))
	SRFT_std.append(np.std(SRFTerror))
	GRHTsketch.append(np.mean(GRHTerror))
	GRHT_std.append(np.std(GRHTerror))
	Magicsketch.append(np.mean(Magic_error))
	MS_std.append(np.std(Magic_error))
	
	sketchsize+=increment
np.savez('errorlowrank.npz',m=m, n=n,rank=rank,sindex=sindex, Gsketch=Gsketch,Csketch=Csketch,SRHTsketch=SRHTsketch,SRFTsketch=SRFTsketch,GRHTsketch=GRHTsketch,Magicsketch=Magicsketch,Gsketch_std=Gsketch_std,Csketch_std=Csketch_std,SRHT_std=SRHT_std,SRFT_std=SRFT_std,GRHT_std=GRHT_std,MS_std=MS_std)
plt.plot(sindex, Gsketch,marker='.', label='Gaussian Sketch')
plt.plot(sindex, Csketch,marker='o', label='Count Sketch')
plt.plot(sindex, SRHTsketch,marker='v', label='SRHT')
plt.plot(sindex, SRFTsketch,marker='s', label='SRFT')
plt.plot(sindex, GRHTsketch,marker='h', label='GRHT')
plt.plot(sindex, Magicsketch,marker='+', label='MagicGraph')
plt.legend()
plt.xlabel("sketching size s",fontsize=15)
plt.ylabel(r'$||(A^TA)^{\dagger/2}\tilde{A}^T\tilde{A}(A^TA)^{\dagger/2}-P_{A^TA}||_2$',fontsize=15)
plt.title('sketching methods on A with size '+str(m)+' by '+str(n)+' and rank = '+str(rank))
#fig.savefig("plot.png")
plt.show()