import numpy as np
import matplotlib.pyplot as plt

a = np.load('errorlowrank.npz')
Gsketch 	= a['Gsketch']
Csketch 	= a['Csketch']
SRHTsketch 	= a['SRHTsketch']
SRFTsketch 	= a['SRFTsketch']
GRHTsketch 	= a['GRHTsketch']
Magicsketch = a['Magicsketch']
sindex		= a['sindex']
m = a['m']
n = a['n']
rank=a['rank']
G_std = a['Gsketch_std']
C_std = a['Csketch_std']
SRHT_std = a['SRHT_std']
SRFT_std = a['SRFT_std']
GRHT_std = a['GRHT_std']
MS_std = a['MS_std']

#print(G_std,C_std,SRHT_std,SRFT_std,GRHT_std,MS_std)


standard = 1
print(SRHTsketch)
print(Gsketch)
plt.plot(sindex, Gsketch/standard,marker='.', label='Gaussian Sketch')
plt.plot(sindex, Csketch/standard,marker='o', label='Count Sketch')
plt.plot(sindex, SRHTsketch/standard,marker='v', label='SRHT')
plt.plot(sindex, SRFTsketch/standard,marker='s', label='SRFT')
plt.plot(sindex, GRHTsketch/standard,marker='h', label='GRHT')
plt.plot(sindex, Magicsketch/standard,marker='+', label='MagicGraph')
plt.legend(fontsize=20)
plt.xlabel("sketching size s",fontsize=30)
plt.ylabel(r'$||(A^TA)^{\dagger/2}\tilde{A}^T\tilde{A}(A^TA)^{\dagger/2}||_2$',fontsize=30)
#plt.ylabel(r'$\frac{||SAx||_2-||Ax||_2}{||Ax||_2}$',fontsize=15)
plt.title('sketching methods on A with size '+str(m)+' by '+str(n)+' and rank = '+str(rank), fontsize=30)
#fig.savefig("plot.png")
plt.show()
#plt.cla()
#
#plt.plot(sindex, G_std,marker='.', label='Gaussian Sketch')
#plt.plot(sindex, C_std,marker='o', label='Count Sketch')
#plt.plot(sindex, SRHT_std,marker='v', label='SRHT')
#plt.plot(sindex, SRFT_std,marker='s', label='SRFT')
#plt.plot(sindex, GRHT_std,marker='h', label='GRHT')
#plt.plot(sindex, MS_std,marker='+', label='MagicGraph')
#plt.legend()
#plt.xlabel("sketching size s")
#plt.ylabel('magnitude')
#plt.title('Std for different sketching methods')
#plt.show()