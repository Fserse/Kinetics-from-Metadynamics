#! \usr\bin\python 3

import numpy as np
from  matplotlib import pyplot as plt

file = np.loadtxt("outsim.dat", usecols = (0,1))
file2 = np.loadtxt("nsim.dat", usecols = (0,1))

time = file[:,0]
distance = file[:,1]
start = file2[:,1]
tot = len(distance)
threshold = 5
committor = 0
counter = 0
prob = []
for elem in distance:

	if elem > threshold:
		committor = committor + 1
	else:
		committor = committor

	counter = counter + 1
	prob.append(committor/counter)

#print(prob)
# plot probability convergence
plt.plot(prob)
plt.title("Committor probability")
plt.xlabel("Sample size N")
plt.savefig("commprob.png")
plt.show()

