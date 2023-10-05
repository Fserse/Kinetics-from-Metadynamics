#! \usr\bin\python 3

## XYZ Coordinate transformation sctipt

## Read coordintes from .xyz file

import numpy as np
import subprocess
from time import sleep
import os
import sys

atom_names = np.loadtxt("mol.xyz", skiprows=1, usecols = 0, dtype=np.str)
coordinates = np.loadtxt("mol.xyz", skiprows=1, usecols = (1,2,3))

#import atomic coordinates in XYZ format

X = coordinates[:,0]
Y = coordinates[:,1]
Z = coordinates[:,2]

natoms = len(X)

frag1 = [9,10,11,12,13,14,15,16,17,18,19,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
frag2 = [0,1,2,3,4,5,6,7,8,20,21,22,23,24,25,26,27,28,29,44,45,46,47]

def find_distance(X,Y,Z,atoms_index):
  
  # changes the distance between two selected atoms  
  
  dx = X[atoms_index[0]] - X[atoms_index[1]]
  dy = Y[atoms_index[0]] - Y[atoms_index[1]] 
  dz = Z[atoms_index[0]] - Z[atoms_index[1]]

  rho = np.sqrt(dx**2+dy**2+dz**2)  

  # Find angles

  # Case 1. 
  if dx > 0: 
    if dy >= 0:
      theta = np.arctan(dy/dx)
    elif dy < 0:
      theta = np.arctan(dy/dx) #+ np.pi

  # Case 2. 
  if dx == 0: 
    if dy == 0:
            print("not defined")
    elif y > 0:
      theta = np.pi/2
    elif y < 0:
      theta = - np.pi/2


  # Case 3. 
  if dx < 0: 
    if dy < 0:
      theta = np.arctan(dy/dx) - np.pi
    elif dy >= 0:
      theta = np.arctan(dy/dx) + np.pi

  
  phi = np.arccos(dz/rho)

  return [rho,phi,theta]



def change_distance(rho_old,phi,theta,X,Y,Z,atoms_index, flag=False):

  distances = []
  int_coords = []
  for i in frag1[1:]:

    [r,p,t] = find_distance(X,Y,Z,[i,atoms_index[0]])
  
    int_coords.append([r,p,t])
    
  if flag==False:

  # change the distance randomly

    rho_new = rho_old + (np.random.random_sample()-np.random.random_sample())*1e-2

  elif flag==True:

    rho_new =  2.30  #rho_old + 0.02 

#  print(rho_new, rho_old)
  dx_new = rho_new*np.sin(phi)*np.cos(theta)
  dy_new = rho_new*np.sin(phi)*np.sin(theta)
  dz_new = rho_new*np.cos(phi)
  
  X[atoms_index[0]] = X[atoms_index[1]] + dx_new
  Y[atoms_index[0]] = Y[atoms_index[1]] + dy_new
  Z[atoms_index[0]] = Z[atoms_index[1]] + dz_new


  for i in np.arange(0,len(frag1)-1):                                             
                  
    rho = int_coords[i][0]
    phi = int_coords[i][1]
    theta = int_coords[i][2]

    dx = rho*np.sin(phi)*np.cos(theta)
    dy = rho*np.sin(phi)*np.sin(theta)
    dz = rho*np.cos(phi)

    X[frag1[i+1]] = X[atoms_index[0]] + dx
    Y[frag1[i+1]] = Y[atoms_index[0]] + dy
    Z[frag1[i+1]] = Z[atoms_index[0]] + dz

  print("the new distance is: "+str(rho_new*1.88973))

  return [X,Y,Z]


def print2file(x,y,z,natoms,atom_names):

  with open('newmol.xyz', 'w') as f:

    f.write(str(natoms)+"\n")  
    f.write("  \n")  

    for i in np.arange(0,len(x)):
      f.write(" "+str(atom_names[i])+" \t "+str(x[i])+ " \t "+str(y[i])+" \t "+str(z[i])+" \n")  
  
    f.close()

  return
 

#print(rho,phi,theta)


def find_dividing(x,y,z,atoms_index,natoms,atom_names, flag1=True):

  maxiter = 10 
  committor = 0
  threshold = 5.1  # Product state threshold on the distance
  nsteps = 400

  if flag1 == True:

    for i in range(1,maxiter):

      [rho,phi,theta] = find_distance(X,Y,Z,atoms_index)
      [X_new,Y_new,Z_new] = change_distance(rho,phi,theta,X,Y,Z,atoms_index, flag1)
      print2file(X_new,Y_new,Z_new,natoms,atom_names)

      subprocess.run(" nohup /home/chimica2/lpratali/anaconda3/envs/cp2k_env/bin/cp2k.ssmp -i mol_solv.inp -o out.out &", shell=True)

      # monitor output
      for i in range(1,35):

        sleep(60)
        p = subprocess.run("tail -n 1 NBA_DIMER-COLVAR-1.metadynLog | awk '{print $2 }'", shell=True, stdout=subprocess.PIPE)
        distance  = float(p.stdout)

        if distance < threshold:
          flag=False
          continue
        else:
          flag=True
          break

      # kill cp2k subprocess
      pid = subprocess.run("ps -u apoli |grep -i 'cp2k' | awk '{print $1}'", shell=True, stdout=subprocess.PIPE)
      procid = int(pid.stdout)
      subprocess.run(" kill "+str(procid), shell=True)

      if flag == False:
        continue
      else:
        break

  else:

      for i in range(1,maxiter):

        [rho,phi,theta] = find_distance(X,Y,Z,atoms_index)
        [X_new,Y_new,Z_new] = change_distance(rho,phi,theta,X,Y,Z,atoms_index, flag1)
        print2file(X_new,Y_new,Z_new,natoms,atom_names)

        subprocess.run(" nohup /home/chimica2/lpratali/anaconda3/envs/cp2k_env/bin/cp2k.ssmp -i mol_solv.inp -o out.out &", shell=True)

        # monitor output
        for i in range(1,maxiter):

          sleep(1000)
          stepnum = subprocess.run("egrep -i 'step number' out.out | tail -n 1 | awk '{print $4}' ", shell=True, stdout=subprocess.PIPE)
          stepnum  = int(stepnum.stdout)

          # break cycle if jos is finished
          if stepnum == nsteps:

            p = subprocess.run("tail -n 1 NBA_DIMER-COLVAR-1.metadynLog | awk '{print $2 }'", shell=True, stdout=subprocess.PIPE)
            distance  = float(p.stdout)
      
            if distance > threshold:
              committor = committor + 1
            else:
              committor = committor

            print(committor)
            break
          else:
            continue

        # kill eventual running cp2k jobs for safety
        #pid = subprocess.run("ps -u apoli |grep -i 'cp2k' | awk '{print $1}'", shell=True, stdout=subprocess.PIPE)
        #procid = int(pid.stdout)
        #subprocess.run(" kill "+str(procid), shell=True)
  
  return 

[rho, phi, theta] = find_distance(X,Y,Z,[9,45])
[xx,yy,zz] = change_distance(rho,phi,theta,X,Y,Z,[9,45], True)
print2file(xx,yy,zz,natoms,atom_names)

#find_dividing(X,Y,Z,[9,45],natoms,atom_names)
#sleep(10)
find_dividing(X,Y,Z,[9,45],natoms,atom_names,False)
#print(committor)



