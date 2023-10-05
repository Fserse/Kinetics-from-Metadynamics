#!usr/bin/bash

import os

python coordinate_transformation.py 

sleep 0.5 

cat natoms.dat newcoordinates.xyz > newmol.xyz 

sleep 0.5 
#cp mol.xyz newcoordinates.xyz &

nohup cp2k.ssmp -i mol_solv.inp -o out.out &


