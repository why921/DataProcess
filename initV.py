import os


path='E:\ALOSPALSAR\ValidationData\ALPSRP201761520'

os.makedirs(path+'\\'+'pauli24',exist_ok=True)
os.makedirs(path+'\\'+'pauli36',exist_ok=True)
os.makedirs(path+'\\'+'pauli48',exist_ok=True)

os.makedirs(path+'\\'+'slc24',exist_ok=True)
os.makedirs(path+'\\'+'slc36',exist_ok=True)
os.makedirs(path+'\\'+'slc48',exist_ok=True)

os.makedirs(path+'\\'+'spe4bands12',exist_ok=True)

f1=open(path+'\\'+'pauli24.txt','w')
f2=open(path+'\\'+'pauli36.txt','w')
f3=open(path+'\\'+'pauli48.txt','w')

f4=open(path+'\\'+'slc24.txt','w')
f5=open(path+'\\'+'slc36.txt','w')
f6=open(path+'\\'+'slc48.txt','w')

f7=open(path+'\\'+'spe4bands12.txt','w')

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
f7.close()