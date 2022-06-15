import os


sar='ALPSRP277371560'
path='E:\ALOSPALSAR\TrainData'
os.makedirs(path+'\\'+sar+'\\'+sar+'_24',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_SLC_24',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_spe_24',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_spe_24_4bands',exist_ok=True)

os.makedirs(path+'\\'+sar+'\\'+sar+'_36',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_SLC_36',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_spe_36',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_spe_36_4bands',exist_ok=True)

os.makedirs(path+'\\'+sar+'\\'+sar+'_48',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_SLC_48',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_spe_48',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_spe_48_4bands',exist_ok=True)

f1=open(path+'\\'+sar+'\\'+sar+'_24.txt','w')
f2=open(path+'\\'+sar+'\\'+sar+'_SLC_24.txt','w')
f3=open(path+'\\'+sar+'\\'+sar+'_spe_24.txt','w')
f4=open(path+'\\'+sar+'\\'+sar+'_spe_24_4bands.txt','w')

f11=open(path+'\\'+sar+'\\'+sar+'_36.txt','w')
f22=open(path+'\\'+sar+'\\'+sar+'_SLC_36.txt','w')
f33=open(path+'\\'+sar+'\\'+sar+'_spe_36.txt','w')
f44=open(path+'\\'+sar+'\\'+sar+'_spe_36_4bands.txt','w')

f111=open(path+'\\'+sar+'\\'+sar+'_48.txt','w')
f222=open(path+'\\'+sar+'\\'+sar+'_SLC_48.txt','w')
f333=open(path+'\\'+sar+'\\'+sar+'_spe_48.txt','w')
f444=open(path+'\\'+sar+'\\'+sar+'_spe_48_4bands.txt','w')

f1.close()
f2.close()
f3.close()
f4.close()

f11.close()
f22.close()
f33.close()
f44.close()

f111.close()
f222.close()
f333.close()
f444.close()