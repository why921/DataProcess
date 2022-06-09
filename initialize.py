import os


sar='ALPSRP233871400'
path='E:\ALOSPALSAR\TrainData'
os.makedirs(path+'\\'+sar+'\\'+sar+'_24',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_SLC_24',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_spe_24',exist_ok=True)
os.makedirs(path+'\\'+sar+'\\'+sar+'_sep_24_4bands',exist_ok=True)

f1=open(path+'\\'+sar+'\\'+sar+'_24.txt','w')
f2=open(path+'\\'+sar+'\\'+sar+'_SLC_24.txt','w')
f3=open(path+'\\'+sar+'\\'+sar+'_spe_24.txt','w')
f4=open(path+'\\'+sar+'\\'+sar+'_spe_24_4bands.txt','w')

f1.close()
f2.close()
f3.close()
f4.close()
