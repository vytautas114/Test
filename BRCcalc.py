#!/Users/student/Anaconda3/bin/python
import matplotlib as mpl
import sys
mpl.use('Agg')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
from scipy.interpolate import griddata
import datetime
import subprocess
from spektrasBRCfran import spektras

cdict3 = {'red':  ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.8, 1.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 0.4, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.9, 0.9),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.4),
                   (0.25, 1.0, 1.0),
                   (0.5, 1.0, 0.8),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
    
plt.register_cmap(name='BlueRed3', data=cdict3)   

#s,om,j,T,virpnum2=[float(i) for i in (sys.argv[1:])]
s=float(sys.argv[1])
om=float(sys.argv[2])
T=float(sys.argv[3])
virpnum2=int(sys.argv[4])
fig, ax=plt.subplots(figsize=(8,6))
vardas='1arew1000_'+str(s)+'_'+str(om)+'_'+str(T)+'K_'+str(virpnum2)+"_diskretus_test"
spektras(ax,s,om,T,Kvsk=virpnum2,nam="BRC/"+vardas)
ax.set_xlim([10000,15000])


fig.savefig('BRCspekt/'+vardas+'.png',dpi=300)
# cmd = ['cp outr.txt spekt/'+vardas+'_redfieldrez.txt']
# subprocess.Popen(cmd, shell=True).wait()
# cmd = ['cp outasr.txt spekt/'+vardas+'_redfieldout.txt']
# subprocess.Popen(cmd, shell=True).wait()

# print(datetime.datetime.now())
# cmd = ['../uqcfp/bin/tba.calculator_3rd_levels input_level2d outl2d.txt > outasl2d.txt']
# subprocess.Popen(cmd,shell=True).wait()

# print(datetime.datetime.now())

# file2=open("outl2d.txt")

# l1 = []
# l2=[]
# n=0
# for line in file2:
#     if n<1:
#         n+=1
#         continue
#     else:
#         l1.append(line.split())
# file2.close()        
# l1=np.array(l1).astype(np.float)
# l1=l1.transpose()  
# l1[1]=l1[1]*-1  
# x=np.arange(11000,14000,30)
# y=np.arange(11000,14000,30)
# X,Y=np.meshgrid(x,y)
# zi = griddata((l1[1], l1[0]),l1[3]*-1,(X, Y),method='linear')  

# vardas='2d_'+str(s)+'_'+str(om)+'_'+str(j)+'_'+str(T)+'K_'+str(virpnum2)
# cmd = ['cp input_level2d spekt/'+vardas+'_input.txt']
# subprocess.Popen(cmd,shell=True).wait()
# cmd = ['cp outl2d.txt spekt/'+vardas+'_rezult.txt']
# subprocess.Popen(cmd,shell=True).wait()
# cmd = ['cp outasl2d.txt spekt/'+vardas+'_output.txt']
# subprocess.Popen(cmd,shell=True).wait()

# skk=40
# pj=1 
# #ii=int(10000000/860)#12450
# #ii=np.arange(11000,14000,10)[np.diag(zi/(1*zi.max()))>0.65][0]
# plt.figure()
# plt.contourf(X,Y,zi/(1*zi.max()),np.arange(-pj,pj+2*pj/skk,2*pj/skk), cmap='BlueRed3')
# plt.colorbar()
# skk=20
# plt.contour(X,Y,zi/(1*zi.max()),np.arange(-pj,pj+2*pj/skk,2*pj/skk),colors=('k',))
# plt.plot(x,y,'k-',linewidth=1.5) 
# plt.axes().set_aspect('equal','box-forced')



# #vardas='2d_'+str(s)+'_'+str(om)+'_'+str(j)+'_'+str(T)+'K_'+str(virpnum2)
# plt.savefig('spekt/'+vardas+'.png')