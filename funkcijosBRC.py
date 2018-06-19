import subprocess
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import datetime
from numba import jit

def rink(sk):
        ll=[]
        for i in range(sk-1):
            for j in range(i+1,sk):
                ll.append([i,j])        
        return ll    

def  lor(x,x0,H):
    return 1/3.1415*(1/2*H)/((x-x0)*(x-x0)+(1/2*H)*(1/2*H))    
    
def makeinput(EnergG, EnergE, dipol,T,CorrD,CorrOffd):
    numG=np.size(EnergG)
    numE=np.size(EnergE)
    file=open('input_level','w')
    file.write('key-setup-complete\n')
    file.write('#number of levels in g band and e band\n')
    file.write('%d\n%d\n'% (numG, numE))
    file.write('#energies\n')
    for i in EnergG:
        file.write('%.3f\n'% i)
    file.write('\n')
    for i in EnergE:
        file.write('%.3f\n'% i)
    file.write('\n')
    file.write('#number of dipoles\n')
    file.write('%d\n' % np.size(dipol[:,0]))
    file.write('#from, to, dipoles \n')
    for i in dipol:
        file.write('%d %d %.6f %.6f %.5f\n' % (i[0],i[1],i[2],i[3],i[4]))
        
    file.write('#bath\n#temperature\n')
    file.write('%f\n'% T)
    file.write('#ZPL\n')
    file.write('%.2f\n'%0)
    file.write('#number of SDF\n%d\n'% 1)
    file.write('SDF.txt\n')
    file.write('#offdiag\n')
    for ii in range(numG+numE):
        for jj in range(ii+1):
            file.write('%.6f '%(CorrOffd[ii,jj]))
            #if ii==jj:
            #    file.write('1 ')
            #else:
            #    file.write('0 ')
        file.write('\n')    
    file.write('#diag\n')
    for ii in range(numG+numE):
        for jj in range(ii+1):
            file.write('%.6f '%CorrD[ii,jj])
            #if ii==jj:
            #    file.write('1 ')
            #else:
            #    file.write('0 ')
        file.write('\n')    
        
    file.write('1 0 0\n1 0 0\n')
    
    file.write('0\n')
    
    file.write('10000\n')
    file.write('20000\n')
    file.write('10000\n')

    file.close()
    
def makeinput2d(EnergG, EnergE,EnergF, dipol,T,CorrD,CorrOffd,expattern):
    numG=np.size(EnergG)
    numE=np.size(EnergE)
    numF=np.size(EnergF)
    file=open('input_level2d','w')
    file.write('key-setup-complete\n')
    file.write('#number of levels in g band and e band and f bands\n')
    file.write('%d\n%d\n%d\n'% (numG, numE, numF))
    file.write('#energies\n')
    for i in EnergG:
        file.write('%.3f\n'% i)
    file.write('\n')
    for i in EnergE:
        file.write('%.3f\n'% i)
    file.write('\n')
    for i in EnergF:
        file.write('%.3f\n'% i)    
    file.write('\n')
    file.write('#number of dipoles\n')
    file.write('%d\n' % np.size(dipol[:,0]))
    file.write('#from, to, dipoles \n')
    for i in dipol:
        file.write('%d %d %.6f %.6f %.6f\n' % (i[0],i[1],i[2],i[3],i[4]))
        
    file.write('#bath\n#temperature\n')
    file.write('%f\n'% T)
    file.write('#ZPL\n')
    file.write('%.2f\n'% 1)
    file.write('#number of SDF\n%d\n'% 1)
    file.write('SDF.txt\n')
    file.write('#offdiag\n')
    for ii in range(numG+numE+numF):
        for jj in range(ii+1):
            file.write('%.10f '%(CorrOffd[ii,jj]))
            #if ii==jj:
            #    file.write('1 ')
            #else:
            #    file.write('0 ')
        file.write('\n')    
    file.write('#diag\n')
    for ii in range(numG+numE+numF):
        for jj in range(ii+1):
            file.write('%.10f '%CorrD[ii,jj])
            #if ii==jj:
            #    file.write('1 ')
            #else:
            #    file.write('0 ')
        file.write('\n') 
    file.write('#dummy zero\n0\n')
    
    file.write("# pattern\n")
    
    file.write(str(expattern)+'\n')
    
    file.write('#WTW file\n2\n')
        
    file.write('1 0 0\n1 0 0\n')
    file.write('1 0 0\n1 0 0\n')
    
    file.write('0\n')
    file.write('#Fre1\n')
    file.write('-11000\n')
    file.write('-14000\n')
    file.write('#Fre2\n')
    file.write('11000\n')
    file.write('14000\n')
    
    file.write('100\n')
    
    file.write("delay t2: one fs = 0.0001884\n")
    file.write("%f" % 0)
    
    

    file.close()

    

def deriniai2(numM,numL):
    class unique_element:
        def __init__(self,value,occurrences):
            self.value = value
            self.occurrences = occurrences

    def perm_unique(elements):
        eset=set(elements)
        listunique = [unique_element(i,elements.count(i)) for i in eset]
        u=len(elements)
        return perm_unique_helper(listunique,[0]*u,u-1)

    def perm_unique_helper(listunique,result_list,d):
        if d < 0:
            yield tuple(result_list)
        else:
            for i in listunique:
                if i.occurrences > 0:
                    result_list[d]=i.value
                    i.occurrences-=1
                    for g in  perm_unique_helper(listunique,result_list,d-1):
                        yield g
                    i.occurrences+=1
    a=[]

    for i in range(numL):
        for j in range(i+1):
            a += list(perm_unique([j,i]+[0]*(numM-2)))
    return np.array(a)        

def visideriniai(saitnum, virpnum):
    maxRange = np.ones(saitnum,dtype=int)*virpnum
    virp = np.array([i for i in product(*(range(i) for i in maxRange)) ])
    return virp

def deriniairev(numM,numKv):
    class unique_element:
        def __init__(self,value,occurrences):
            self.value = value
            self.occurrences = occurrences

    def perm_unique(elements):
        eset=set(elements)
        listunique = [unique_element(i,elements.count(i)) for i in eset]
        u=len(elements)
        return perm_unique_helper(listunique,[0]*u,u-1)

    def perm_unique_helper(listunique,result_list,d):
        if d < 0:
            yield tuple(result_list)
        else:
            for i in listunique:
                if i.occurrences > 0:
                    result_list[d]=i.value
                    i.occurrences-=1
                    for g in  perm_unique_helper(listunique,result_list,d-1):
                        yield g
                    i.occurrences+=1
    a=[]
    if numM==1:
        for i in range(numKv+1):
                if i<=numKv:
                    a += list(perm_unique([i]))
    if numM==2:
        for i in range(numKv+1):
            for j in range(i+1):
                if i+j<=numKv:
                    a += list(perm_unique([j,i]+[0]*(numM-2)))
    if numM==3:
        for i in range(numKv+1):
            for j in range(i+1):
                for k in range(j+1):
                    if i+j+k<=numKv:
                        a += list(perm_unique([j,i,k]+[0]*(numM-3)))
    if numM==4:
        for i in range(numKv+1):
            for j in range(i+1):
                for k in range(j+1):
                    for k2 in range(k+1):
                        if i+j+k+k2<=numKv:
                            a += list(perm_unique([j,i,k,k2]+[0]*(numM-4)))
    if numM==5:
        for i in range(numKv+1):
            for j in range(i+1):
                for k in range(j+1):
                    for k2 in range(k+1):
                        for k3 in range(k2+1):
                            if i+j+k+k2+k3<=numKv:
                                a += list(perm_unique([j,i,k,k2,k3]+[0]*(numM-5)))
    if numM==6:
        for i in range(numKv+1):
            for j in range(i+1):
                for k in range(j+1):
                    for k2 in range(k+1):
                        for k3 in range(k2+1):
                            for k4 in range(k3+1):
                                if i+j+k+k2+k3+k4<=numKv:
                                    a += list(perm_unique([j,i,k,k2,k3,k4]+[0]*(numM-6)))                                                                        
    return np.array(a)        	


#def lor(x, x0, H):
#    return 1/np.pi*(1/2*H)/((x-x0)**2+(1/2*H)**2)


def bolc(E,E0,T):
    return np.exp(-E/(T*0.695028 ))/np.exp(-E0/(T*0.695028 ))


def fur(E_eg,G_def,inh):
    tim=np.arange(0,2,0.0001)
    rewwww=np.exp(-1j*E_eg*tim-G_def*tim-1/2*inh*tim*tim)
    ft=np.fft.fftshift(np.fft.fft(rewwww))
    lenght=np.shape(rewwww)[0]
    freq = np.fft.fftshift(np.fft.fftfreq(lenght, d=0.0001/(2*np.pi)))
    #plt.plot(freq,np.real(ft),'r-')#,freq[:int(lenght/2)],np.imag(ft)[:int(lenght/2)],'g*-')
    #plt.show()
    #plt.xlim([-13000,-11000])
    return [freq,ft]

def SDF(om,om_c,a,s):
    return s*np.pi*om*(om/om_c)**(a-1)*np.exp(-om/om_c)
def SDFDEB(om,gama,s):
    
    if s==0:
        lamb=0
    else:
        lamb=s*25
    return 2*lamb*om/(om**2+gama**2)*gama
def gaussianls(x, x0, H,s):
    y=np.exp(-((x-x0)**2)/(2*(H**2)))
    norm=np.trapz(y[1:]/(x[1:]*x[1:]),x[1:])/np.pi
    return y/norm*s
def rectang(x,x0,H):
    y=np.where(np.logical_and(x>(x0-H),x<(x0+H)),1,0)
    #norm=np.trapz(y[1:]/(x[1:]*x[1:]),x[1:])/np.pi
    return y#/norm*s

def kurtSDF(om,s,s_0):
    x=np.arange(0,20000,0.1,dtype=np.float32)
    y=SDF(x,50,1,0)+gaussianls(x,om,1,s)+SDFDEB(x,50,s_0)
    L=np.trapz(y[1:]/x[1:],x[1:])/np.pi
    S=np.trapz(y[1:]/(x[1:]*x[1:]),x[1:])/np.pi
    #print(L,S)
    #plt.plot(x,y)
    file=open('SDF.txt','w')
    file.write('%d \n'% len(y))
    file.write('%f \n'% x[0])
    file.write('%.4f \n'% (x[1]-x[0]))
    for i in y:
        file.write('%.14e \n'% i)
    file.close()
    return L
def Redfield(om,s,j,temp,ax=plt):
#     x=np.arange(0,5000,0.1,dtype=np.float32)
#     y=SDF(x,50,1,0)+gaussianls(x,om,1,s)#+SDFDEB(x,50,s)
#     L=np.trapz(y[1:]/x[1:],x[1:])/np.pi
#     S=np.trapz(y[1:]/(x[1:]*x[1:]),x[1:])/np.pi
#     #print(L,S)
#     #plt.plot(x,y)
#     file=open('SDF.txt','w')
#     file.write('%d \n'% len(y))
#     file.write('%f \n'% x[0])
#     file.write('%.4f \n'% (x[1]-x[0]))
#     for i in y:
#         file.write('%.14e \n'% i)
#     file.close()
    L=kurtSDF(om,s,1)
    file=open('inputmod.txt','w')
    file.write('key-setup-complete\n\n')
    file.write('%d \n\n' %2)
    file.write('%.3f \n' %(12000+L))
    file.write('%.3f %.3f \n\n' % (j, (12000+L)))
    file.write('1 0 0\n')
    #file.write('-1 0 0\n')
    file.write('-0.308 0.95 0\n\n')
    #file.write('0 0 0\n\n')
    file.write('%.3f\n\n'% temp)
    file.write('0\n\n')
    file.write('1\n\n')
    file.write('SDF.txt\n\n')
    file.write('0\n\n')
    file.write('1 1\n\n')
    file.write('1 0 0\n')
    file.write('1 0 0\n\n')
    file.write('0\n\n')
    file.write('7000\n')
    file.write('30000\n\n')
    file.write('23000\n\n')

    file.close()
    
    
    #cmd = ['/run/myscript', '--arg', 'value']
    cmd = [r'../uqcfp/bin/tba.calculator_abs_excitons inputmod.txt outr.txt > outasr.txt']
    subprocess.Popen(cmd, shell=True).wait()
    #for line in p.stdout:
    #    print line
    #p.wait()
    
    #!../uqcfp/bin/tba.calculator_abs_excitons inputmod.txt outr.txt > outasr.txt
    #!rm *.wrk
    l = []
    file=open('outr.txt')
    n=0
    for line in file:
        if n<1:
            n+=1
            continue
        else:
            l.append(line.split())
    l=np.array(l).astype(np.float)
    l=l.transpose() 
    ax.plot(l[0],l[1]/max(l[1]),'r')
    return L

def fourje(G,A,K_,gfun2,T,miumod,CorrD,Cy,Cx):
    step=0.00002
    dinh=0
    ddef=0#70/3.14
    zpl=0
    numG=len(G)
    numE=len(A)
    #print(numG,numE)
    x0=Cx#np.arange(0,20000,0.1,dtype=np.float32)
    y0=Cy#SDFDEB(x0,50,1)
    x0[0]=0.0001
    y0[0]=0
    def Cw(te,x0):
        x=x0

#np.tanh(x/(2*T*0.695028))**(-1)

        y=np.zeros(np.size(x0))
        y=y0/(2*np.pi*x**2)*((1+np.exp(-x/(T*0.695028)))/(1-np.exp(-x/(T*0.695028)))*(1-np.cos(x*te))+1j*np.sin(x*te)-1j*x*te)
        y[0]=0
        return y
    tim=np.arange(0,step*2**16,step,dtype=np.float32)
    tim2=np.arange(0,step*2**8,2**4*step,dtype=np.float32)
    tim3=np.arange(step*2**8,step*2**16,2**7*step,dtype=np.float32)
    tim3=np.concatenate((tim2,tim3))
    rew=np.zeros(np.shape(tim)[0],dtype=np.complex64)
    g__3=np.zeros(np.shape(tim3)[0],dtype=np.complex64)
    g__=np.zeros(np.shape(tim)[0],dtype=np.complex64)
    for index, te in np.ndenumerate(tim3):
         g__3[index[0]]=2*np.trapz(Cw(te,x0),x=x0)
    #      if index[0]%10==0:
    #          print(index[0],g__3[index[0]],gfun2(te))
    g__=np.interp(tim,tim3,g__3)         
    # for index, te in np.ndenumerate(tim):
    #      g__[index[0]]=2*np.trapz(Cw(te,x0),x=x0)
    #g__=gfun2(tim)
    for i in range(numE):
        for j in range(numG):
            if bolc(G[j],G[0],T)*miumod[i][j]/miumod.max() <1e-6:
                continue
            rew+=(np.exp((-1j*(A[i]-G[j])-ddef+(K_(i+numG,i+numG)+K_(j,j))/2-1/2*dinh*tim)*tim-g__*(CorrD[j,j]+CorrD[i+numG,i+numG]-CorrD[i+numG,j]-CorrD[j,i+numG])))*miumod[i][j]*bolc(G[j],G[0],T)
            #-np.conjugate(np.exp((-1j*(A[i]-G[j])-ddef-(K_(i+numG,i+numG)+K_(j,j))/2-1/2*dinh*tim)*tim-(gfun2(tim))*(CorrD[j,j]+CorrD[i+numG,i+numG]-CorrD[i+numG,j]-CorrD[j,i+numG])))
    ft=np.fft.fftshift(np.fft.fft(rew)) 
    length=np.shape(rew)[0]
    freq = np.fft.fftshift(np.fft.fftfreq(length, d=step/(2*np.pi)))
    return freq,ft,rew
@jit
def Cw(te,x,T,y0):
    #np.tanh(x/(2*T*0.695028))**(-1)
            #y=np.zeros(np.size(x0))
            y0=y0/(2*np.pi*x**2)*((1+np.exp(-x/(T*0.695028)))/(1-np.exp(-x/(T*0.695028)))*(1-np.cos(x*te))+1j*np.sin(x*te)-1j*x*te)
#            y[0]=0
            return y0

# @jit
def fourje2(G,A,K_,gfun2,T,miumod,CorrD,Cy,Cx,snum):
    step=0.00001
    dinh=0
    ddef=0#70/3.14
    zpl=0
    numG=len(G)
    numE=len(A)
    g__=0
   # print("1")
   # print(datetime.datetime.now())
    #print(numG,numE)
    for zz in range(snum):
        x0=Cx[zz]#np.arange(0,20000,0.1,dtype=np.float32)
        y0=Cy[zz]#SDFDEB(x0,50,1)
        x0[0]=0.0001
        y0[0]=0
        
        tim=np.arange(0,step*2**16,step,dtype=np.float32)
        tim2=np.arange(0,step*2**8,step,dtype=np.float32)
        #tim2=np.arange(0,step*2**8,2**2*step,dtype=np.float32)
        tim3=np.arange(step*2**8,step*2**16,2**4*step,dtype=np.float32)
        #tim3=np.logspace(step*2**8,step*2**16,100)
        tim3=np.concatenate((tim2,tim3))
        rew=np.zeros(np.shape(tim)[0],dtype=np.complex64)
        g__3=np.zeros(np.shape(tim3)[0],dtype=np.complex64)
        g__t=np.zeros(np.shape(tim)[0],dtype=np.complex64)
        for ii in range(np.size(tim3)):
             g__3[ii]=2*np.trapz(Cw(tim3[ii],x0,T,y0),x=x0)
        #      if index[0]%10==0:
        #          print(index[0],g__3[index[0]],gfun2(te))
        g__t=np.interp(tim,tim3,g__3) 
        if zz==0:
            g__=g__t
        else:
            g__=np.stack((g__,g__t))    
   # print("2")
   # print(datetime.datetime.now())
    # for index, te in np.ndenumerate(tim):
    #      g__[index[0]]=2*np.trapz(Cw(te,x0),x=x0)
    #g__=gfun2(tim)
    for i in range(numE):
        for j in range(numG):
            if bolc(G[j],G[0],T)*miumod[i][j]/miumod.max() <1e-6:
                continue
            else:
                rew+=(np.exp((-1j*(A[i]-G[j])-ddef+(K_(i+numG,i+numG)+K_(j,j))/2-1/2*dinh*tim)*tim-np.dot(g__.T,(CorrD[:,j,j]+CorrD[:,i+numG,i+numG]-CorrD[:,i+numG,j]-CorrD[:,j,i+numG]))))*miumod[i][j]*bolc(G[j],G[0],T)
            #-np.conjugate(np.exp((-1j*(A[i]-G[j])-ddef-(K_(i+numG,i+numG)+K_(j,j))/2-1/2*dinh*tim)*tim-(gfun2(tim))*(CorrD[j,j]+CorrD[i+numG,i+numG]-CorrD[i+numG,j]-CorrD[j,i+numG])))
   # print("3")
   # print(datetime.datetime.now())
    #rew=np.concatenate((rew,np.zeros(np.size(tim)*3)))
    ft=np.fft.fftshift(np.fft.fft(rew)) 
   # print("5")
   # print(datetime.datetime.now())
    length=np.shape(rew)[0]
    freq = np.fft.fftshift(np.fft.fftfreq(length, d=step/(2*np.pi)))
    return freq,ft,rew

