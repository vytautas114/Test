import numpy as np
import datetime
from funkcijosBRC import * 
from koreleBRC import Corrcoff_2
from scipy.special import factorial
import os
#from numba import jit

os.environ['MKL_NUM_THREADS'] = '1'

#@jit
def spektras(ax,s0,om0,T,Kvsk=2,nam='BRC/1td_test'):

    print(datetime.datetime.now())
    # A=np.loadtxt('SDF.txt')
    # Cy=A[3:] #A[0]-N A[1]-X0 A[2]-dx
    # Cx=np.arange(A[1],A[1]+int(A[0])*A[2],A[2])
    wc=40
    sig=0.5
    Cx=np.linspace(0.0001,2000,20000)
    
    
    
    #S=np.trapz(y[1:]/(x[1:]*x[1:]),x[1:])/np.pi
    def GL(w,wm,gl,A):
        sg=gl/(2*np.sqrt(2*np.log(2)))
        if w<=wm:
            return A*np.exp(-(w-wm)**2/(2*sg**2))
        elif w>1800:
            return A*(gl/2)**2/((w-wm)**2+(gl/2)**2)*np.exp(-w+1800)
        else:
            return A*(gl/2)**2/((w-wm)**2+(gl/2)**2)
    mul0=np.array([1.95,1.95,1.10,0.70,1.2,1.6])
    mul02=np.array([1.95,1.95,0,0,0,0])
    GL = np.vectorize(GL)
    Cy=1.7*Cx*np.pi/(sig*np.sqrt(2*np.pi))*np.exp(-(np.log(Cx/wc))**2/(2*sig**2))
    Cy2=GL(Cx,125,30,1650)
    Cx2=Cx 
    L1=np.trapz(Cy[1:]/Cx[1:],Cx[1:])/np.pi*mul0*0
    L2=np.trapz(Cy2[1:]/Cx2[1:],Cx2[1:])/np.pi*mul02*0

    def condonFactors(levels,s):
        Cfact=np.zeros((levels,levels))
        def qwave(x,n):

            tem=np.zeros(n+1)
            tem[n]=1
            return np.polynomial.hermite.hermval(x,tem)\
            *np.exp(-x**2/2)*np.power((1/np.pi),1/4)*1/np.sqrt(2**n*factorial(n,exact=True))

        x=np.linspace(-10,10,3000)
    #     s=0.5
    #     om=100
        for i in range(levels):
            for j in range(levels):
                y=qwave(x,i)
                y1=qwave(x+np.sqrt(2*s),j)
                Cfact[i,j]=np.trapz(y*y1,x)
                
        return Cfact 
    
   
    Factors=condonFactors(Kvsk+1,s0)#np.zeros((virpnum+2,virpnum+2))

    saitnum=6
    # en0=np.array([12630,13340,12540,13550,11990,12290])
    en0=np.array([11990,12290,12540,12630,13340,13550])
    om=np.array([om0,om0,om0,om0,om0,om0])
    J=[[0,650,-20,-119,27,-11],
       [0,0,-119,-17,-9,24],
       [0,0,0,18,-7,104],
       [0,0,0,0,104,-7],
       [0,0,0,0,0,3],
       [0,0,0,0,0,0]]
    J=np.array(J)   
    J=J.T+J   
    s=np.array([s0,s0,s0,s0,s0,s0])
    #mul0=[0.7,1.2,1.1,1.6,1.95,1.95]
    #mul0=[1.95,1.95,1.10,0.70,1.2,1.6]
  #  lemd=L*np.ones(saitnum)#om*s+
    lemd=L1+L2
    D=np.array([[-0.6611,-0.4041,-0.1110],
                [0.7385,-0.0524,-0.1608],
                [0.9733,-0.1161,-0.1824],
                [-0.7794,-0.5123,-0.3631],
                [-0.0096,0.6633,0.1238],
                [-0.2151,0.2293,0.5958]])
   
    
    nn=np.arange(0,saitnum)
    
    virp=deriniairev(saitnum,Kvsk) 
    v2=np.shape(virp)[0]    

    en=np.zeros(saitnum)
    for i in range(saitnum):
        en[i]=en0[i]#np.random.normal(en0[i], 50, 1)
        
    G=np.zeros(v2)
    for i in range(v2):
        for m in range(saitnum): 
            G[i]=G[i]+om[m]*(1/2+virp[i,m])
    np.savetxt(nam+"_energG.txt", G,fmt='%.2f')        
    H=np.zeros([saitnum*v2,saitnum*v2])
    for i in range(saitnum*v2):
        for j in range(saitnum*v2):        
            if i==j:
                H[i,j]=en[i//v2]+lemd[i//v2]
                for m in range(saitnum):
                    H[i,j]+=om[m]*(1/2+virp[i%v2,m])
            elif i//v2!=j//v2:
                H[i,j]=J[i//v2,j//v2]*Factors[virp[i%v2,i//v2],virp[j%v2,i//v2]]*Factors[virp[j%v2,j//v2],virp[i%v2,j//v2]]*(1 if np.all(virp[j%v2,nn[np.logical_and(nn!=j//v2,nn!=i//v2)]]==virp[i%v2,nn[np.logical_and(nn!=j//v2,nn!=i//v2)]]) else 0 )
    np.savetxt('Hamilton.out',H,fmt='%1.2f',delimiter='      ')
    if os.path.isfile(nam+"_tikr.txt"):
        os.remove(nam+"_tikr.txt")
    if os.path.isfile(nam+"_energ.txt"):
        os.remove(nam+"_energ.txt")
    if os.path.isfile(nam+"_dip.txt"):
        os.remove(nam+"_dip.txt")    
    f_tikr=open(nam+"_tikr.txt",'a')
    f_enr=open(nam+"_energ.txt",'a')
    f_dip=open(nam+"_dip.txt",'a')

    for itera in range(1):
        print(itera)
        en=np.random.normal(en0, [0]*saitnum)#[87,87,30,30,45,60])#
        for i in range(saitnum*v2):
            H[i,i]=lemd[i//v2]+en[i//v2]
            for m in range(saitnum):
                H[i,i]+=om[m]*(1/2+virp[i%v2,m])
                  
        A, B =  np.linalg.eigh(H)
        ev_list = zip( A, B.T )
        ev_list=sorted(ev_list,key=lambda tup:tup[0], reverse=False)
        A, B = zip(*ev_list)
        A=np.array(A)
        B=np.array(B).T
        
        B.tofile(f_tikr,format='%2.9e')
        A.tofile(f_enr,format='%1.6f')


        print("dip start",datetime.datetime.now())
        miumod=np.zeros([saitnum*v2,v2])
        miu=np.zeros([saitnum*v2,v2,3])
        FC2=np.zeros((v2,v2,saitnum))
        KFF=B.reshape(saitnum,v2,v2*saitnum)
        for j in range(v2):
            for i in range(v2):
                for m in range(saitnum):
                    FC2[i,j,m]=(1 if np.all(virp[j,nn[nn!=m]]==virp[i,nn[nn!=m]]) else 0 )
        FCC=np.zeros((v2,v2,saitnum))
        for i in range(v2):
                for j in range(v2):
                    for m in range(saitnum):
                        FCC[i,j,m]=Factors[virp[j,m],virp[i,m]]
        miu=np.einsum("md,mjp,jim,ijm->pid",D,KFF,FCC,FC2)        
        miumod=np.einsum("ikj,ikj->ik",miu,miu)
        # kff=B.reshape(saitnum,v2,v2*saitnum)


        print("dip end",datetime.datetime.now())
        miumod.tofile(f_dip,format='%2.4e')
        energG=np.array(G[:])
        energE=np.array(A[:])
        numE=np.size(energE)
        numG=np.size(energG)
        numF=0
        Bf=[0]
        mul_2=np.stack((mul0,mul02))
        #S0=[0.05,0.05,0.05,0.05,0.05,0.05]
        #CorrOffd,CorrD=Corrcoff(numG,numE,numF,B,Bf,virp,v2,Kvsk+1,saitnum,S=s,mul=mul0,OM=om)
        CorrOffd,CorrD=Corrcoff_2(numG,numE,numF,B,Bf,virp,v2,Kvsk+1,saitnum,S=s,mul=mul_2,OM=om,snum=2)
        print("koeff end",datetime.datetime.now())
        #np.savetxt("corrd.txt",CorrD,fmt='%1.4f\t')
        #np.savetxt("corrcoff.txt",CorrOffd,fmt='%1.5f\t')
        
       


     
        
        def K_2(a,b):
            tarp1=0
            tarp2=0
            if a==b:
                return 0
                kx=0
                for i in range(numG+numE):
                    if(a!=i):
                        kx-=K_2(i,a)
                return kx     
            if a>=numG and b>=numG and (A[a-numG]-A[b-numG]!=0):
                tem=A[b-numG]-A[a-numG]
                if tem<0:
                    tarp1=-np.interp(-tem,Cx,Cy)
                    tarp2=-np.interp(-tem,Cx2,Cy2)
                else:
                    tarp1=np.interp(tem,Cx,Cy)  
                    tarp2=np.interp(tem,Cx2,Cy2)  
         
                return np.abs((tarp1*CorrOffd[0,b,a]+tarp2*CorrOffd[1,b,a])*(np.tanh((tem)/(2*T*0.695028))**(-1)+1)) #if (A[a-numG]-A[b-numG])>om[0]-1 and (A[a-numG]-A[b-numG])<om[0]+1 else 0
            elif a<numG and b<numG:
                tem=G[b]-G[a]
                if tem<0:
                    tarp1=-np.interp(-tem,Cx,Cy)
                    tarp2=-np.interp(-tem,Cx2,Cy2)
                else:
                    tarp1=np.interp(tem,Cx,Cy) 
                    tarp2=np.interp(tem,Cx2,Cy2)  
           
                return np.abs((tarp1*CorrOffd[0,b,a]+tarp2*CorrOffd[1,b,a])*(np.tanh((tem)/(2*T*0.695028))**(-1)+1)) if tem!=0 else 0 #if (G[a]-G[b])>om[0]-1 #and (G[a]-G[b])<om[0]+1 else 0
            else:
                return 0

        

        Spartos=np.zeros((numG+numE,numG+numE))
        for i in range(numG+numE):
            for j in range(numG+numE):
                if i!=j:
                    Spartos[i,j]=K_2(i, j)
        for i in range(numG+numE):
            Spartos[i,i]-=np.sum(Spartos[:,i])      
        print("spart end",datetime.datetime.now())        
        if itera==0:
            Spartos_vid=Spartos
        else:
            Spartos_vid+=Spartos    
      
        cxt=np.stack((Cx,Cx2))
        cyt=np.stack((Cy,Cy2))
        freq,ft,resp=fourje2(G,A,Spartos,T,miumod,CorrD,cyt,cxt,2)
        print("fft end",datetime.datetime.now())
        
        
        if itera==0:
            ftsum=ft
        else:
            ftsum=ft+ftsum


    np.savetxt(nam+'_spart.txt', Spartos_vid/(itera+1),fmt='%1.9e')            
    ft=ftsum
    apatinis=np.argwhere(10000<freq)[0][0]
    virsutinis=np.argwhere(freq<16000)[-1][0]
    sugertis=((np.real(ft)/max(np.real(ft)))[::-1]) 
    freq=freq[apatinis:virsutinis]
    sugertis=sugertis[apatinis:virsutinis]
    sugertis=sugertis/np.max(sugertis)
    rezz=np.stack((freq,sugertis),axis=-1)
    ax.plot(freq,sugertis,'k',lw=1)    
    np.savetxt(nam+'_rezul.txt', rezz)
    ax.set_xlim([10000,15990])
    ax.set_ylim([0,1.1])
    ax.text(12900,0.6,'$\omega_v$ = %.f $cm^{-1}$\nS = %.5f' % (om[0],s[0]),style='italic',fontsize=13)
    print(datetime.datetime.now())
   