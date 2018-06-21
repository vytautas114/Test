import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import datetime
from funkcijosBRC import * 
from koreleBRC import Corrcoff_2
from scipy.special import factorial
import os
from numba import jit

def spektras(ax,s0,om0,j0,T,Kvsk=2,nam='BRC/1td_test'):
    
   

    print(datetime.datetime.now())
    #kurtSDF(100,0,1)

    # A=np.loadtxt('SDF.txt')
    # Cy=A[3:] #A[0]-N A[1]-X0 A[2]-dx
    # Cx=np.arange(A[1],A[1]+int(A[0])*A[2],A[2])
    wc=40
    sig=0.5
    Cx=np.linspace(0.01,2000,20000)
    
    
    
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
    # J=np.zeros([6,6])
    # J[0,1]=j0#200
    # J[1,0]=J[0,1]
    # J[2,3]=104
    # J[3,2]=J[2,3]
    # J[4,5]=650
    # J[5,4]=J[4,5]
    # J[4,0]=-119
    # J[0,4]=J[4,0]
    # J[2,5]=-119
    # J[5,2]=J[2,5]
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
    #D=np.array([[1,0,0],[-0.308,0.95,0]])
    # D=np.array([[-0.7794,-0.5123,-0.3631],[-0.0096,0.6633,0.1238],[0.9733,-0.1161,-0.1824],[-0.2151,0.2293,0.5958],[-0.6611,-0.4041,-0.1110],[0.7385,-0.0524,-0.1608]])
    D=np.array([[-0.6611,-0.4041,-0.1110],
                [0.7385,-0.0524,-0.1608],
                [0.9733,-0.1161,-0.1824],
                [-0.7794,-0.5123,-0.3631],
                [-0.0096,0.6633,0.1238],
                [-0.2151,0.2293,0.5958]])
    x=np.linspace(9000,18000,20000,dtype=np.float32)
    y=np.zeros(len(x),dtype=np.float32)

    #virpnum=4
    
    nn=np.arange(0,saitnum)
    #nam='1td_'+str(s0)+'_'+str(om0)+'_'+str(j0)+'_'+str(T)+'K_'+str(virpnum)
    
    #virp=visideriniai(saitnum,virpnum)
    #virp=deriniai2(saitnum,virpnum)
    virp=deriniairev(saitnum,Kvsk) 
    v2=np.shape(virp)[0]    
    #print(virp)
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
            if i==j: #and np.array_equal(virp[i%v2],virp[j%v2]):
                H[i,j]=en[i//v2]+lemd[i//v2]#+om[0]*(1/2+virp[i%v2,0])+om[1]*(1/2+virp[i%v2,1])
                for m in range(saitnum):
                    H[i,j]+=om[m]*(1/2+virp[i%v2,m])
            elif i//v2!=j//v2:
                H[i,j]=J[i//v2,j//v2]*Factors[virp[i%v2,i//v2],virp[j%v2,i//v2]]*Factors[virp[j%v2,j//v2],virp[i%v2,j//v2]]*(1 if np.all(virp[j%v2,nn[np.logical_and(nn!=j//v2,nn!=i//v2)]]==virp[i%v2,nn[np.logical_and(nn!=j//v2,nn!=i//v2)]]) else 0 )
    #print(H)
    np.savetxt('test.out',H,fmt='%1.2f',delimiter='      ')
    # Fbus=rink(saitnum)
    
    # H_f=np.zeros([saitnum*(saitnum-1)*v2//2,saitnum*(saitnum-1)*v2//2])
    # for i in range(saitnum*(saitnum-1)*v2//2):
    #     for j in  range(saitnum*(saitnum-1)*v2//2):
    #         indk=Fbus[(i//v2)][0]
    #         indk_=Fbus[(j//v2)][0]
    #         indl=Fbus[(i//v2)][1]
    #         indl_=Fbus[(j//v2)][1]
            
    #         if i==j:
    #             H_f[i,j]=en[indk]+en[indl]+lemd[indk]+lemd[indl]
    #             for m in range(saitnum):
    #                 H_f[i,j]=H_f[i,j]+om[m]*(1/2+virp[i%v2,m])
    #         if i%v2==j%v2:
    #             if indk==indk_ and indl!=indl_:
    #                 H_f[i,j]+=J[indl,indl_]
    #             if indk!=indk_ and indl==indl_:
    #                 H_f[i,j]+=J[indk,indk_]    
    #         if indk==indk_ and indl==indl_:
    #             if np.array_equal(virp[i%v2,np.arange(len(virp[i%v2]))!=indk],virp[j%v2,np.arange(len(virp[j%v2]))!=indk]):
    #                 if virp[i%v2,indk]==virp[j%v2,indk]+1 and (virp[j%v2,indk]+1)<virpnum:
    #                     H_f[i,j]=-om[indk]*np.sqrt(s[indk]*virp[i%v2,indk]) 
    #                 if virp[i%v2,indk]==virp[j%v2,indk]-1 and (virp[j%v2,indk]-1)>=0:
    #                     H_f[i,j]=-om[indk]*np.sqrt(s[indk]*virp[j%v2,indk])    
    #             if np.array_equal(virp[i%v2,np.arange(len(virp[i%v2]))!=indl],virp[j%v2,np.arange(len(virp[j%v2]))!=indl]):
    #                 if virp[i%v2,indl]==virp[j%v2,indl]+1 and (virp[j%v2,indl]+1)<virpnum:
    #                     H_f[i,j]=om[indl]*np.sqrt(s[indl]*virp[i%v2,indl]) 
    #                 if virp[i%v2,indl]==virp[j%v2,indl]-1 and (virp[j%v2,indl]-1)>=0:
    #                     H_f[i,j]=om[indl]*np.sqrt(s[indl]*virp[j%v2,indl])                                 
                                                
        
    #print(H_f)
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
            H[i,i]=lemd[i//v2]+en[i//v2]#+np.sum(om*(1/2+virp[i%v2]))#+om[0]*(1/2+virp[i%v2,0])+om[1]*(1/2+virp[i%v2,1])
            for m in range(saitnum):
                H[i,i]+=om[m]*(1/2+virp[i%v2,m])
                #for m in range(saitnum):
                 #   H[i,i]=H[i,i]+om[m]*(1/2+virp[i%v2,m])
        
        # for i in range(saitnum*(saitnum-1)*v2//2):
        #     H_f[i,i]=en[indk]+en[indl]+lemd[indk]+lemd[indl]
        #     for m in range(saitnum):
        #         H_f[i,i]=H_f[i,i]+om[m]*(1/2+virp[i%v2,m])

       # H_f=np.zeros(())
        #A,B=np.linalg.eig(H)
        # if saitnum!=1:
        #     Af,Bf=np.linalg.eig(H_f)

        #     ev_list = zip( Af, Bf.T )
        #     ev_list=sorted(ev_list,key=lambda tup:tup[0], reverse=False)
        #     Af, Bf = zip(*ev_list)
        #     Af=np.array(Af)
        #     Bf=np.array(Bf).T
        # else:
        #     Af=np.array([])
        #     Bf=0
    
        A, B =  np.linalg.eigh(H)
       # print(A,'\n',B)
       # print(B[1,0])
        ev_list = zip( A, B.T )
        ev_list=sorted(ev_list,key=lambda tup:tup[0], reverse=False)
        A, B = zip(*ev_list)
        A=np.array(A)
        B=np.array(B).T
        
       # np.savetxt(f_tikr, B,fmt='%1.2e')
        
       # np.savetxt(f_enr, A,fmt='%.2f')
        B.tofile(f_tikr,format='%2.9e')
        A.tofile(f_enr,format='%1.6f')

        # print(A,'\n',B)
        # print(B[1,1])
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
                          FCC[j,i,m]=Factors[virp[j,m],virp[i,m]]          
        # for p in range(saitnum*v2):
        #     for i in range(v2):
        #         for j in range(v2):
        #             for m in range(saitnum):
                        
        #                 #nn=np.all(nn[nn!=m]==nn[nn!=m])
        #                 #miu[p,i]=miu[p,i]+D[m]*B[m*v2+j%v2,p]*Factors[virp[j,m],virp[i,m]]*FC2[i,j,m]#Factors[virp[j,0],virp[i,0]]#(1 if j==i else 0 )
        #                 miu[p,i]=miu[p,i]+D[m]*KFF[m,j,p]*FCC[j,i,m]*FC2[i,j,m]
        #                 #print(Factors[virp[j,m],virp[i,m]])
        #         #miumod[p,i]=np.dot(miu[p,i],miu[p,i])
        miu=np.einsum("md,mjp,jim,ijm->pid",D,KFF,FCC,FC2)        
        miumod=np.einsum("ikj,ikj->ik",miu,miu)
        # kff=B.reshape(saitnum,v2,v2*saitnum)
        # miu[p,i]=miu[p,i]+D[m]*kff[m,j,p]*Factors[virp[j,m],virp[i,m]]*FC2[i,j,m]
        # F3=np.zeros((v2,v2,saitnum))

        # miu=np.einsum("mz,mjp,jmim,i,j,m->piz",D,kff,Fa)

        print("dip end",datetime.datetime.now())
        # if saitnum!=1:
        #     miu_f=np.zeros([saitnum*v2,saitnum*(saitnum-1)*v2//2,3])
        #     miumod_f=np.zeros([saitnum*v2,saitnum*(saitnum-1)*v2//2])
        #     for p in range(saitnum*v2):
        #         for r in range(saitnum*(saitnum-1)*v2//2):
        #             for jj in range(len(Fbus)):

        #                 for i in range(v2):
        #                     miu_f[p,r]+=D[Fbus[jj][0]]*B[Fbus[jj][1]*v2+i%v2,p]*Bf[v2*jj+i%v2,r]+D[Fbus[jj][1]]*B[Fbus[jj][0]*v2+i%v2,p]*Bf[v2*jj+i%v2,r]
        #             miumod_f[p][r]=np.dot(miu_f[p,r],miu_f[p,r])
        miumod.tofile(f_dip,format='%2.4e')
        #np.savetxt(nam+"_dip2.txt",miumod,fmt='%1.4f\t')
        energG=np.array(G[:])
        energE=np.array(A[:])
        #energF=np.array(Af[:])
        numE=np.size(energE)
        numG=np.size(energG)
        #numF=np.size(energF)
        numF=0
        Bf=[0]
        dipoll=[]
        

        mul_2=np.stack((mul0,mul02))
        #S0=[0.05,0.05,0.05,0.05,0.05,0.05]
        #CorrOffd,CorrD=Corrcoff(numG,numE,numF,B,Bf,virp,v2,Kvsk+1,saitnum,S=s,mul=mul0,OM=om)
        CorrOffd,CorrD=Corrcoff_2(numG,numE,numF,B,Bf,virp,v2,Kvsk+1,saitnum,S=s,mul=mul_2,OM=om,snum=2)
        print("koeff end",datetime.datetime.now())
        np.savetxt("corrd.txt",CorrD[1],fmt='%1.4f\t')
        np.savetxt("corrcoff.txt",CorrOffd[1],fmt='%1.5f\t')
        
        # for i in range(numG+numE+numF):
        #     for j in range(numG+numE+numF):
        #         CorrOffd[i,j]=CorrOffd[j,i]
        #         CorrD[i,j]=CorrD[j,i]


        def gfun2(t):
            #return 0
            summ=0
            ii=0
            #summ+=2*s[ii]*((np.tanh(om[ii]/(2*T*0.69))**(-1))*(1-np.cos(om[ii]*t))+1j*(np.sin(om[ii]*t)-om[ii]*t))
            beta=(1/(T*0.695028))
            lam=25
            gam=50
            summ+=lam/(gam)*(2/(beta*gam)-1j)*(np.exp(-gam*t)+gam*t-1)
            return summ
    
        #www=np.arange(0,20000,0.1,dtype=np.float32)
        #CCC=SDFDEB(www,50,1)
        def K_(a,b):
            tarp=0
            if a==b:
                kx=0
                for i in range(numG+numE):
                    if(a!=i):
                        kx-=K_(i,a)
                return kx     
            if a>=numG and b>=numG and (A[a-numG]-A[b-numG]!=0):
                tem=A[a-numG]-A[b-numG]
                if tem<0:
                    tarp=-np.interp(-tem,Cx,Cy)
                else:
                    tarp=np.interp(tem,Cx,Cy)   
                #tarp=SDFDEB(tem,50,1)     
                return np.abs(tarp*CorrOffd[a,b]*(np.tanh((tem)/(2*T*0.695028))**(-1)-1)) #if (A[a-numG]-A[b-numG])>om[0]-1 and (A[a-numG]-A[b-numG])<om[0]+1 else 0
            elif a<numG and b<numG:
                tem=G[a]-G[b]
                if tem<0:
                    tarp=-np.interp(-tem,Cx,Cy)
                else:
                    tarp=np.interp(tem,Cx,Cy)   
                #tarp=SDFDEB(tem,50,1)
                return np.abs(tarp*CorrOffd[a,b]*(np.tanh((tem)/(2*T*0.695028))**(-1)-1)) if tem!=0 else 0 #if (G[a]-G[b])>om[0]-1 #and (G[a]-G[b])<om[0]+1 else 0
            else:
                return 0
        
        def K_2(a,b):
            tarp1=0
            tarp2=0
            temp=0
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
                #tarp=SDFDEB(tem,50,1) 

                return np.abs((tarp1*CorrOffd[0,b,a]+tarp2*CorrOffd[1,b,a])*(np.tanh((tem)/(2*T*0.695028))**(-1)+1)) #if (A[a-numG]-A[b-numG])>om[0]-1 and (A[a-numG]-A[b-numG])<om[0]+1 else 0
            elif a<numG and b<numG:
                tem=G[b]-G[a]
                if tem<0:
                    tarp1=-np.interp(-tem,Cx,Cy)
                    tarp2=-np.interp(-tem,Cx2,Cy2)
                else:
                    tarp1=np.interp(tem,Cx,Cy) 
                    tarp2=np.interp(tem,Cx2,Cy2)  
                #tarp=SDFDEB(tem,50,1)

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
    #    np.savetxt(nam+'_spart.txt', Spartos,fmt='%1.5e')
    #    np.savetxt(nam+'_diag.txt', CorrD)
    #    np.savetxt(nam+'_offdiag.txt', CorrOffd)
        def K0(a,b):
            return Spartos[a,b]
        #freq,ft,resp=fourje(G,A,K0,gfun2,T,miumod,CorrD,Cy,Cx)
        cxt=np.stack((Cx,Cx2))
        cyt=np.stack((Cy,Cy2))
        freq,ft,resp=fourje2(G,A,Spartos,gfun2,T,miumod,CorrD,cyt,cxt,2)
        print("fft end",datetime.datetime.now())
        # ax.plot(-freq,np.real(ft)/max(np.real(ft)),'k',lw=1)#,-freq,np.imag(ft)/max(np.imag(ft)))#lenght)        
        #freq=-freq[::-1]
        
        if itera==0:
            ftsum=ft
        else:
            ftsum=ft+ftsum


    #    ax.plot(freq,sugertis,'k',lw=1)
    #    np.savetxt(nam+'_rezul.txt', rezz)

        # for i in range(saitnum*v2):
        #     for j in range(v2):
        #         #fre,ft=fur(A[i]-G[j],20,1)
        #         if bolc(G[j],G[0],T)*miumod[i][j]/miumod.max() < 1e-6:
        #             continue
        #         N=rectang(x,A[i]-G[j],1)#lor(x,(A[i]-G[j]),1)*x
        #         y+=N*miumod[i][j]*bolc(G[j],G[0],T)   
        #         #print(y)
        #         dipoll.append([j,i+numG,miu[i][j][0], miu[i][j][1], miu[i][j][2]])
                
        dipoll1D=np.array(dipoll)    
        # for p in range(saitnum*v2):
        #     for r in range(saitnum*(saitnum-1)*v2//2):
        #         if miumod_f[p][r] > 1e-6:
        #             dipoll.append([p+numG,r+numG+numE,miu_f[p][r][0], miu_f[p][r][1], miu_f[p][r][2]])
    np.savetxt(nam+'_spart.txt', Spartos_vid/(itera+1),fmt='%1.9e')            
    ft=ftsum
    apatinis=np.argwhere(10000<freq)[0][0]
    virsutinis=np.argwhere(freq<16000)[-1][0]
    #print(apatinis,virsutinis)
    sugertis=((np.real(ft)/max(np.real(ft)))[::-1]) 
    freq=freq[apatinis:virsutinis]
    sugertis=sugertis[apatinis:virsutinis]
    sugertis=sugertis/np.max(sugertis)
    ax.plot(freq,sugertis,'k',lw=1)    
    rezz=np.stack((freq,sugertis),axis=-1)
    np.savetxt(nam+'_rezul.txt', rezz)
    # dipoll2D=np.array(dipoll)
    # CorrD1d=np.zeros((numG+numE,numG+numE))
    # CorrOffd1d=np.zeros((numG+numE,numG+numE))
    # CorrD1d=CorrD[:numG+numE,:numG+numE]
    # CorrOffd1d=CorrOffd[:numG+numE,:numG+numE]
    


    #makeinput(energG,energE,dipoll1D,T*0.695028,CorrD1d,CorrOffd1d)
    #makeinput(energG,energE,dipoll1D,T*0.695028,CorrOffd1d,CorrD1d)

    
        
    #makeinput2d(energG,energE,energF,dipoll2D,T*0.695028,CorrD,CorrOffd,11111100000000)
#     np.savetxt("energijosG.txt",energG,fmt='%1.5f')
#     np.savetxt("energijosE.txt",energE,fmt='%1.5f')
#     np.savetxt("dipolltest.txt",dipoll,fmt='%1.5f')
    #np.savetxt('test.out',H,fmt='%1.2f',delimiter='      ')
    #plt.plot(-fre,y)
    # if virpnum==1:
    #     s[0]=0
 #   Redfield(om[0],s[0],j0,T*0.695028,ax)
    # kurtSDF(om[0],0,1)
    # #!../uqcfp/bin/tba.calculator_abs_levels input_level outl.txt > outasl.txt
    # subprocess.Popen(["../uqcfp/bin/tba.calculator_abs_levels input_level outl.txt >outasl.txt"],shell=True).wait()
    # #!rm *.wrk
    # l = []
    # file=open('outl.txt')
    # n=0
    # for line in file:
    #     if n<1:
    #         n+=1
    #         continue
    #     else:
    #         l.append(line.split())
    # file.close()        
    # l=np.array(l).astype(np.float)
    # l=l.transpose() 
    # ax.plot(l[0],l[1]/max(l[1]),'g')
    #print(x,y)
    #ax.plot(x,y/max(y),'b')
    ax.set_xlim([10000,15990])
    ax.set_ylim([0,1.1])
    ax.text(12900,0.6,'E = %.f $cm^{-1}$ \n$\omega_v$ = %.f $cm^{-1}$\nS = %.2f  \nJ = %.f $cm^{-1}$' % (en0[0],om[0],s[0],J[0,1]),style='italic',fontsize=13)
    print(datetime.datetime.now())
    #ax.set_xlabel('$\omega, cm^{-1}$',fontsize=18)
    

    #plt.plot(tim,np.real(rew),'b',tim,np.imag(rew),'r',tim,np.real(gfun2(tim)),'k',tim,np.imag(gfun2(tim)),'g')