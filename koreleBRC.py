import numpy as np
from numba import jit
#import datetime
num_of_proces=4
ke=1#1#1
kvv=1#1#10#1#
kv=np.sqrt(kvv)#0.000000001#1#1


# mul=[0.7,1.2,1.1,1.6,1.95,1.95]
# S=[0.05,0.05,0.05,0.05,0.05,0.05]
#@jit
def koff(B,p,i, m, v2):
    return B[m*v2+i%v2,p]

#@jit#(nopython=True)#(nopython=True)
def ksi__(pa, pb, i, i_s, sg, saitnum, virpnum, virp, kff, hvirp):
    if virp[i,i_s]==0 and sg<0:
         return 0
    elif virp[i,i_s]==virpnum-1 and sg>0:
         return 0
    else:
        i_=np.zeros(saitnum,dtype=np.int)
        i_[i_s]+=sg
        hh=(virp[i]+i_).tobytes()
        if hh in hvirp:
            return np.conjugate(kff[pa,i,i_s])*kff[pb,hvirp[hh],i_s]
        else:
           return 0     

def ksi_opt(pa, pb, i, i_s, sg, saitnum, virpnum, virp, kff, hvirp):
    if virp[i,i_s]==0 and sg<0:
         return [0,0,0]  
    elif virp[i,i_s]==virpnum-1 and sg>0:
         return [0,0,0]  
    else:
        i_=np.zeros(saitnum,dtype=np.int)
        i_[i_s]+=sg
        hh=(virp[i]+i_).tobytes()
        if hh in hvirp:
            return np.conjugate(kff[pa,i,i_s])*kff[pb,hvirp[hh],i_s] , i , hvirp[hh]
        else:
           return [0,0,0]
@jit(cache=True)
def ksi_opt_lop(numE,numG,v2,kvv, saitnum, virpnum, virp, kff, hvirp):
    Dll1=np.zeros([numE,numE,v2,saitnum])
    Dll_1=np.zeros([numE,numE,v2,saitnum])
    if kv!=0 or kvv!=0:
        for pb in range(numE):
            for pa in range(numE):
                for i in range(v2):
                    for i_s in range(saitnum):
                        i_=np.zeros(saitnum,dtype=np.int)
                        i_[i_s]+=1
                        hh=(virp[i]+i_).tobytes()
                        if hh in hvirp:
                            Dll1[pa,pb,i,i_s]=np.conjugate(kff[pa,i,i_s])*kff[pb,hvirp[hh],i_s]
                            Dll_1[pb,pa,hvirp[hh],i_s]=np.conjugate(Dll1[pa,pb,i,i_s])

    return  Dll1, Dll_1






@jit(nopython=True,cache=True)
def ksi2(pa,pb,pc,pd, saitnum,mul,kff):
    summ=0
    for l in range(saitnum):
        summ+=np.dot(np.conjugate(kff[pa,:,l]),kff[pb,:,l])*np.dot(np.conjugate(kff[pc,:,l]),kff[pd,:,l])*mul[l]
    #summ = np.einsum("il,il,jl,jl,l",np.conjugate(kff[pa]),kff[pb],np.conjugate(kff[pc]),kff[pd],mul)
    return summ 



@jit(nopython=True,cache=True)
def h_e(p1, p2,p3,p4, saitnum, virp,S,mul,dll1,dll_1,kff,om0,FR=True):
    sum2=0
    if kvv!=0:
        for ss in range(saitnum):
            sum2+=kvv*mul[ss]*(np.dot(np.sqrt(virp[:,ss]),dll_1[p1,p2,:,ss])*np.dot(np.sqrt(virp[:,ss]),dll_1[p3,p4,:,ss])
            +np.dot(np.sqrt((virp[:,ss]+1)),dll1[p1,p2,:,ss])*np.dot(np.sqrt(virp[:,ss]),dll_1[p3,p4,:,ss])
            +np.dot(np.sqrt(virp[:,ss]),dll_1[p1,p2,:,ss])*np.dot(np.sqrt((virp[:,ss]+1)),dll1[p3,p4,:,ss])
            +np.dot(np.sqrt((virp[:,ss]+1)),dll1[p1,p2,:,ss])*np.dot(np.sqrt((virp[:,ss]+1)),dll1[p3,p4,:,ss]))/(2*om0[ss])

            if FR: 
                sum2+=np.dot(np.conjugate(kff[p1,:,ss]),kff[p2,:,ss])*kvv*2*mul[ss]*np.sqrt(S[ss])*(np.dot(np.sqrt(virp[:,ss]),dll_1[p3,p4,:,ss])+np.dot(np.sqrt(virp[:,ss]+1),dll1[p3,p4,:,ss]))/(2*om0[ss])
            if FR:
                sum2+=np.dot(np.conjugate(kff[p3,:,ss]),kff[p4,:,ss])*kvv*2*mul[ss]*np.sqrt(S[ss])*(np.dot(np.sqrt(virp[:,ss]),dll_1[p1,p2,:,ss])+np.dot(np.sqrt(virp[:,ss]+1),dll1[p1,p2,:,ss]))/(2*om0[ss])  
            if  FR:
                sum2+=4*kvv*mul[ss]*S[ss]*np.dot(np.conjugate(kff[p1,:,ss]),kff[p2,:,ss])*np.dot(np.conjugate(kff[p3,:,ss]),kff[p4,:,ss])/(2*om0[ss])  

    if kv!=0:
        for ss in range(saitnum):
            sum2+=mul[ss]*kv*np.dot(np.conjugate(kff[p3,:,ss]),kff[p4,:,ss])*(np.dot(np.sqrt(virp[:,ss]),dll_1[p1,p2,:,ss])+np.dot(np.sqrt(virp[:,ss]+1),dll1[p1,p2,:,ss]))/np.sqrt(2*om0[ss])   
            sum2+=mul[ss]*kv*np.dot(np.conjugate(kff[p1,:,ss]),kff[p2,:,ss])*(np.dot(np.sqrt(virp[:,ss]),dll_1[p3,p4,:,ss])+np.dot(np.sqrt(virp[:,ss]+1),dll1[p3,p4,:,ss]))/np.sqrt(2*om0[ss])
               
        if FR:
            for ss in range(saitnum):
                sum2+=np.dot(np.conjugate(kff[p1,:,ss]),kff[p2,:,ss])*mul[ss]*kv*np.dot(np.conjugate(kff[p3,:,ss]),kff[p4,:,ss])*np.sqrt(2*S[ss]/om0[ss]) 
            for ss in range(saitnum):
                sum2+=np.dot(np.conjugate(kff[p3,:,ss]),kff[p4,:,ss])*mul[ss]*kv*np.dot(np.conjugate(kff[p1,:,ss]),kff[p2,:,ss])*np.sqrt(2*S[ss]/om0[ss])      
    if ke!=0:
        sum2+=ke*ksi2(p1,p2,p3,p4,saitnum,mul,kff)

    return sum2

# def theta( ra,  rb,  i,Bf, v2,saitnum):
#     summ=0
#     for l in range(saitnum*(saitnum-1)//2):
#             summ+=np.conjugate(Bf[l*v2+i%v2,ra])*Bf[l*v2+i%v2,rb]    
#     return summ 
# def theta2( ra,  rb,rc,rd,  i,j,Bf, v2,saitnum):
#     summ=0
#     for l in range(saitnum*(saitnum-1)//2):
#             summ+=np.conjugate(Bf[l*v2+i%v2,ra])*Bf[l*v2+i%v2,rb]*np.conjugate(Bf[l*v2+j%v2,rc])*Bf[l*v2+j%v2,rd]        
#     return summ 
# def theta__( ra, rb, i, i_s, sg,Bf,v2, saitnum, virpnum,virp):
#     summ=0
#     for l in range(saitnum*(saitnum-1)//2):
#         if virp[i,i_s]==0 and sg<0:
#             continue
#         elif virp[i,i_s]==virpnum-1 and sg>0:
#             continue
#         else:
#             i_=np.zeros(saitnum)
#             i_=np.copy(virp[i])
#             i_[i_s]+=sg
#             i__=np.where(np.all(virp==i_,axis=1))[0][0]
#             summ+=np.conjugate(Bf[l*v2+i%v2,ra])*Bf[l*v2+i__,rb] #sg*virpnum**(np.shape(virp)[1]-1-i_s)+
#             #print(sg*virpnum**(np.shape(virp)[1]-1-i_s)+i,i)
#     return summ    


@jit
def h_g(i, j,k, l, virp, virpnum, saitnum,mul):
    summ=0
    def krok(iii,jjj,ss):
        if virp[jjj,ss]==virpnum-1:
            return 0    
        elif [x for y,x in enumerate(virp[iii]) if y!=ss]==[x for y,x in enumerate(virp[jjj]) if y!=ss] and virp[iii][ss]==virp[jjj][ss]+1 :
            return 1
        return 0
    for ii in range(saitnum):
        summ+=mul[ii]*kvv*(np.sqrt(virp[i,ii]*virp[j,ii])*krok(i,j,ii)*krok(k,l,ii)
        +np.sqrt((virp[i,ii]+1)*virp[j,ii])*krok(j,i,ii)*krok(k,l,ii)
        +np.sqrt(virp[i,ii]*(virp[k,ii]+1))*krok(i,j,ii)*krok(l,k,ii)
        +np.sqrt((virp[i,ii]+1)*(virp[k,ii]+1))*krok(j,i,ii)*krok(l,k,ii))
    return summ

# @jit        
# def Corrcoff(numG,numE,numF,B,Bf,virp,v2,virpnum,saitnum,S=0,mul=1,FR=True,OM=1):
#     if np.size(OM)==1:
#         OM=np.array([OM]*saitnum)
#     if np.size(S)==1:
#         S=np.array([S]*saitnum)
#     if np.size(mul)==1:
#         mul=np.array([mul]*saitnum)

#     kff=np.zeros([numE,v2,saitnum])
#     for p in range(numE):
#             for nn in range(saitnum):
#                 for ii in range(v2):
#                     kff[p,ii,nn]=koff(B,p,ii, nn, v2)
#     Dll1=np.zeros([numE,numE,v2,saitnum])
#     Dll_1=np.zeros([numE,numE,v2,saitnum])
#     print(datetime.datetime.now())
#     print("Calculating coff\n")
#     if kv!=0 or kvv!=0:
#         for p11 in range(numE):
#             for p22 in range(numE):
#                 for ii in range(v2):
#                     for ss in range(saitnum):
#                         Dll1[p11,p22,ii,ss]=ksi__(p11,p22,ii,ss,+1,B,v2,saitnum, virpnum,virp,kff)
#                         Dll_1[p11,p22,ii,ss]=ksi__(p11,p22,ii,ss,-1,B,v2,saitnum, virpnum,virp,kff)

#     print(datetime.datetime.now())
#     print("Calculating coff done\n")
    
#     CorrOffd=np.zeros((numE+numG+numF,numE+numG+numF))
#     CorrD=np.zeros((numE+numG+numF,numE+numG+numF))
#     for ii in range(numE+numG+numF):
#         for jj in range(numE+numG+numF):#ii+1):
#             if ii<numG and jj<numG:
#                 CorrOffd[ii,jj]=0#h_g(ii,jj,jj,ii,virp,virpnum,saitnum,mul)
#                 CorrD[ii,jj]=0#h_g(ii,ii,jj,jj,virp,virpnum,saitnum,mul) 
#             if jj<numG and ii>=numG and ii<numG+numE:
#                 CorrOffd[ii,jj]=0
#                 CorrD[ii,jj]=0    
#             if jj>=numG and ii>=numG and jj<(numG+numE) and ii<(numG+numE):
#                 CorrOffd[ii,jj]=h_e(ii-numG,jj-numG,jj-numG,ii-numG,v2,saitnum, virpnum,B, virp,S,mul,Dll1,Dll_1,kff,OM,FR)
#                 CorrD[ii,jj]=h_e(ii-numG,ii-numG,jj-numG,jj-numG,v2,saitnum, virpnum,B, virp,S,mul,Dll1,Dll_1,kff,OM,FR)             
#             # if ii>=(numG+numE) and jj>=(numG) and jj<(numG+numE):
#             #     #CorrOffd[ii,jj]=1#h_ef(ii-numG,jj-numG,ii-numG-numE,jj-numG-numE,v2,saitnum, virpnum,B,Bf, virp)
#             #     CorrD[ii,jj]=h_ef(jj-numG,jj-numG,ii-numG-numE,ii-numG-numE,v2,saitnum, virpnum,B,Bf, virp) 
#             # if jj>=(numG+numE) and ii>=(numG+numE):
#             #     CorrOffd[ii,jj]=h_ff(ii-numG-numE,jj-numG-numE,jj-numG-numE,ii-numG-numE,v2,saitnum, virpnum,Bf, virp)
#             #     CorrD[ii,jj]=h_ff(ii-numG-numE,ii-numG-numE,jj-numG-numE,jj-numG-numE,v2,saitnum, virpnum,Bf, virp) 
#     print(datetime.datetime.now(),CorrD)
#     print("returning coff done\n")            
#     return CorrOffd,CorrD


@jit(cache=True)    
def Corrcoff_2(numG,numE,numF,B,Bf,virp,v2,virpnum,saitnum,S=0,mul=1,FR=True,OM=1,snum=1):
    if np.size(OM)==1:
        OM=np.array([OM]*saitnum)
    if np.size(S)==1:
        S=np.array([S]*saitnum)
    hvirp={}
    for i in range(v2):
         hvirp[virp[i].tobytes()]=i
    kff=np.moveaxis((B.reshape(saitnum,v2,saitnum*v2)), [0, 1, 2], [2, 1, 0])

    # for p in range(numE):
    #         for nn in range(saitnum):
    #             for ii in range(v2):
    #                 kff[p,ii,nn]=koff(B,p,ii, nn, v2)
    #print("22")
    Dll1=np.zeros([numE,numE,v2,saitnum])
    Dll_1=np.zeros([numE,numE,v2,saitnum])
    Dll1,Dll_1=ksi_opt_lop(numE,numG,v2,kvv, saitnum, virpnum, virp, kff, hvirp)
    # if kv!=0 or kvv!=0:
    #     for p11 in range(numE):
    #         for p22 in range(numE):
    #             for ii in range(v2):
    #                 for ss in range(saitnum):
    #                    # Dll1[p11,p22,ii,ss]=ksi__(p11,p22,ii,ss,+1,saitnum, virpnum,virp,kff,hvirp)
    #                    # Dll_1[p11,p22,ii,ss]=ksi__(p11,p22,ii,ss,-1,saitnum, virpnum,virp,kff,hvirp)
    #                     Dll1[p11,p22,ii,ss],a,b=ksi_opt(p11,p22,ii,ss,+1,saitnum, virpnum,virp,kff,hvirp)
    #                     if a!=b:
    #                         Dll_1[p22,p11,b,ss]=Dll1[p11,p22,ii,ss]
    #print("22")
    CorrOffd=np.zeros((snum,numE+numG+numF,numE+numG+numF))
    CorrD=np.zeros((snum,numE+numG+numF,numE+numG+numF))
    for zz in range(snum):
        for ii in range(numE+numG+numF):
            for jj in range(numE+numG+numF):
                if ii<numG and jj<numG:
                    CorrOffd[zz,ii,jj]=0#h_g(ii,jj,jj,ii,virp,virpnum,saitnum,mul)
                    CorrD[zz,ii,jj]=0#h_g(ii,ii,jj,jj,virp,virpnum,saitnum,mul) 
                if jj<numG and ii>=numG and ii<numG+numE:
                    CorrOffd[zz,ii,jj]=0
                    CorrD[zz,ii,jj]=0    
                if jj>=numG and ii>=numG and jj<(numG+numE) and ii<(numG+numE):
                    CorrOffd[zz,ii,jj]=h_e(ii-numG,jj-numG,jj-numG,ii-numG,saitnum, virp,S,mul[zz],Dll1,Dll_1,kff,OM,FR)
                    CorrD[zz,ii,jj]=h_e(ii-numG,ii-numG,jj-numG,jj-numG,saitnum, virp,S,mul[zz],Dll1,Dll_1,kff,OM,FR)
    return CorrOffd,CorrD


# @jit
# def ksi21(pa,pb,pc,pd  ,i,j,B, v2, saitnum,mul,kff):
#     summ=0
#           #  summ+=np.conjugate(koff(B,pa,i,l,v2))*koff(B,pb,i,l,v2)*np.conjugate(koff(B,pc,j,l,v2))*koff(B,pd,j,l,v2)*mul[l]  
#            # kff[p,nn,ii]    
#     summ+=np.dot(np.conjugate(kff[pa,:]),kff[pb,:])*np.dot(np.conjugate(kff[pc,:]),kff[pd,:])*mul
#     return summ 


# @jit
# def ksi__1(pa, pb, i, i_s, sg,B,v2, saitnum, virpnum, virp,kff):
#     summ=0
#     if virp[i]==0 and sg<0:
#         return 0
#     elif virp[i]==virpnum-1 and sg>0:
#         return 0
#     elif virp[i]==virpnum-1 and sg>0:
#         return 0     
#     else:
#         i_=np.zeros(saitnum)
#         i_=np.copy(virp[i])
#         i_+=sg
#         # tes=np.size(np.where(np.all(virp==i_,axis=1)))
#         # if tes==0:
#         #     summ+=0
#         #     return 0
        
#         #summ+=np.conjugate(koff(B,pa,i,l,v2))*koff(B,pb,i__,l,v2)
#         summ+=np.dot(np.conjugate(kff[pa,i]),kff[pb,i_])        
#     return summ    
# @jit
# def h_e1(p1, p2,p3,p4,v2, saitnum, virpnum,B, virp,S,mul,dll1,dll_1,kff,om0,FR=True):
 
#     sum2=0
#     if kvv!=0:
            
#         sum2+=kvv*mul*(np.dot(np.sqrt(virp[:]),dll_1[p1,p2,:])*np.dot(np.sqrt(virp[:]),dll_1[p3,p4,:])
#         +np.dot(np.sqrt((virp[:]+1)),dll1[p1,p2,:])*np.dot(np.sqrt(virp[:]),dll_1[p3,p4,:])
#         +np.dot(np.sqrt(virp[:]),dll_1[p1,p2,:])*np.dot(np.sqrt((virp[:]+1)),dll1[p3,p4,:])
#         +np.dot(np.sqrt((virp[:]+1)),dll1[p1,p2,:])*np.dot(np.sqrt((virp[:]+1)),dll1[p3,p4,:]))/(2*om0)


#         #print(sum2)
#         if FR: # and ii==0: p1==p2 and 
#             #sum2+=kvv*2*mul[ss]*np.sqrt(S[ss])*(np.sqrt(virp[jj,ss])*ksi__(p3,p4,jj,ss,-1,B,v2,saitnum, virpnum,virp)+np.sqrt(virp[jj,ss]+1)*ksi__(p3,p4,jj,ss,+1,B,v2,saitnum, virpnum,virp))
#             #sum2+=kvv*2*mul[ss]*np.sqrt(S[ss])*(np.sqrt(virp[jj,ss])*dll_1[p3,p4,jj,ss]+np.sqrt(virp[jj,ss]+1)*dll1[p3,p4,jj,ss])
#             sum2+=np.dot(np.conjugate(kff[p1,:]),kff[p2,:])*kvv*2*mul*np.sqrt(S)*(np.dot(np.sqrt(virp[:]),dll_1[p3,p4,:])+np.dot(np.sqrt(virp[:]+1),dll1[p3,p4,:]))/(2*om0)
#         if FR: # and jj==0: p3==p4 and 
#             #sum2+=kvv*2*mul[ss]*np.sqrt(S[ss])*(np.sqrt(virp[ii,ss])*ksi__(p1,p2,ii,ss,-1,B,v2,saitnum, virpnum,virp)+np.sqrt(virp[ii,ss]+1)*ksi__(p1,p2,ii,ss,+1,B,v2,saitnum, virpnum,virp))
#             #sum2+=kvv*2*mul[ss]*np.sqrt(S[ss])*(np.sqrt(virp[ii,ss])*dll_1[p1,p2,ii,ss]+np.sqrt(virp[ii,ss]+1)*dll1[p1,p2,ii,ss])  
#             sum2+=np.dot(np.conjugate(kff[p3,:]),kff[p4,:])*kvv*2*mul*np.sqrt(S)*(np.dot(np.sqrt(virp[:]),dll_1[p1,p2,:])+np.dot(np.sqrt(virp[:]+1),dll1[p1,p2,:]))/(2*om0)  
#         if FR:# and jj==0 and ii==0: (p1==p2) and (p3==p4) and 
#             sum2+=4*kvv*mul*S*np.dot(np.conjugate(kff[p1,:]),kff[p2,:])*np.dot(np.conjugate(kff[p3,:]),kff[p4,:])/(2*om0)  

#     if kv!=0:

#         sum2+=mul*kv*np.dot(np.conjugate(kff[p3,:]),kff[p4,:])*(np.dot(np.sqrt(virp[:]),dll_1[p1,p2,:])+np.dot(np.sqrt(virp[:]+1),dll1[p1,p2,:]))/np.sqrt(2*om0)
 
#         sum2+=mul*kv*np.dot(np.conjugate(kff[p1,:]),kff[p2,:])*(np.dot(np.sqrt(virp[:]),dll_1[p3,p4,:])+np.dot(np.sqrt(virp[:]+1),dll1[p3,p4,:]))/np.sqrt(2*om0)   
#         if FR:
#            # if p1==p2:
#             sum2+=np.dot(np.conjugate(kff[p1,:]),kff[p2,:])*mul*kv*np.dot(np.conjugate(kff[p3,:]),kff[p4,:])*np.sqrt(2*S/om0)#*np.sqrt(4*S)# 
#            # if p3==p4:

#             sum2+=np.dot(np.conjugate(kff[p3,:]),kff[p4,:])*mul*kv*np.dot(np.conjugate(kff[p1,:]),kff[p2,:])*np.sqrt(2*S/om0)#*np.sqrt(4*S)#       
#     if ke!=0:
#        # for ii in range(v2):
#         #    for jj in range(v2)
#         ii=0
#         jj=0
#         sum2+=ke*ksi21(p1,p2,p3,p4,ii,jj,B,v2,saitnum,mul,kff)#ksi(p1,p2,ii,B,v2,saitnum)*ksi(p3,p4,jj,B,v2,saitnum)    
#     return sum2    
# @jit
# def Corrcoff1(numG,numE,numF,B,Bf,virp,v2,virpnum,saitnum,S=0,mul=1,FR=True,OM=1):
#     kff=np.zeros([numE,v2])
#     for p in range(numE):
#             for nn in range(saitnum):
#                 for ii in range(v2):
#                     kff[p,ii]=koff(B,p,ii, nn, v2)
#     Dll1=np.zeros([numE,numE,v2])
#     Dll_1=np.zeros([numE,numE,v2])
#     print(datetime.datetime.now())
#     print("Calculating coff\n")
#     if kv!=0 or kvv!=0:
#         for p11 in range(numE):
#             for p22 in range(numE):
#                 for ii in range(v2):
#                     for ss in range(saitnum):
#                         Dll1[p11,p22,ii]=ksi__1(p11,p22,ii,ss,+1,B,v2,saitnum, virpnum,virp,kff)
#                         Dll_1[p11,p22,ii]=ksi__1(p11,p22,ii,ss,-1,B,v2,saitnum, virpnum,virp,kff)
#     print(datetime.datetime.now())
#     print("Calculating coff done\n")
    
#     CorrOffd=np.zeros((numE+numG+numF,numE+numG+numF))
#     CorrD=np.zeros((numE+numG+numF,numE+numG+numF))
#     for ii in range(numE+numG+numF):
#         for jj in range(numE+numG+numF):#ii+1):
#             if ii<numG and jj<numG:
#                 CorrOffd[ii,jj]=0#h_g(ii,jj,jj,ii,virp,virpnum,saitnum,mul)
#                 CorrD[ii,jj]=0#h_g(ii,ii,jj,jj,virp,virpnum,saitnum,mul) 
#             if jj<numG and ii>=numG and ii<numG+numE:
#                 CorrOffd[ii,jj]=0
#                 CorrD[ii,jj]=0    
#             if jj>=numG and ii>=numG and jj<(numG+numE) and ii<(numG+numE):
#                 CorrOffd[ii,jj]=h_e1(ii-numG,jj-numG,jj-numG,ii-numG,v2,saitnum, virpnum,B, virp,S,mul,Dll1,Dll_1,kff,OM,FR)
#                 CorrD[ii,jj]=h_e1(ii-numG,ii-numG,jj-numG,jj-numG,v2,saitnum, virpnum,B, virp,S,mul,Dll1,Dll_1,kff,OM,FR)             
#             # if ii>=(numG+numE) and jj>=(numG) and jj<(numG+numE):
#             #     #CorrOffd[ii,jj]=1#h_ef(ii-numG,jj-numG,ii-numG-numE,jj-numG-numE,v2,saitnum, virpnum,B,Bf, virp)
#             #     CorrD[ii,jj]=h_ef(jj-numG,jj-numG,ii-numG-numE,ii-numG-numE,v2,saitnum, virpnum,B,Bf, virp) 
#             # if jj>=(numG+numE) and ii>=(numG+numE):
#             #     CorrOffd[ii,jj]=h_ff(ii-numG-numE,jj-numG-numE,jj-numG-numE,ii-numG-numE,v2,saitnum, virpnum,Bf, virp)
#             #     CorrD[ii,jj]=h_ff(ii-numG-numE,ii-numG-numE,jj-numG-numE,jj-numG-numE,v2,saitnum, virpnum,Bf, virp) 
#     print(datetime.datetime.now())#,CorrD)
#     print("returning coff done\n")            
#     return CorrOffd,CorrD
