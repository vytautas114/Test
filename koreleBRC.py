import numpy as np
from numba import jit
# import datetime
# num_of_proces=4

# @jit
def koff(B, p, i, m, v2):
    return B[m * v2 + i % v2, p]


@jit(cache=True)
def ksi_opt_lop(numE, numG, v2, saitnum, virp, kff, hvirp, kvv=0, kv=0):
    Dll1 = np.zeros([numE, numE, v2, saitnum])
    Dll_1 = np.zeros([numE, numE, v2, saitnum])
    if kv != 0 or kvv != 0:
        for pb in range(numE):
            for pa in range(numE):
                for i in range(v2):
                    for i_s in range(saitnum):
                        i_ = np.zeros(saitnum, dtype=np.int)
                        i_[i_s] += 1
                        hh = (virp[i] + i_).tobytes()
                        if hh in hvirp:
                            Dll1[pa, pb, i, i_s] = np.conjugate(kff[pa, i, i_s]) * kff[pb, hvirp[hh], i_s]
                            Dll_1[pb, pa, hvirp[hh], i_s] = np.conjugate(Dll1[pa, pb, i, i_s])

    return Dll1, Dll_1


@jit(cache=True)
def calc_Xi(numE, v2, saitnum, virp, kff, hvirp):
    Dll1 = np.zeros([numE, numE, saitnum])
    Dll_1 = np.zeros([numE, numE, saitnum])
    i_ = np.zeros(saitnum, dtype=np.int)
    for i_s in range(saitnum):
        i_[i_s] += 1
        for i in range(v2):
            hh = (virp[i] + i_).tobytes()
            if hh in hvirp:
                Dll1,Dll_1=Xi_inner(i_s,Dll1,Dll_1,numE,saitnum,i,hvirp[hh],virp,kff)
        i_[i_s] -=1
    return Dll1, Dll_1


@jit(cache=True,nopython=True)
def Xi_inner(i_s,Dll1,Dll_1,numE,saitnum,i,i2,virp,kff):
    for pb in range(numE):
        for pa in range(numE):
            for n in range(saitnum):
                Dll1[pa, pb, i_s] += np.sqrt(virp[i,i_s]+1)*np.conjugate(kff[pa, i, n]) * kff[pb, i2, n]#np.einum("ik,lk",np.conjugate(kff[pa, i, n]) * kff[pb, i2, n])
                Dll_1[pa, pb, i_s] += np.sqrt(virp[i2,i_s])*np.conjugate(kff[pa, i2, n])*kff[pb, i, n]
    return Dll1,Dll_1


@jit(cache=True,nopython=True)
def He_e(pa,pb,pc,pd,mul,numE,saitnum,kff):
    temp=0.0
    temp=ksi2(pa, pb, pc, pd, saitnum, mul, kff)

    return temp


@jit(cache=True,nopython=True)
def He_vv(pa,pb,pc,pd,mul,numE,saitnum,v2,s,kff,Xip,Xim):
    temp=0.0
    temp+=4*ksi2(pa, pb, pc, pd, saitnum, mul*s, kff)
    for n in range(saitnum):
        temp+=mul[n]*(Xip[pa,pb,n]*Xip[pc,pd,n]
        +Xip[pa,pb,n]*Xim[pc,pd,n]
        +Xim[pa,pb,n]*Xip[pc,pd,n]
        +Xim[pa,pb,n]*Xim[pc,pd,n])
        for i in range(v2):
            temp+=2*np.sqrt(s[n])*mul[n]*(kff[pa,i,n]*kff[pb,i,n]*(Xip[pc,pd,n]+Xim[pc,pd,n])+
                                kff[pc,i,n]*kff[pd,i,n]*(Xip[pa,pb,n]+Xim[pa,pb,n]))

    return temp


@jit(nopython=True, cache=True)
def ksi2(pa, pb, pc, pd, saitnum, mul, kff):
    summ = 0
    for l in range(saitnum):
        summ += np.dot(np.conjugate(kff[pa, :, l]), kff[pb, :, l]) * np.dot(np.conjugate(kff[pc, :, l]), kff[pd, :, l]) * mul[l]
    # summ = np.einsum("il,il,jl,jl,l",np.conjugate(kff[pa]),kff[pb],np.conjugate(kff[pc]),kff[pd],mul)
    return summ


@jit(nopython=True, cache=True)
def h_e(p1, p2, p3, p4, saitnum, virp, S, mul, dll1, dll_1, kff, om0, ke=1, kvv=0, kv=0, FR=True):
    sum2 = 0
    if kvv != 0:
        for ss in range(saitnum):
            sum2 += kvv * mul[ss] * (np.dot(np.sqrt(virp[:, ss]), dll_1[p1, p2, :, ss]) * np.dot(np.sqrt(virp[:, ss]), dll_1[p3, p4, :, ss]) +
                                     np.dot(np.sqrt((virp[:, ss] + 1)), dll1[p1, p2, :, ss]) * np.dot(np.sqrt(virp[:, ss]), dll_1[p3, p4, :, ss]) +
                                     np.dot(np.sqrt(virp[:, ss]), dll_1[p1, p2, :, ss]) * np.dot(np.sqrt((virp[:, ss] + 1)), dll1[p3, p4, :, ss]) +
                                     np.dot(np.sqrt((virp[:, ss] + 1)), dll1[p1, p2, :, ss]) * np.dot(np.sqrt((virp[:, ss] + 1)), dll1[p3, p4, :, ss])) / (2 * om0[ss])

            if FR:
                sum2 += np.dot(np.conjugate(kff[p1, :, ss]), kff[p2, :, ss]) * kvv * 2 * mul[ss] * np.sqrt(S[ss]) * (np.dot(np.sqrt(virp[:, ss]), dll_1[p3, p4, :, ss]) + np.dot(np.sqrt(virp[:, ss] + 1), dll1[p3, p4, :, ss])) / (2 * om0[ss])
            if FR:
                sum2 += np.dot(np.conjugate(kff[p3, :, ss]), kff[p4, :, ss]) * kvv * 2 * mul[ss] * np.sqrt(S[ss]) * (np.dot(np.sqrt(virp[:, ss]), dll_1[p1, p2, :, ss]) + np.dot(np.sqrt(virp[:, ss] + 1), dll1[p1, p2, :, ss])) / (2 * om0[ss])
            if FR:
                sum2 += 4 * kvv * mul[ss] * S[ss] * np.dot(np.conjugate(kff[p1, :, ss]), kff[p2, :, ss]) * np.dot(np.conjugate(kff[p3, :, ss]), kff[p4, :, ss]) / (2 * om0[ss])

    if kv != 0:
        for ss in range(saitnum):
            sum2 += mul[ss] * kv * np.dot(np.conjugate(kff[p3, :, ss]), kff[p4, :, ss]) * (np.dot(np.sqrt(virp[:, ss]), dll_1[p1, p2, :, ss]) + np.dot(np.sqrt(virp[:, ss] + 1), dll1[p1, p2, :, ss])) / np.sqrt(2 * om0[ss])
            sum2 += mul[ss] * kv * np.dot(np.conjugate(kff[p1, :, ss]), kff[p2, :, ss]) * (np.dot(np.sqrt(virp[:, ss]), dll_1[p3, p4, :, ss]) + np.dot(np.sqrt(virp[:, ss] + 1), dll1[p3, p4, :, ss])) / np.sqrt(2 * om0[ss])

        if FR:
            for ss in range(saitnum):
                sum2 += np.dot(np.conjugate(kff[p1, :, ss]), kff[p2, :, ss]) * mul[ss] * kv * np.dot(np.conjugate(kff[p3, :, ss]), kff[p4, :, ss]) * np.sqrt(2 * S[ss] / om0[ss])
            for ss in range(saitnum):
                sum2 += np.dot(np.conjugate(kff[p3, :, ss]), kff[p4, :, ss]) * mul[ss] * kv * np.dot(np.conjugate(kff[p1, :, ss]), kff[p2, :, ss]) * np.sqrt(2 * S[ss] / om0[ss])
    if ke != 0:
        sum2 += ke * ksi2(p1, p2, p3, p4, saitnum, mul, kff)

    return sum2


@jit
def h_g(i, j, k, l, virp, virpnum, saitnum, mul, kvv):
    summ = 0

    def krok(iii, jjj, ss):
        if virp[jjj, ss] == virpnum - 1:
            return 0
        elif [x for y, x in enumerate(virp[iii]) if y != ss] == [x for y, x in enumerate(virp[jjj]) if y != ss] and virp[iii][ss] == virp[jjj][ss] + 1:
            return 1
        return 0
    for ii in range(saitnum):
        summ += mul[ii] * kvv * (np.sqrt(virp[i, ii] * virp[j, ii]) * krok(i, j, ii) * krok(k, l, ii) +
                                 np.sqrt((virp[i, ii] + 1) * virp[j, ii]) * krok(j, i, ii) * krok(k, l, ii) +
                                 np.sqrt(virp[i, ii] * (virp[k, ii] + 1)) * krok(i, j, ii) * krok(l, k, ii) +
                                 np.sqrt((virp[i, ii] + 1) * (virp[k, ii] + 1)) * (j, i, ii) * krok(l, k, ii))
    return summ


@jit(cache=True)
def Corrcoff_2(numG, numE, B, virp, v2, saitnum, S=0, mul=1, snum=1,C_type=['e']):

    if np.size(S) == 1:
        S = np.array([S] * saitnum)
    hvirp = {}
    for i in range(v2):
        hvirp[virp[i].tobytes()] = i
    kff = np.moveaxis((B.reshape(saitnum, v2, saitnum * v2)), [0, 1, 2], [2, 1, 0])


    # Dll1, Dll_1 = ksi_opt_lop(numE, numG, v2, saitnum, virp, kff, hvirp, kvv=kvv, kv=kv)
    Xim,Xip=calc_Xi(numE, v2, saitnum, virp, kff, hvirp)

    CorrOffd = np.zeros((snum, numE + numG, numE + numG))
    CorrD = np.zeros((snum, numE + numG, numE + numG))
    for zz in range(snum):
        for ii in range(numE + numG):
            for jj in range(numE + numG):
                if ii < numG and jj < numG:
                    CorrOffd[zz, ii, jj] = 0  # h_g(ii,jj,jj,ii,virp,virpnum,saitnum,mul)
                    CorrD[zz, ii, jj] = 0  # h_g(ii,ii,jj,jj,virp,virpnum,saitnum,mul)
                if jj < numG and ii >= numG and ii < numG + numE:
                    CorrOffd[zz, ii, jj] = 0
                    CorrD[zz, ii, jj] = 0
                if jj >= numG and ii >= numG and jj < (numG + numE) and ii < (numG + numE):
                    if C_type[zz]=='e':
                        CorrOffd[zz, ii, jj] = He_e(ii - numG, jj - numG, jj - numG, ii - numG,mul[zz],numE,saitnum,kff)
                        CorrD[zz, ii, jj] = He_e(ii - numG, ii - numG, jj - numG, jj - numG,mul[zz],numE,saitnum,kff)
                    elif C_type[zz]=='v':
                        CorrOffd[zz, ii, jj] = He_vv(ii - numG, jj - numG, jj - numG, ii - numG,mul[zz],numE,saitnum,v2,S,kff,Xip,Xim)#0#h_e(ii - numG, jj - numG, jj - numG, ii - numG, saitnum, virp, S, mul[zz], Dll1, Dll_1, kff, OM, ke=ke, kvv=kvv, kv=kv, FR=FR)
                        CorrD[zz, ii, jj] = He_vv(ii - numG, ii - numG, jj - numG, jj - numG,mul[zz],numE,saitnum,v2,S,kff,Xip,Xim)#0#h_e(ii - numG, ii - numG, jj - numG, jj - numG, saitnum, virp, S, mul[zz], Dll1, Dll_1, kff, OM, ke=ke, kvv=kvv, kv=kv, FR=FR)
    return CorrOffd, CorrD
