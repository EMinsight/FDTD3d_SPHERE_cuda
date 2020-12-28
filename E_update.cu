#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fdtd3d.h"

__global__ void E_update( int Nr, int Nth, int Nph,
                    float *nEr, float *nEth, float *nEph,
                    float *oEr, float *oEth, float *oEph,
                    float *nDr, float *nDth, float *nDph,
                    float *oDr, float *oDth, float *oDph, 
                    float *Cmat, float *Fmat, 
                    int ion_L, float eps )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
    int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

    if( ((i >= 0)&&(i < Nr - ion_L)) && ((j >= 1)&&(j < Nth)) && ((k >= 1)&&(k < Nph)) ){
        int idx = idx_Er_d( i, j, k, Nth, Nph );
        nEr[idx] = oEr[idx]
                + (nDr[idx] - oDr[idx])/eps;
        
       oDr[idx] = nDr[idx];
    }

    if( ((i >= 1)&&(i < Nr - ion_L)) && ((j >= 0)&&(j < Nth)) && ((k >= 1)&&(k < Nph)) ){
        int idx = idx_Eth_d( i, j, k, Nth, Nph );
        
        nEth[idx] = oEth[idx]
                + (nDth[idx] - oDth[idx])/eps;
    
        oDth[idx] = nDth[idx];
    }

    if( ((i >= 1)&&(i < Nr - ion_L)) && ((j >= 1)&&(j < Nth)) && ((k >= 0)&&(k < Nph)) ){
        int idx = idx_Eph_d( i, j, k, Nth, Nph );

        nEph[idx] = oEph[idx]
                + (nDph[idx] - oDph[idx])/eps;
        
        oDph[idx] = nDph[idx];
    }

    if( ((i >= Nr - ion_L)&&(i < Nr)) && ((j >= 1)&&(j < Nth)) && ( (k >= 1)&&(k < Nph)) ){
        int m = i - (Nr - ion_L);
        int Ir = 0;
        int Ith = 1;
        int Iph = 2;
        int idx_Er = idx_Er_d(i, j, k, Nth, Nph);
        int idx_Eth1 = idx_Eth_d(i, j, k, Nth, Nph);
        int idx_Eth2 = idx_Eth_d(i, j-1, k, Nth, Nph);
        int idx_Eth3 = idx_Eth_d(i+1, j, k, Nth, Nph);
        int idx_Eth4 = idx_Eth_d(i+1, j-1, k, Nth, Nph);
        int idx_Eph1 = idx_Eph_d(i, j, k, Nth, Nph);
        int idx_Eph2 = idx_Eph_d(i, j, k-1, Nth, Nph);
        int idx_Eph3 = idx_Eph_d(i+1, j, k, Nth, Nph);
        int idx_Eph4 = idx_Eph_d(i+1, j, k-1, Nth, Nph);

        float Interpol_Eth = (
            oEth[idx_Eth1] + oEth[idx_Eth2] + 
            oEth[idx_Eth3] + oEth[idx_Eth4] )/float(4.0);
        float Interpol_nDth = (
            nDth[idx_Eth1] + nDth[idx_Eth2] + 
            nDth[idx_Eth3] + nDth[idx_Eth4] )/float(4.0);
        float Interpol_oDth = (
            oDth[idx_Eth1] + oDth[idx_Eth2] + 
            oDth[idx_Eth3] + oDth[idx_Eth4] )/float(4.0);

        float Interpol_Eph = (
            oEph[idx_Eph1] + oEph[idx_Eph2] +
            oEph[idx_Eph3] + oEph[idx_Eph4] )/float(4.0);
        float Interpol_nDph = (
            nDph[idx_Eph1] + nDph[idx_Eph2] +
            nDph[idx_Eph3] + nDph[idx_Eph4] )/float(4.0);
        float Interpol_oDph = (
            oDph[idx_Eph1] + oDph[idx_Eph2] +
            oDph[idx_Eph3] + oDph[idx_Eph4] )/float(4.0);
        
        nEr[idx_Er] = 
            Cmat[idx_mat_d(m,j,k,Ir,Ir,ion_L,Nth,Nph)] * oEr[idx_Er] +
            Cmat[idx_mat_d(m,j,k,Ir,Ith,ion_L,Nth,Nph)] * Interpol_Eth +
            Cmat[idx_mat_d(m,j,k,Ir,Iph,ion_L,Nth,Nph)] * Interpol_Eph +
            Fmat[idx_mat_d(m,j,k,Ir,Ir,ion_L,Nth,Nph)] * (nDr[idx_Er] - oDr[idx_Er]) +
            Fmat[idx_mat_d(m,j,k,Ir,Ith,ion_L,Nth,Nph)] * (Interpol_nDth - Interpol_oDth) +
            Fmat[idx_mat_d(m,j,k,Ir,Iph,ion_L,Nth,Nph)] * (Interpol_nDph - Interpol_oDph);
        
        oDr[idx_Er] = nDr[idx_Er];
    }

    if( ((i >= Nr - ion_L)&&(i < Nr)) && ((j >= 0)&&(j < Nth)) && ((k >= 1)&&(k < Nph)) ){
        int m = i - (Nr - ion_L);
        int Ir = 0;
        int Ith = 1;
        int Iph = 2;
        int idx_Eth = idx_Eth_d(i, j, k, Nth, Nph);
        int idx_Er1 = idx_Er_d(i, j, k, Nth, Nph);
        int idx_Er2 = idx_Er_d(i-1, j, k, Nth, Nph);
        int idx_Er3 = idx_Er_d(i, j+1, k, Nth, Nph);
        int idx_Er4 = idx_Er_d(i-1, j+1, k, Nth, Nph);
        int idx_Eph1 = idx_Eph_d(i, j, k, Nth, Nph);
        int idx_Eph2 = idx_Eph_d(i, j, k-1, Nth, Nph);
        int idx_Eph3 = idx_Eph_d(i, j+1, k, Nth, Nph);
        int idx_Eph4 = idx_Eph_d(i, j+1, k-1, Nth, Nph);

        float Interpol_Er = (
            oEr[idx_Er1] + oEr[idx_Er2] +
            oEr[idx_Er3] + oEr[idx_Er4])/float(4.0);
        float Interpol_nDr = (
            nDr[idx_Er1] + nDr[idx_Er2] +
            nDr[idx_Er3] + nDr[idx_Er4])/float(4.0); 
        float Interpol_oDr = (
            oDr[idx_Er1] + oDr[idx_Er2] +
            oDr[idx_Er3] + oDr[idx_Er4])/float(4.0);

        float Interpol_Eph = (
            oEph[idx_Eph1] + oEph[idx_Eph2] +
            oEph[idx_Eph3] + oEph[idx_Eph4])/float(4.0);
        float Interpol_nDph = (
            nDph[idx_Eph1] + nDph[idx_Eph2] +
            nDph[idx_Eph3] + nDph[idx_Eph4])/float(4.0);
        float Interpol_oDph = (
            oDph[idx_Eph1] + oDph[idx_Eph2] +
            oDph[idx_Eph3] + oDph[idx_Eph4])/float(4.0);
        
        nEth[idx_Eth] = 
            Cmat[idx_mat_d(m,j,k,Ith,Ir,ion_L,Nth,Nph)] * Interpol_Er +
            Cmat[idx_mat_d(m,j,k,Ith,Ith,ion_L,Nth,Nph)] * oEth[idx_Eth] +
            Cmat[idx_mat_d(m,j,k,Ith,Iph,ion_L,Nth,Nph)] * Interpol_Eph +
            Fmat[idx_mat_d(m,j,k,Ith,Ir,ion_L,Nth,Nph)] * (Interpol_nDr - Interpol_oDr) +
            Fmat[idx_mat_d(m,j,k,Ith,Ith,ion_L,Nth,Nph)] * (nDth[idx_Eth] - oDth[idx_Eth]) +
            Fmat[idx_mat_d(m,j,k,Ith,Iph,ion_L,Nth,Nph)] * (Interpol_nDph - Interpol_oDph);
        
        oDth[idx_Eth] = nDth[idx_Eth];
    }

    if( ((i >= Nr - ion_L)&&(i < Nr)) && ((j >= 1)&&(j < Nth)) && ((k >= 0)&&(k < Nph)) ){
        int m = i - (Nr - ion_L);
        int Ir = 0;
        int Ith = 1;
        int Iph = 2;
        int idx_Eph = idx_Eph_d(i, j, k, Nth, Nph);
        int idx_Er1 = idx_Er_d(i, j, k, Nth, Nph);
        int idx_Er2 = idx_Er_d(i-1, j, k, Nth, Nph);
        int idx_Er3 = idx_Er_d(i, j, k+1, Nth, Nph);
        int idx_Er4 = idx_Er_d(i-1, j, k+1, Nth, Nph);
        int idx_Eth1 = idx_Eth_d(i, j, k, Nth, Nph);
        int idx_Eth2 = idx_Eth_d(i, j-1, k, Nth, Nph);
        int idx_Eth3 = idx_Eth_d(i, j, k+1, Nth, Nph);
        int idx_Eth4 = idx_Eth_d(i, j-1, k+1, Nth, Nph);

        float Interpol_Er = (
            oEr[idx_Er1] + oEr[idx_Er2] +
            oEr[idx_Er3] + oEr[idx_Er4])/float(4.0);
        float Interpol_nDr = (
            nDr[idx_Er1] + nDr[idx_Er2] +
            nDr[idx_Er3] + nDr[idx_Er4])/float(4.0);
        float Interpol_oDr = float(
            oDr[idx_Er1] + oDr[idx_Er2] +
            oDr[idx_Er3] + oDr[idx_Er4])/float(4.0);
        
        float Interpol_Eth = (
            oEth[idx_Eth1] + oEth[idx_Eth2] +
            oEth[idx_Eth3] + oEth[idx_Eth4])/float(4.0);
        float Interpol_nDth = (
            nDth[idx_Eth1] + nDth[idx_Eth2] +
            nDth[idx_Eth3] + nDth[idx_Eth4])/float(4.0);
        float Interpol_oDth = (
            oDth[idx_Eth1] + oDth[idx_Eth2] +
            oDth[idx_Eth3] + oDth[idx_Eth4])/float(4.0);
        
        nEph[idx_Eph] =
            Cmat[idx_mat_d(m,j,k,Iph,Ir,ion_L,Nth,Nph)] * Interpol_Er +
            Cmat[idx_mat_d(m,j,k,Iph,Ith,ion_L,Nth,Nph)] * Interpol_Eth +
            Cmat[idx_mat_d(m,j,k,Iph,Iph,ion_L,Nth,Nph)] * oEph[idx_Eph] +
            Fmat[idx_mat_d(m,j,k,Iph,Ir,ion_L,Nth,Nph)] * (Interpol_nDr - Interpol_oDr) +
            Fmat[idx_mat_d(m,j,k,Iph,Ith,ion_L,Nth,Nph)] * (Interpol_nDth - Interpol_oDth) +
            Fmat[idx_mat_d(m,j,k,Iph,Iph,ion_L,Nth,Nph)] * (nDph[idx_Eph] - oDph[idx_Eph]);

        oDph[idx_Eph] = nDph[idx_Eph];
    }
}