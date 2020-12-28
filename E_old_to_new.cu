#include <cuda.h>
#include <cuda_runtime.h>
#include "fdtd3d.h"

__global__ void E_old_to_new( 
        int Nr, int Nth, int Nph,
        float *nEr, float *nEth, float *nEph,
        float *oEr, float *oEth, float *oEph )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
    int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

    if( ((i >= 0 ) && (i < Nr)) && ((j >= 1) && (j < Nth)) && ((k >= 1) && (k < Nph)) ){
        int idx = idx_Er_d( i, j, k, Nth, Nph );
        oEr[idx] = nEr[idx];
    }

    if( ((i >= 1) && (i < Nr)) && ((j >= 0) && (j < Nth)) && ((k >= 1) && (k < Nph)) ){
        int idx = idx_Eth_d( i, j, k, Nth, Nph );
        oEth[idx] = nEth[idx];
    }

    if( ((i >= 1) && (i < Nr)) && ((j >= 1) && (j < Nth)) && ((k >= 0) && (k < Nph)) ){
        int idx = idx_Eph_d( i, j, k, Nth, Nph );
        oEph[idx] = nEph[idx];
    }

}