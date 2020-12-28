#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fdtd3d.h"

__global__ void D_update( int Nr, int Nth, int Nph, float *nDr, float *nDth, float *nDph,
                         float *oDr, float *oDth, float *oDph, float *Hr, float *Hth, float *Hph,
                         float del_r, float del_th, float del_ph, float dt, float th0, float r0, int L )
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
    int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

    if( ((i>=0)&&(i<Nr)) && ((j>=L+1)&&(j<Nth-L)) && ((k>=L+1)&&(k<Nph-L)) ){
        int idx_Dr = idx_Er_d( i, j, k, Nth, Nph );
        int idx_Hph1 = idx_Hph_d( i, j, k, Nth, Nph );
        int idx_Hph2 = idx_Hph_d( i, j-1, k, Nth, Nph );
        int idx_Hth1 = idx_Hth_d( i, j, k, Nth, Nph );
        int idx_Hth2 = idx_Hth_d( i, j, k-1, Nth, Nph );

        float r_i2 = dist_d( float(i)+0.5, del_r, r0 );
        float si_th1 = std::sin( th_d(float(j)-0.5, del_th, th0) );
        float si_th2 = std::sin( th_d(float(j), del_th, th0) );
        float si_th3 = std::sin( th_d(float(j)+0.5, del_th, th0) );
        
        float CDr1 = dt/r_i2/si_th2/del_th;
        float CDr2 = dt/r_i2/si_th2/del_ph;

        nDr[idx_Dr] = oDr[idx_Dr]
            + CDr1*( si_th3*Hph[idx_Hph1] - si_th1*Hph[idx_Hph2] )
            - CDr2*( Hth[idx_Hth1] - Hth[idx_Hth2] );

    }

    if( ((i>=1)&&(i<Nr)) && ((j>=L)&&(j<Nth-L)) && ((k>=L+1)&&(k<Nph-L)) ){
        int idx_Dth = idx_Eth_d( i, j, k, Nth, Nph );
        int idx_Hr1 = idx_Hr_d( i, j, k, Nth, Nph );
        int idx_Hr2 = idx_Hr_d( i, j, k-1, Nth, Nph );
        int idx_Hph1 = idx_Hph_d( i, j, k, Nth, Nph );
        int idx_Hph2 = idx_Hph_d( i-1, j, k, Nth, Nph );

        float r_i1 = dist_d( float(i)-0.5, del_r, r0 );
        float r_i2 = dist_d( float(i), del_r, r0 );
        float r_i3 = dist_d( float(i)+0.5, del_r, r0 );
        float si_th3 = std::sin( th_d(float(j)+0.5, del_th, th0) );
        
        float CDth1 = dt/r_i2/si_th3/del_ph;
        float CDth2 = dt/r_i2/del_r;

        nDth[idx_Dth] = oDth[idx_Dth]
            + CDth1*( Hr[idx_Hr1] - Hr[idx_Hr2])
            - CDth2*( r_i3*Hph[idx_Hph1] - r_i1*Hph[idx_Hph2]);

    }

    if( ((i>=1)&&(i<Nr)) && ((j>=L+1)&&(j<Nth-L)) && ((k>=L)&&(k<Nph-L)) ){
        int idx_Dph = idx_Eph_d( i, j, k, Nth, Nph );
        int idx_Hth1 = idx_Hth_d( i, j, k, Nth, Nph );
        int idx_Hth2 = idx_Hth_d( i-1, j, k, Nth, Nph );
        int idx_Hr1 = idx_Hr_d( i, j, k, Nth, Nph );
        int idx_Hr2 = idx_Hr_d( i, j-1, k, Nth, Nph );
        
        float r_i1 = dist_d( float(i)-0.5, del_r, r0 );
        float r_i2 = dist_d( float(i), del_r, r0 );
        float r_i3 = dist_d( float(i)+0.5, del_r, r0 );

        float CDph1 = dt/r_i2/del_r;
        float CDph2 = dt/r_i2/del_th;

        nDph[idx_Dph] = oDph[idx_Dph]
            + CDph1*( r_i3*Hth[idx_Hth1] - r_i1*Hth[idx_Hth2])
            - CDph2*( Hr[idx_Hr1] - Hr[idx_Hr2] );

    }

}