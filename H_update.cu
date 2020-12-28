#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fdtd3d.h"

__global__ void H_update( int Nr, int Nth, int Nph, float *Er, float *Eth, float *Eph,
                    float *Hr, float *Hth, float *Hph, float del_r, float del_th, float del_ph, float dt,
                    float th0, float r0, float mu, int L )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
    int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

    if( ((i>=1)&&(i<Nr+1)) && ((j>=L)&&(j<Nth-L)) && ((k>=L)&&(k<Nph-L)) ){
        int idx_Hr = idx_Hr_d( i, j, k, Nth, Nph );
        int idx_Eph1 = idx_Eph_d( i, j+1, k, Nth, Nph );
        int idx_Eph2 = idx_Eph_d( i, j, k, Nth, Nph );
        int idx_Eth1 = idx_Eth_d( i, j, k+1, Nth, Nph );
        int idx_Eth2 = idx_Eth_d( i, j, k, Nth, Nph );

        float r_i1 = dist_d( float(i), del_r, r0 );
        float si_th1 = std::sin( th_d( float(j), del_th, th0 ) );
        float si_th2 = std::sin( th_d( float(j)+0.5, del_th, th0 ) );
        float si_th3 = std::sin( th_d( float(j)+1.0, del_th, th0 ) );

        float CHr1 = dt/mu/r_i1/si_th2/del_th;
        float CHr2 = dt/mu/r_i1/si_th2/del_ph;

        Hr[idx_Hr] = Hr[idx_Hr]
            - CHr1*(si_th3*Eph[idx_Eph1] - si_th1*Eph[idx_Eph2])
            + CHr2*(Eth[idx_Eth1] - Eth[idx_Eth2] );
            
    }

    if( ((i>=1)&(i<Nr)) && ((j>=L+1)&&(j<Nth-L)) && ((k>=L)&&(k<Nph-L)) ){
        int idx_Hth = idx_Hth_d( i, j, k, Nth, Nph );
        int idx_Er1 = idx_Er_d( i, j, k+1, Nth, Nph );
        int idx_Er2 = idx_Er_d( i, j, k, Nth, Nph );
        int idx_Eph1 = idx_Eph_d( i+1, j, k, Nth, Nph );
        int idx_Eph2 = idx_Eph_d( i, j, k, Nth, Nph );

        float r_i1 = dist_d( float(i), del_r, r0 );
        float r_i2 = dist_d( float(i)+0.5, del_r, r0 );
        float r_i3 = dist_d( float(i)+1.0, del_r, r0 );
        float si_th1 = std::sin( th_d( float(j), del_th, th0 ) );

        float CHth1 = dt/mu/r_i2/si_th1/del_ph;
        float CHth2 = dt/mu/r_i2/del_r;

        Hth[idx_Hth] = Hth[idx_Hth]
            - CHth1*(Er[idx_Er1] - Er[idx_Er2])
            + CHth2*(r_i3*Eph[idx_Eph1] - r_i1*Eph[idx_Eph2]);

    }

    if( ((i>=1)&&(i<Nr)) && ((j>=L)&&(j<Nth-L)) && ((k>=L+1)&&(k<Nph-L)) ){
        int idx_Hph = idx_Hph_d( i, j, k, Nth, Nph );
        int idx_Eth1 = idx_Eth_d( i+1, j, k, Nth, Nph );
        int idx_Eth2 = idx_Eth_d( i, j, k, Nth, Nph );
        int idx_Er1 = idx_Er_d( i, j+1, k, Nth, Nph );
        int idx_Er2 = idx_Er_d( i, j, k, Nth, Nph );

        float r_i1 = dist_d( float(i), del_r, r0 );
        float r_i2 = dist_d( float(i)+0.5, del_r, r0 );
        float r_i3 = dist_d( float(i)+1.0, del_r, r0 );
    
        float CHph1 = dt/mu/r_i2/del_r;
        float CHph2 = dt/mu/r_i2/del_th;

        Hph[idx_Hph] = Hph[idx_Hph]
            - CHph1*(r_i3*Eth[idx_Eth1] - r_i1*Eth[idx_Eth2])
            + CHph2*(Er[idx_Er1] - Er[idx_Er2]);

    }
}