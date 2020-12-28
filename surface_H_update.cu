#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fdtd3d.h"

__global__ void surface_H_update(
    int Nr, int Nth, int Nph,
    float *nEr, float *nEth, float *nEph, float *Hth, float *Hph,
    float z_real, float z_imag, float del_r, float del_th, float del_ph, float Dt, 
    float r0, float th0, float MU0 )
{  
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
    int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

    if( (i==0) && ((j >= 1)&&(j < Nth+1)) && ((k >= 0)&&(k < Nph)) ){
        int idx_Hth = idx_Hth_d(i, j, k, Nth, Nph);
        int idx_Er1 = idx_Er_d(i, j, k+1, Nth, Nph);
        int idx_Er2 = idx_Er_d(i, j, k, Nth, Nph);
        int idx_Eph = idx_Eph_d(i+1, j, k, Nth, Nph);

        float coeff_1 = (z_real/2.0) + (z_imag/Dt);
        float sin_th = std::sin( th_d(j, del_th, th0) );
        float r_i1 = dist_d( 0.0, del_r, r0 );
        float r_i2 = dist_d( 0.5, del_r, r0 );
        float r_i3 = dist_d( 1.0, del_r, r0 );
        float val_1 = Dt/MU0/r_i2/sin_th/del_ph;
        float val_2 = Dt/MU0/r_i2/del_r;
        float alpha = 1.0 + ( val_2 * r_i1 * coeff_1 );

        Hth[idx_Hth] = (( 1.0 - val_2*r_i1*coeff_1 )*Hth[idx_Hth]
                                - val_1*( nEr[idx_Er1] - nEr[idx_Er2] ) + val_2*r_i3*nEph[idx_Eph])/alpha;
    }

    if( (i==0) && ((j >= 0)&&(j < Nth)) && ((k >= 1)&&(k < Nph+1)) ){
        int idx_Hph = idx_Hph_d(i,j,k,Nth,Nph);
        int idx_Er1 = idx_Er_d(i,j+1,k,Nth,Nph);
        int idx_Er2 = idx_Er_d(i,j,k,Nth,Nph);
        int idx_Eth = idx_Eth_d(i+1,j,k,Nth,Nph);

        float coeff_1 = (z_real/2.0) + (z_imag/Dt);
        float coeff_2 = (z_real/2.0) - (z_imag/Dt);
        float r_i1 = dist_d( 0.0, del_r, r0 );
        float r_i2 = dist_d( 0.5, del_r, r0 );
        float r_i3 = dist_d( 1.0, del_r, r0 );
        float val_3 = Dt/MU0/r_i2/del_r;
        float val_4 = Dt/MU0/r_i2/del_th;
        float beta = 1.0 + ( val_3 * r_i1 *coeff_1 );

        Hph[idx_Hph] = ((1.0 - val_3*r_i1*coeff_2)*Hph[idx_Hph]
                                - val_3*r_i3*nEth[idx_Eth] + val_4*(nEr[idx_Er1] - nEr[idx_Er2]))/beta;
    }

}