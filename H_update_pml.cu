#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fdtd3d.h"

__global__ void H_update_pml(
    int Nr, int Nth, int Nph, float *Er, float *Eth, float *Eph,
    float *Hr, float *Hth, float *Hph, float *Hr_th1, float *Hr_th2, float*Hr_ph,
    float *Hth_ph, float *Hth_r, float *Hph_r, float *Hph_th, float *sig_th, float *sig_ph,
    PML *pml_Hr, PML *pml_Hth, PML *pml_Hph, float del_r, float del_th, float del_ph,
    float Dt, float r0, float th0, float mu0 )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
    int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

    for( int area = 0; area < 2; area++ ){

        if( ((i>=1)&&(i<Nr+1)) && ((j>=pml_Hr[area].p_j1)&&(j<=pml_Hr[area].p_j2)) && ((k>=pml_Hr[area].p_k1)&&(k<=pml_Hr[area].p_k2)) ){
            int j_area = j - pml_Hr[area].p_j1;
            int k_area = k - pml_Hr[area].p_k1;
            int j_band = pml_Hr[area].p_j2 - pml_Hr[area].p_j1 + 1;
            int k_band = pml_Hr[area].p_k2 - pml_Hr[area].p_k1 + 1;
            int idx_pml_Hr = area*((Nr+1)*j_band*k_band) + i*(j_band*k_band) + j_area*k_band + k_area;
            int idx_Hr = idx_Hr_d( i, j, k, Nth, Nph );
            int idx_Eph1 = idx_Eph_d( i, j+1, k, Nth, Nph );
            int idx_Eph2 = idx_Eph_d( i, j, k, Nth, Nph );
            int idx_Eth1 = idx_Eth_d( i, j, k+1, Nth, Nph );
            int idx_Eth2 = idx_Eth_d( i, j, k, Nth, Nph );

            float r_i1 = dist_d( float(i), del_r, r0 );
            float th_j2 = th_d( float(j)+0.5 , del_th, th0 );

            Hr_th1[idx_pml_Hr] = C1_d( sig_th[j], 1.0/Dt )*Hr_th1[idx_pml_Hr]
                - C2_d( r_i1, sig_th[j], del_th, 1.0/Dt )/mu0 * ( Eph[idx_Eph1] - Eph[idx_Eph2] );
            
            Hr_th2[idx_pml_Hr] = Hr_th2[idx_pml_Hr]
                - C3_d( r_i1, th_j2, Dt )/mu0 * ( Eph[idx_Eph1] + Eph[idx_Eph2] );
            
            Hr_ph[idx_pml_Hr] = C1_d( sig_ph[k], 1.0/Dt )*Hr_ph[idx_pml_Hr]
                + C4_d( r_i1, th_j2, sig_ph[k], del_ph, 1.0/Dt )/mu0 * ( Eth[idx_Eth1] - Eth[idx_Eth2] );
            
            Hr[idx_Hr] = Hr_th1[idx_pml_Hr]
                + Hr_th2[idx_pml_Hr] + Hr_ph[idx_pml_Hr];

        }

        if( ((i>=0)&&(i<Nr)) && ((j>=pml_Hth[area].p_j1)&&(j<=pml_Hth[area].p_j2)) && ((k>=pml_Hth[area].p_k1)&&(k<=pml_Hth[area].p_k2)) ){
            int j_area = j - pml_Hth[area].p_j1;
            int k_area = k - pml_Hth[area].p_k1;
            int j_band = pml_Hth[area].p_j2 - pml_Hth[area].p_j1 + 1;
            int k_band = pml_Hth[area].p_k2 - pml_Hth[area].p_k1 + 1;
            int idx_pml_Hth = area*( Nr*j_band*k_band ) + i*( j_band*k_band ) + j_area*k_band + k_area;
            int idx_Hth = idx_Hth_d( i, j, k, Nth, Nph );
            int idx_Er1 = idx_Er_d( i, j, k+1, Nth, Nph );
            int idx_Er2 = idx_Er_d( i, j, k, Nth, Nph );
            int idx_Eph1 = idx_Eph_d( i+1, j, k, Nth, Nph );
            int idx_Eph2 = idx_Eph_d( i, j, k, Nth, Nph );

            float r_i1 = dist_d( float(i), del_r, r0 );
            float r_i2 = dist_d( float(i)+0.5, del_r, r0 );
            float r_i3 = dist_d( float(i)+1.0, del_r, r0 );
            float th_j1 = th_d( float(j), del_th, th0 );

            Hth_ph[idx_pml_Hth] = C1_d( sig_ph[k], 1.0/Dt )*Hth_ph[idx_pml_Hth]
                - C4_d( r_i2, th_j1, sig_ph[k], del_ph, 1.0/Dt )/mu0 * ( Er[idx_Er1] - Er[idx_Er2] );
            
            Hth_r[idx_pml_Hth] = Hth_r[idx_pml_Hth]
                + C5_d( r_i2, del_r, Dt )/mu0 * ( r_i3*Eph[idx_Eph1] - r_i1*Eph[idx_Eph2] );
            
            Hth[idx_Hth] = Hth_ph[idx_pml_Hth] + Hth_r[idx_pml_Hth];

        }

        if( ((i>=0)&&(i<Nr)) && ((j>=pml_Hph[area].p_j1)&&(j<=pml_Hph[area].p_j2)) && ((k>=pml_Hph[area].p_k1)&&(k<=pml_Hph[area].p_k2)) ){
            int j_area = j - pml_Hph[area].p_j1;
            int k_area = k - pml_Hph[area].p_k1;
            int j_band = pml_Hph[area].p_j2 - pml_Hph[area].p_j1 + 1;
            int k_band = pml_Hph[area].p_k2 - pml_Hph[area].p_k1 + 1;
            int idx_pml_Hph = area*( Nr*j_band*k_band ) + i*( j_band*k_band ) + j_area*k_band + k_area;
            int idx_Hph = idx_Hph_d( i, j, k, Nth, Nph );
            int idx_Eth1 = idx_Eth_d( i+1, j, k, Nth, Nph );
            int idx_Eth2 = idx_Eth_d( i, j, k, Nth, Nph );
            int idx_Er1 = idx_Er_d( i, j+1, k, Nth, Nph );
            int idx_Er2 = idx_Er_d( i, j, k, Nth, Nph );

            float r_i1 = dist_d( float(i), del_r, r0 );
            float r_i2 = dist_d( float(i)+0.5, del_r, r0 );
            float r_i3 = dist_d( float(i)+1.0, del_r, r0 );

            Hph_r[idx_pml_Hph] = Hph_r[idx_pml_Hph]
                - C5_d( r_i2, del_r, Dt )/mu0 * ( r_i3*Eth[idx_Eth1] - r_i1*Eth[idx_Eth2] );
            
            Hph_th[idx_pml_Hph] = C1_d( sig_th[j], 1.0/Dt )*Hph_th[idx_pml_Hph]
                + C6_d( r_i2, sig_th[j], del_th, 1.0/Dt )/mu0 * ( Er[idx_Er1] - Er[idx_Er2] );
            
            Hph[idx_Hph] = Hph_r[idx_pml_Hph] + Hph_th[idx_pml_Hph];

        }
    }

    for( int area = 2; area < 4; area++ ){

        if( ((i>=1)&&(i<Nr+1)) && ((j>=pml_Hr[area].p_j1)&&(j<=pml_Hr[area].p_j2)) && ((k>=pml_Hr[area].p_k1)&&(k<=pml_Hr[area].p_k2)) ){
            int j_area = j - pml_Hr[area].p_j1;
            int k_area = k - pml_Hr[area].p_k1;
            int j_band = pml_Hr[area].p_j2 - pml_Hr[area].p_j1 + 1;
            int k_band = pml_Hr[area].p_k2 - pml_Hr[area].p_k1 + 1;
            int buf = 2*( (Nr+1)*(pml_Hr[area/2].p_j2 - pml_Hr[area/2].p_j1 + 1)*(pml_Hr[area/2].p_k2 - pml_Hr[area/2].p_k1 + 1) );
            int idx_pml_Hr = buf + (area%2)*( (Nr+1)*j_band*k_band ) + i*(j_band*k_band) + j_area*k_band + k_area;
            int idx_Hr = idx_Hr_d( i, j, k, Nth, Nph );
            int idx_Eph1 = idx_Eph_d( i, j+1, k, Nth, Nph );
            int idx_Eph2 = idx_Eph_d( i, j, k, Nth, Nph );
            int idx_Eth1 = idx_Eth_d( i, j, k+1, Nth, Nph );
            int idx_Eth2 = idx_Eth_d( i, j, k, Nth, Nph );

            float r_i1 = dist_d( float(i), del_r, r0 );
            float th_j2 = th_d( float(j)+0.5 , del_th, th0 );

            Hr_th1[idx_pml_Hr] = C1_d( sig_th[j], 1.0/Dt )*Hr_th1[idx_pml_Hr]
                - C2_d( r_i1, sig_th[j], del_th, 1.0/Dt )/mu0 * ( Eph[idx_Eph1] - Eph[idx_Eph2] );
            
            Hr_th2[idx_pml_Hr] = Hr_th2[idx_pml_Hr]
                - C3_d( r_i1, th_j2, Dt )/mu0 * ( Eph[idx_Eph1] + Eph[idx_Eph2] );
            
            Hr_ph[idx_pml_Hr] = C1_d( sig_ph[k], 1.0/Dt )*Hr_ph[idx_pml_Hr]
                + C4_d( r_i1, th_j2, sig_ph[k], del_ph, 1.0/Dt )/mu0 * ( Eth[idx_Eth1] - Eth[idx_Eth2] );
            
            Hr[idx_Hr] = Hr_th1[idx_pml_Hr]
                + Hr_th2[idx_pml_Hr] + Hr_ph[idx_pml_Hr];

        }

        if( ((i>=0)&&(i<Nr)) && ((j>=pml_Hth[area].p_j1)&&(j<=pml_Hth[area].p_j2)) && ((k>=pml_Hth[area].p_k1)&&(k<=pml_Hth[area].p_k2)) ){
            int j_area = j - pml_Hth[area].p_j1;
            int k_area = k - pml_Hth[area].p_k1;
            int j_band = pml_Hth[area].p_j2 - pml_Hth[area].p_j1 + 1;
            int k_band = pml_Hth[area].p_k2 - pml_Hth[area].p_k1 + 1;
            int buf = 2*( Nr*(pml_Hth[area/2].p_j2 - pml_Hth[area/2].p_j1 + 1)*(pml_Hth[area/2].p_k2 - pml_Hth[area/2].p_k1 + 1) );
            int idx_pml_Hth = buf + (area%2)*( Nr*j_band*k_band ) + i*( j_band*k_band ) + j_area*k_band + k_area;
            int idx_Hth = idx_Hth_d( i, j, k, Nth, Nph );
            int idx_Er1 = idx_Er_d( i, j, k+1, Nth, Nph );
            int idx_Er2 = idx_Er_d( i, j, k, Nth, Nph );
            int idx_Eph1 = idx_Eph_d( i+1, j, k, Nth, Nph );
            int idx_Eph2 = idx_Eph_d( i, j, k, Nth, Nph );


            float r_i1 = dist_d( float(i), del_r, r0 );
            float r_i2 = dist_d( float(i)+0.5, del_r, r0 );
            float r_i3 = dist_d( float(i)+1.0, del_r, r0 );
            float th_j1 = th_d( float(j), del_th, th0 );

            Hth_ph[idx_pml_Hth] = C1_d( sig_ph[k], 1.0/Dt )*Hth_ph[idx_pml_Hth]
                - C4_d( r_i2, th_j1, sig_ph[k], del_ph, 1.0/Dt )/mu0 * ( Er[idx_Er1] - Er[idx_Er2] );
            
            Hth_r[idx_pml_Hth] = Hth_r[idx_pml_Hth]
                + C5_d( r_i2, del_r, Dt )/mu0 * ( r_i3*Eph[idx_Eph1] - r_i1*Eph[idx_Eph2] );
            
            Hth[idx_Hth] = Hth_ph[idx_pml_Hth] + Hth_r[idx_pml_Hth];

        }

        if( ((i>=0)&&(i<Nr)) && ((j>=pml_Hph[area].p_j1)&&(j<=pml_Hph[area].p_j2)) && ((k>=pml_Hph[area].p_k1)&&(k<=pml_Hph[area].p_k2)) ){
            int j_area = j - pml_Hph[area].p_j1;
            int k_area = k - pml_Hph[area].p_k1;
            int j_band = pml_Hph[area].p_j2 - pml_Hph[area].p_j1 + 1;
            int k_band = pml_Hph[area].p_k2 - pml_Hph[area].p_k1 + 1;
            int buf = 2*( Nr*(pml_Hph[area/2].p_j2 - pml_Hph[area/2].p_j1 + 1)*(pml_Hph[area/2].p_k2 - pml_Hph[area/2].p_k1 + 1) ); 
            int idx_pml_Hph = buf + (area%2)*( Nr*j_band*k_band ) + i*( j_band*k_band ) + j_area*k_band + k_area;
            int idx_Hph = idx_Hph_d( i, j, k, Nth, Nph );
            int idx_Eth1 = idx_Eth_d( i+1, j, k, Nth, Nph );
            int idx_Eth2 = idx_Eth_d( i, j, k, Nth, Nph );
            int idx_Er1 = idx_Er_d( i, j+1, k, Nth, Nph );
            int idx_Er2 = idx_Er_d( i, j, k, Nth, Nph );

            float r_i1 = dist_d( float(i), del_r, r0 );
            float r_i2 = dist_d( float(i)+0.5, del_r, r0 );
            float r_i3 = dist_d( float(i)+1.0, del_r, r0 );

            Hph_r[idx_pml_Hph] = Hph_r[idx_pml_Hph]
                - C5_d( r_i2, del_r, Dt )/mu0 * ( r_i3*Eth[idx_Eth1] - r_i1*Eth[idx_Eth2] );
            
            Hph_th[idx_pml_Hph] = C1_d( sig_th[j], 1.0/Dt )*Hph_th[idx_pml_Hph]
                + C6_d( r_i2, sig_th[j], del_th, 1.0/Dt )/mu0 * ( Er[idx_Er1] - Er[idx_Er2] );
            
            Hph[idx_Hph] = Hph_r[idx_pml_Hph] + Hph_th[idx_pml_Hph];

        }
    }

}