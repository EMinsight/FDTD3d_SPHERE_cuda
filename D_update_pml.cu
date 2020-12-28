#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fdtd3d.h"

__global__ void D_update_pml(
        int Nr, int Nth, int Nph, float *Dr, float *Dth, float *Dph,
        float *Hr, float *Hth, float *Hph, float *Dr_th1, float *Dr_th2, float *Dr_ph,
        float *Dth_ph, float *Dth_r, float *Dph_r, float *Dph_th, float *sig_th, float *sig_ph,
        PML *pml_Dr, PML *pml_Dth, PML *pml_Dph, float del_r, float del_th, float del_ph,
        float Dt, float r0, float th0 )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = ( blockDim.y * blockIdx.y + threadIdx.y ) / Nph;
    int k = ( blockDim.y * blockIdx.y + threadIdx.y ) % Nph;

    for( int area = 0; area < 2; area++ ){
    
        if( ((i>=0)&&(i<Nr)) && ((j>=pml_Dr[area].p_j1)&&(j<=pml_Dr[area].p_j2)) && ((k>=pml_Dr[area].p_k1)&&(k<=pml_Dr[area].p_k2)) ){
            int j_area = j - pml_Dr[area].p_j1;
            int k_area = k - pml_Dr[area].p_k1;
            int j_band = pml_Dr[area].p_j2 - pml_Dr[area].p_j1 + 1;
            int k_band = pml_Dr[area].p_k2 - pml_Dr[area].p_k1 + 1;
            int idx_pml_Dr = area*(Nr*j_band*k_band) + i*(j_band*k_band) + j_area*k_band + k_area;
            int idx_Dr = idx_Er_d( i, j, k, Nth, Nph );
            int idx_Hph1 = idx_Hph_d( i, j, k, Nth, Nph );
            int idx_Hph2 = idx_Hph_d( i, j-1, k, Nth, Nph );
            int idx_Hth1 = idx_Hth_d( i, j, k, Nth, Nph );
            int idx_Hth2 = idx_Hth_d( i, j, k-1, Nth, Nph );
                   
            float r_i2 = dist_d( float(i)+0.5, del_r, r0 );
            float th_j1 = th_d( float(j), del_th, th0 );

            Dr_th1[idx_pml_Dr] = C1_d( sig_th[j], 1.0/Dt )*Dr_th1[idx_pml_Dr]
                + C2_d( r_i2, sig_th[j], del_th, 1.0/Dt )*( Hph[idx_Hph1] - Hph[idx_Hph2] );
            Dr_th2[idx_pml_Dr] = Dr_th2[idx_pml_Dr]
                + C3_d( r_i2, th_j1, Dt )*( Hph[idx_Hph1] + Hph[idx_Hph2] );
            Dr_ph[idx_pml_Dr] = C1_d( sig_ph[k], 1.0/Dt )*Dr_ph[idx_pml_Dr]
                - C4_d( r_i2, th_j1, sig_ph[k], del_ph, 1.0/Dt )*( Hth[idx_Hth1] - Hth[idx_Hth2] );
            
            Dr[idx_Dr] = Dr_th1[idx_pml_Dr] + Dr_th2[idx_pml_Dr] + Dr_ph[idx_pml_Dr];
        
        }

        if( ((i>=1)&&(i<Nr)) && ((j>=pml_Dth[area].p_j1)&&(j<=pml_Dth[area].p_j2)) && ((k>=pml_Dth[area].p_k1)&&(k<=pml_Dth[area].p_k2)) ){
            int j_area = j - pml_Dth[area].p_j1;
            int k_area = k - pml_Dth[area].p_k1;
            int j_band = pml_Dth[area].p_j2 - pml_Dth[area].p_j1 + 1;
            int k_band = pml_Dth[area].p_k2 - pml_Dth[area].p_k1 + 1;
            int idx_pml_Dth = area*(Nr*j_band*k_band) + i*(j_band*k_band) + j_area*k_band + k_area;  
            int idx_Dth = idx_Eth_d(i, j, k, Nth, Nph);
            int idx_Hr1 = idx_Hr_d( i, j, k, Nth, Nph );
            int idx_Hr2 = idx_Hr_d( i, j, k-1, Nth, Nph );
            int idx_Hph1 = idx_Hph_d( i, j, k, Nth, Nph );
            int idx_Hph2 = idx_Hph_d( i-1, j, k, Nth, Nph );

            float r_i1 = dist_d( float(i)-0.5, del_r, r0 );
            float r_i2 = dist_d( float(i), del_r, r0 );
            float r_i3 = dist_d( float(i)+0.5, del_r, r0 );
            float th_j2 = th_d( float(j)+0.5, del_th, th0 );

            Dth_ph[idx_pml_Dth] = C1_d( sig_ph[k], 1.0/Dt )*Dth_ph[idx_pml_Dth]
                + C4_d( r_i2, th_j2, sig_ph[k], del_ph, 1.0/Dt )*( Hr[idx_Hr1] - Hr[idx_Hr2] );
            
            Dth_r[idx_pml_Dth] = Dth_r[idx_pml_Dth]
                - C5_d( r_i2, del_r, Dt )*( r_i3*Hph[idx_Hph1] - r_i1*Hph[idx_Hph2] );
            
            Dth[idx_Dth] = Dth_ph[idx_pml_Dth] + Dth_r[idx_pml_Dth];

        }

        if( ((i>=1)&&(i<Nr)) && ((j>=pml_Dph[area].p_j1)&&(j<=pml_Dph[area].p_j2)) && ((k>=pml_Dph[area].p_k1)&&(k<=pml_Dph[area].p_k2)) ){
            int j_area = j - pml_Dph[area].p_j1;
            int k_area = k - pml_Dph[area].p_k1;
            int j_band = pml_Dph[area].p_j2 - pml_Dph[area].p_j1 + 1;
            int k_band = pml_Dph[area].p_k2 - pml_Dph[area].p_k1 + 1;
            int idx_pml_Dph = area*(Nr*j_band*k_band) + i*(j_band*k_band) + j_area*k_band + k_area;
            int idx_Dph = idx_Eph_d( i, j, k, Nth, Nph );
            int idx_Hth1 = idx_Hth_d( i, j, k, Nth, Nph );
            int idx_Hth2 = idx_Hth_d( i-1, j, k, Nth, Nph );
            int idx_Hr1 = idx_Hr_d( i, j, k, Nth, Nph );
            int idx_Hr2 = idx_Hr_d( i, j-1, k, Nth, Nph );

            float r_i1 = dist_d( float(i)-0.5, del_r, r0 );
            float r_i2 = dist_d( float(i), del_r, r0 );
            float r_i3 = dist_d( float(i)+0.5, del_r, r0 );

            Dph_r[idx_pml_Dph] = Dph_r[idx_pml_Dph]
                + C5_d( r_i2, del_r, Dt )*( r_i3*Hth[idx_Hth1] - r_i1*Hth[idx_Hth2] );
            Dph_th[idx_pml_Dph] = C1_d( sig_th[j], 1.0/Dt )*Dph_th[idx_pml_Dph]
                - C6_d( r_i2, sig_th[j], del_th, 1.0/Dt )*( Hr[idx_Hr1] - Hr[idx_Hr2] );
            
            Dph[idx_Dph] = Dph_r[idx_pml_Dph] + Dph_th[idx_pml_Dph];
        
        }
         
    }

    for( int area = 2; area < 4; area++ ){
        if( ((i>=0)&&(i<Nr)) && ((j>=pml_Dr[area].p_j1)&&(j<=pml_Dr[area].p_j2)) && ((k>=pml_Dr[area].p_k1)&&(k<=pml_Dr[area].p_k2)) ){
            int j_area = j - pml_Dr[area].p_j1;
            int k_area = k - pml_Dr[area].p_k1;
            int j_band = pml_Dr[area].p_j2 - pml_Dr[area].p_j1 + 1;
            int k_band = pml_Dr[area].p_k2 - pml_Dr[area].p_k1 + 1;
            int buf = 2*( Nr*(pml_Dr[area/2].p_j2 - pml_Dr[area/2].p_j1 + 1)*(pml_Dr[area/2].p_k2 - pml_Dr[area/2].p_k1 + 1) );
            int idx_pml_Dr = buf + (area%2)*(Nr*j_band*k_band) + i*(j_band*k_band) + j_area*k_band + k_area;
            int idx_Dr = idx_Er_d( i, j, k, Nth, Nph );
            int idx_Hph1 = idx_Hph_d( i, j, k, Nth, Nph );
            int idx_Hph2 = idx_Hph_d( i, j-1, k, Nth, Nph );
            int idx_Hth1 = idx_Hth_d( i, j, k, Nth, Nph );
            int idx_Hth2 = idx_Hth_d( i, j, k-1, Nth, Nph );
                   
            float r_i2 = dist_d( float(i)+0.5, del_r, r0 );
            float th_j1 = th_d( float(j), del_th, th0 );

            Dr_th1[idx_pml_Dr] = C1_d( sig_th[j], 1.0/Dt )*Dr_th1[idx_pml_Dr]
                + C2_d( r_i2, sig_th[j], del_th, 1.0/Dt )*( Hph[idx_Hph1] - Hph[idx_Hph2] );
            Dr_th2[idx_pml_Dr] = Dr_th2[idx_pml_Dr]
                + C3_d( r_i2, th_j1, Dt )*( Hph[idx_Hph1] + Hph[idx_Hph2] );
            Dr_ph[idx_pml_Dr] = C1_d( sig_ph[k], 1.0/Dt )*Dr_ph[idx_pml_Dr]
                - C4_d( r_i2, th_j1, sig_ph[k], del_ph, 1.0/Dt )*( Hth[idx_Hth1] - Hth[idx_Hth2] );
            
            Dr[idx_Dr] = Dr_th1[idx_pml_Dr] + Dr_th2[idx_pml_Dr] + Dr_ph[idx_pml_Dr];
        }

        if( ((i>=1)&&(i<Nr)) && ((j>=pml_Dth[area].p_j1)&&(j<=pml_Dth[area].p_j2)) && ((k>=pml_Dth[area].p_k1)&&(k<=pml_Dth[area].p_k2)) ){
            int j_area = j - pml_Dth[area].p_j1;
            int k_area = k - pml_Dth[area].p_k1;
            int j_band = pml_Dth[area].p_j2 - pml_Dth[area].p_j1 + 1;
            int k_band = pml_Dth[area].p_k2 - pml_Dth[area].p_k1 + 1;
            int buf = 2*( Nr*(pml_Dth[area/2].p_j2 - pml_Dth[area/2].p_j1 + 1)*(pml_Dth[area/2].p_k2 - pml_Dth[area/2].p_k1 + 1) );
            int idx_pml_Dth = buf + (area%2)*(Nr*j_band*k_band) + i*(j_band*k_band) + j_area*k_band + k_area;

            int idx_Dth = idx_Eth_d(i, j, k, Nth, Nph);
            int idx_Hr1 = idx_Hr_d( i, j, k, Nth, Nph );
            int idx_Hr2 = idx_Hr_d( i, j, k-1, Nth, Nph );
            int idx_Hph1 = idx_Hph_d( i, j, k, Nth, Nph );
            int idx_Hph2 = idx_Hph_d( i-1, j, k, Nth, Nph );

            float r_i1 = dist_d( float(i)-0.5, del_r, r0 );
            float r_i2 = dist_d( float(i), del_r, r0 );
            float r_i3 = dist_d( float(i)+0.5, del_r, r0 );
            float th_j2 = th_d( float(j)+0.5, del_th, th0 );

            Dth_ph[idx_pml_Dth] = C1_d( sig_ph[k], 1.0/Dt )*Dth_ph[idx_pml_Dth]
                + C4_d( r_i2, th_j2, sig_ph[k], del_ph, 1.0/Dt )*( Hr[idx_Hr1] - Hr[idx_Hr2] );
            
            Dth_r[idx_pml_Dth] = Dth_r[idx_pml_Dth]
                - C5_d( r_i2, del_r, Dt )*( r_i3*Hph[idx_Hph1] - r_i1*Hph[idx_Hph2] );
            
            Dth[idx_Dth] = Dth_ph[idx_pml_Dth] + Dth_r[idx_pml_Dth];
        }

        if( ((i>=1)&&(i<Nr)) && ((j>=pml_Dph[area].p_j1)&&(j<=pml_Dph[area].p_j2)) && ((k>=pml_Dph[area].p_k1)&&(k<=pml_Dph[area].p_k2)) ){
            int j_area = j - pml_Dph[area].p_j1;
            int k_area = k - pml_Dph[area].p_k1;
            int j_band = pml_Dph[area].p_j2 - pml_Dph[area].p_j1 + 1;
            int k_band = pml_Dph[area].p_k2 - pml_Dph[area].p_k1 + 1;
            int buf = 2*( Nr*(pml_Dph[area/2].p_j2 - pml_Dph[area/2].p_j1 + 1)*(pml_Dph[area/2].p_k2 - pml_Dph[area/2].p_k1 + 1) );
            int idx_pml_Dph = buf + (area%2)*(Nr*j_band*k_band) + i*(j_band*k_band) + j_area*k_band + k_area;

            int idx_Dph = idx_Eph_d( i, j, k, Nth, Nph );
            int idx_Hth1 = idx_Hth_d( i, j, k, Nth, Nph );
            int idx_Hth2 = idx_Hth_d( i-1, j, k, Nth, Nph );
            int idx_Hr1 = idx_Hr_d( i, j, k, Nth, Nph );
            int idx_Hr2 = idx_Hr_d( i, j-1, k, Nth, Nph );

            float r_i1 = dist_d( float(i)-0.5, del_r, r0 );
            float r_i2 = dist_d( float(i), del_r, r0 );
            float r_i3 = dist_d( float(i)+0.5, del_r, r0 );

            Dph_r[idx_pml_Dph] = Dph_r[idx_pml_Dph]
                + C5_d( r_i2, del_r, Dt )*( r_i3*Hth[idx_Hth1] - r_i1*Hth[idx_Hth2] );
            Dph_th[idx_pml_Dph] = C1_d( sig_th[j], 1.0/Dt )*Dph_th[idx_pml_Dph]
                - C6_d( r_i2, sig_th[j], del_th, 1.0/Dt )*( Hr[idx_Hr1] - Hr[idx_Hr2] );
            
            Dph[idx_Dph] = Dph_r[idx_pml_Dph] + Dph_th[idx_pml_Dph];

        }

    }
}