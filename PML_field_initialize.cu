#include <iostream>
#include <stdio.h>
#include "fdtd3d.h"

void PML_field_initialize(
    float* Dr_th1, float* Dr_th2, float* Dr_ph,
    float* Dth_ph, float* Dth_r,
    float* Dph_r, float* Dph_th,
    float* Hr_th1, float* Hr_th2, float* Hr_ph,
    float* Hth_ph, float* Hth_r,
    float* Hph_r, float* Hph_th,
    float* Dr_th1_d, float* Dr_th2_d, float* Dr_ph_d,
    float* Dth_ph_d, float* Dth_r_d,
    float* Dph_r_d, float* Dph_th_d,
    float* Hr_th1_d, float* Hr_th2_d, float *Hr_ph_d,
    float* Hth_ph_d, float* Hth_r_d,
    float* Hph_r_d, float* Hph_th_d,
    PML* pml_Dr, PML* pml_Dth, PML* pml_Dph,
    PML* pml_Hr, PML* pml_Hth, PML* pml_Hph )
{
    /*  D in PML area allocate  */
    int Dr_elem = 2*Nr*( (pml_Dr[0].p_j2 - pml_Dr[0].p_j1 + 1)*(pml_Dr[0].p_k2 - pml_Dr[0].p_k1 + 1) + 
                        (pml_Dr[2].p_j2 - pml_Dr[2].p_j1 + 1)*(pml_Dr[2].p_k2 - pml_Dr[2].p_k1 + 1) );
    Dr_th1 = array_ini( Dr_elem, 1.0 );
    Dr_th2 = array_ini( Dr_elem, 0.0 );
    Dr_ph = array_ini( Dr_elem, 0.0 );
    cudaMalloc( (void**)&Dr_th1_d, sizeof(float)*Dr_elem );
    cudaMalloc( (void**)&Dr_th2_d, sizeof(float)*Dr_elem );
    cudaMalloc( (void**)&Dr_ph_d, sizeof(float)*Dr_elem );
    cudaMemcpy( Dr_th1_d, Dr_th1, sizeof(float)*Dr_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Dr_th2_d, Dr_th2, sizeof(float)*Dr_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Dr_ph_d, Dr_ph, sizeof(float)*Dr_elem, cudaMemcpyHostToDevice );

    int Dth_elem = 2*Nr*( (pml_Dth[0].p_j2 - pml_Dth[0].p_j1 + 1)*(pml_Dth[0].p_k2 - pml_Dth[0].p_k1 + 1) + 
                        (pml_Dth[2].p_j2 - pml_Dth[2].p_j1 + 1)*(pml_Dth[2].p_k2 - pml_Dth[2].p_k1 + 1) );
    Dth_ph = array_ini( Dth_elem, 0.0 );
    Dth_r = array_ini( Dth_elem, 0.0 );
    cudaMalloc( (void**)&Dth_ph_d, sizeof(float)*Dth_elem );
    cudaMalloc( (void**)&Dth_r_d, sizeof(float)*Dth_elem );
    cudaMemcpy( Dth_ph_d, Dth_ph, sizeof(float)*Dth_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Dth_r_d, Dth_r, sizeof(float)*Dth_elem, cudaMemcpyHostToDevice );

    int Dph_elem =  2*Nr*( (pml_Dph[0].p_j2 - pml_Dph[0].p_j1 + 1)*(pml_Dph[0].p_k2 - pml_Dph[0].p_k1 + 1) + 
                        (pml_Dph[2].p_j2 - pml_Dph[2].p_j1 + 1)*(pml_Dph[2].p_k2 - pml_Dph[2].p_k1 + 1) );
    Dph_r = array_ini( Dph_elem, 0.0 );
    Dph_th = array_ini( Dph_elem, 0.0 );
    cudaMalloc( (void**)&Dph_r, sizeof(float)*Dph_elem );
    cudaMalloc( (void**)&Dph_th, sizeof(float)*Dph_elem );
    cudaMemcpy( Dph_r_d, Dph_r, sizeof(float)*Dph_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Dph_th_d, Dph_th, sizeof(float)*Dph_elem, cudaMemcpyHostToDevice );

    /*  H in PML area allocate  */
    int Hr_elem = 2*(Nr+1)*( (pml_Hr[0].p_j2 - pml_Hr[0].p_j1 + 1)*(pml_Hr[0].p_k2 - pml_Hr[0].p_k1 + 1) + 
                        (pml_Hr[2].p_j2 - pml_Hr[2].p_j1 + 1)*(pml_Hr[2].p_k2 - pml_Hr[2].p_k1 + 1) );
    Hr_th1 = array_ini( Hr_elem, 0.0 );
    Hr_th2 = array_ini( Hr_elem, 0.0 );
    Hr_ph = array_ini( Hr_elem, 0.0 );
    cudaMalloc( (void**)&Hr_th1_d, sizeof(float)*Hr_elem );
    cudaMalloc( (void**)&Hr_th2_d, sizeof(float)*Hr_elem );
    cudaMalloc( (void**)&Hr_ph_d, sizeof(float)*Hr_elem );
    cudaMemcpy( Hr_th1_d, Hr_th1, sizeof(float)*Hr_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Hr_th2_d, Hr_th2, sizeof(float)*Hr_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Hr_ph_d, Hr_ph, sizeof(float)*Hr_elem, cudaMemcpyHostToDevice );

    int Hth_elem = 2*Nr*( (pml_Hth[0].p_j2 - pml_Hth[0].p_j1 + 1)*(pml_Hth[0].p_k2 - pml_Hth[0].p_k1 + 1) + 
                        (pml_Hth[2].p_j2 - pml_Hth[2].p_j1 + 1)*(pml_Hth[2].p_k2 - pml_Hth[2].p_k1 + 1) );
    Hth_ph = array_ini( Hth_elem, 0.0 );
    Hth_r = array_ini( Hth_elem, 0.0 );
    cudaMalloc( (void**)&Hth_ph_d, sizeof(float)*Hth_elem );
    cudaMalloc( (void**)&Hth_r_d, sizeof(float)*Hth_elem );
    cudaMemcpy( Hth_ph_d, Hth_ph, sizeof(float)*Hth_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Hth_r_d, Hth_r, sizeof(float)*Hth_elem, cudaMemcpyHostToDevice );

    int Hph_elem = 2*Nr*( (pml_Hph[0].p_j2 - pml_Hph[0].p_j1 + 1)*(pml_Hph[0].p_k2 - pml_Hph[0].p_k1 + 1) + 
                        (pml_Hph[2].p_j2 - pml_Hph[2].p_j1 + 1)*(pml_Hph[2].p_k2 - pml_Hph[2].p_k1 + 1) );
    Hph_r = array_ini( Hph_elem, 0.0 );
    Hph_th = array_ini( Hph_elem, 0.0 );
    cudaMalloc( (void**)&Hph_r, sizeof(float)*Hph_elem );
    cudaMalloc( (void**)&Hph_th, sizeof(float)*Hph_elem );
    cudaMemcpy( Hph_r_d, Hph_r, sizeof(float)*Hph_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Hph_th_d, Hph_th, sizeof(float)*Hph_elem, cudaMemcpyHostToDevice );
    
}