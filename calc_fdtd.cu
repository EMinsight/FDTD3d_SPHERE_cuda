#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <complex>
#include <stdio.h>
#include <string>
#include <fstream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include "fdtd3d.h"
#include "gcc.h"

const float PI { 3.14159265258979 };
const float C0 { 3.0e8 };
const float MU0 { 4.0*PI*1.0e-7 };
const float EPS0 { 1.0/MU0/C0/C0 };
const float EPSR { 10.0 };
const float R0 { 6370e3 };
const float THETA0 { float(PI*0.5 - std::atan(50e3/R0)) };
const float E_Q {1.6e-19};
const float E_M {9.11e-31};

const int Nr{100};
const int Nth{100};
const int Nph{1000};

constexpr float R_r{100.0e3};

const float delta_r{ R_r/float(Nr) };
const float delta_th{ 1.0e3/float(R0) };
const float delta_ph{ 1.0e3/float(R0) };
const float Dt { float( 0.99/C0/std::sqrt(1.0/delta_r/delta_r
 + 1.0/R0/R0/delta_th/delta_th
 + 1.0/R0/R0/std::sin(THETA0)/std::sin(THETA0)/delta_ph/delta_ph) ) };
const float inv_Dt = float(1.0/Dt);
const float sigma_t = float(7.0*Dt);
const float t0 = float(6.0*sigma_t);

const float Z0 = std::sqrt(MU0/EPS0);
const float SIGMA_PEC { 1.0e7 };
const float SIGMA_SEA { 1.0 };
const float SIGMA_WET_GROUND { 1.0e-2 };
const float SIGMA_DRY_GROUND { 1.0e-3 };
const float SIGMA_VERY_DRY_GROUND { 1.0e-4 };
const float SIGMA_FRESH_WATER_ICE { 1.0E-5 };

// center point //
/*const int i_0{ Nr/2 };
const int j_0{ Nth/2 };
const int k_0{ Nph/2 };*/

// Source point //
const int i_s{1};
const int j_s{50};
const int k_s{100};

// Receive Point //
const int i_r{1};
const int j_r{50};
const int k_r{ Nph - 50 };

// PML info //
const int L{10};
const float M{3.5};
const float R{1.0e-6};
const float sig_th_max = -(M + 1.0)*C0*std::log(R)/2.0/float(L)/delta_th/R0;
const float sig_ph_max = -(M + 1.0)*C0*std::log(R)/2.0/float(L)/delta_ph/R0;

// Ionosphere info //
constexpr float Alt_lower_ionosphere{ 60.0e3 };
const int ion_L = int( (R_r - Alt_lower_ionosphere)/delta_r );
const float freq{ 22.2e3 };
const float omega = 2.0*M_PI*freq;

// Geomagnetic info //
const float B_abs{4.6468e-5};
const float Dec{-7.0*M_PI/180.0};
const float Inc{49.0*M_PI/180.0};
const float Azim{61.0*M_PI/180.0};

void calc_fdtd(void)
{
    float *Hr = array_ini( (Nr+1)*Nth*Nph, 0.0 );
    float *Hth = array_ini( Nr*(Nth+1)*Nph, 0.0 );
    float *Hph = array_ini( Nr*Nth*(Nph+1), 0.0 );

    float *nEr = array_ini( Nr*(Nth+1)*(Nph+1), 0.0 );
    float *nEth = array_ini( (Nr+1)*Nth*(Nph+1), 0.0 );
    float *nEph = array_ini( (Nr+1)*(Nth+1)*Nph, 0.0 );

    float *oEr = array_ini( Nr*(Nth+1)*(Nph+1), 0.0 );
    float *oEth = array_ini( (Nr+1)*Nth*(Nph+1), 0.0 );
    float *oEph = array_ini( (Nr+1)*(Nth+1)*Nph, 0.0 );

    float *nDr = array_ini( Nr*(Nth+1)*(Nph+1), 0.0);
    float *nDth = array_ini( (Nr+1)*Nth*(Nph+1), 0.0 );
    float *nDph = array_ini( (Nr+1)*(Nth+1)*Nph, 0.0 );

    float *oDr = array_ini( Nr*(Nth+1)*(Nph+1), 0.0 );
    float *oDth = array_ini( (Nr+1)*Nth*(Nph+1), 0.0 );
    float *oDph = array_ini( (Nr+1)*(Nth+1)*Nph, 0.0 );

    PML *pml_Dr, *pml_Dth, *pml_Dph;
    pml_Dr = new PML[4];
    pml_Dth = new PML[4];
    pml_Dph = new PML[4];

    PML *pml_Hr, *pml_Hth, *pml_Hph;
    pml_Hr = new PML[4];
    pml_Hth = new PML[4];
    pml_Hph = new PML[4];

    PML_idx_initialize(
        pml_Dr, pml_Dth, pml_Dph,
        pml_Hr, pml_Hth, pml_Hph );

    float *sig_th, *sig_ph, *sig_th_h, *sig_ph_h;
    sig_th = array_ini( Nth+1, 0.0 );
    sig_ph = array_ini( Nph+1, 0.0 );
    sig_th_h = array_ini( Nth+1, 0.0 );
    sig_ph_h = array_ini( Nph+1, 0.0 );

    sigma_calc(
        sig_th, sig_ph, sig_th_h, sig_ph_h );

    /*  PML field allocate  */
    /*  D in PML area allocate */
    int Dr_elem = 2*Nr*( (pml_Dr[0].p_j2 - pml_Dr[0].p_j1 + 1)*(pml_Dr[0].p_k2 - pml_Dr[0].p_k1 + 1) + 
    (pml_Dr[2].p_j2 - pml_Dr[2].p_j1 + 1)*(pml_Dr[2].p_k2 - pml_Dr[2].p_k1 + 1) );
    float *Dr_th1 = array_ini( Dr_elem, 0.0 );
    float *Dr_th2 = array_ini( Dr_elem, 0.0 );
    float *Dr_ph = array_ini( Dr_elem, 0.0 );
    float *Dr_th1_d, *Dr_th2_d, *Dr_ph_d;
    cudaMalloc( (void**)&Dr_th1_d, sizeof(float)*Dr_elem );
    cudaMalloc( (void**)&Dr_th2_d, sizeof(float)*Dr_elem );
    cudaMalloc( (void**)&Dr_ph_d, sizeof(float)*Dr_elem );
    cudaMemcpy( Dr_th1_d, Dr_th1, sizeof(float)*Dr_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Dr_th2_d, Dr_th2, sizeof(float)*Dr_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Dr_ph_d, Dr_ph, sizeof(float)*Dr_elem, cudaMemcpyHostToDevice );

    int Dth_elem = 2*Nr*( (pml_Dth[0].p_j2 - pml_Dth[0].p_j1 + 1)*(pml_Dth[0].p_k2 - pml_Dth[0].p_k1 + 1) + 
        (pml_Dth[2].p_j2 - pml_Dth[2].p_j1 + 1)*(pml_Dth[2].p_k2 - pml_Dth[2].p_k1 + 1) );
    float *Dth_ph = array_ini( Dth_elem, 0.0 );
    float *Dth_r = array_ini( Dth_elem, 0.0 );
    float *Dth_ph_d, *Dth_r_d;
    cudaMalloc( (void**)&Dth_ph_d, sizeof(float)*Dth_elem );
    cudaMalloc( (void**)&Dth_r_d, sizeof(float)*Dth_elem );
    cudaMemcpy( Dth_ph_d, Dth_ph, sizeof(float)*Dth_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Dth_r_d, Dth_r, sizeof(float)*Dth_elem, cudaMemcpyHostToDevice );

    int Dph_elem =  2*Nr*( (pml_Dph[0].p_j2 - pml_Dph[0].p_j1 + 1)*(pml_Dph[0].p_k2 - pml_Dph[0].p_k1 + 1) + 
        (pml_Dph[2].p_j2 - pml_Dph[2].p_j1 + 1)*(pml_Dph[2].p_k2 - pml_Dph[2].p_k1 + 1) );
    float *Dph_r = array_ini( Dph_elem, 0.0 );
    float *Dph_th = array_ini( Dph_elem, 0.0 );
    float *Dph_r_d, *Dph_th_d;
    cudaMalloc( (void**)&Dph_r_d, sizeof(float)*Dph_elem );
    cudaMalloc( (void**)&Dph_th_d, sizeof(float)*Dph_elem );
    cudaMemcpy( Dph_r_d, Dph_r, sizeof(float)*Dph_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Dph_th_d, Dph_th, sizeof(float)*Dph_elem, cudaMemcpyHostToDevice );

    /*  H in PML area allocate  */
    int Hr_elem = 2*(Nr+1)*( (pml_Hr[0].p_j2 - pml_Hr[0].p_j1 + 1)*(pml_Hr[0].p_k2 - pml_Hr[0].p_k1 + 1) + 
        (pml_Hr[2].p_j2 - pml_Hr[2].p_j1 + 1)*(pml_Hr[2].p_k2 - pml_Hr[2].p_k1 + 1) );
    float *Hr_th1 = array_ini( Hr_elem, 0.0 );
    float *Hr_th2 = array_ini( Hr_elem, 0.0 );
    float *Hr_ph = array_ini( Hr_elem, 0.0 );
    float *Hr_th1_d, *Hr_th2_d, *Hr_ph_d;
    cudaMalloc( (void**)&Hr_th1_d, sizeof(float)*Hr_elem );
    cudaMalloc( (void**)&Hr_th2_d, sizeof(float)*Hr_elem );
    cudaMalloc( (void**)&Hr_ph_d, sizeof(float)*Hr_elem );
    cudaMemcpy( Hr_th1_d, Hr_th1, sizeof(float)*Hr_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Hr_th2_d, Hr_th2, sizeof(float)*Hr_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Hr_ph_d, Hr_ph, sizeof(float)*Hr_elem, cudaMemcpyHostToDevice );

    int Hth_elem = 2*Nr*( (pml_Hth[0].p_j2 - pml_Hth[0].p_j1 + 1)*(pml_Hth[0].p_k2 - pml_Hth[0].p_k1 + 1) + 
        (pml_Hth[2].p_j2 - pml_Hth[2].p_j1 + 1)*(pml_Hth[2].p_k2 - pml_Hth[2].p_k1 + 1) );
    float *Hth_ph = array_ini( Hth_elem, 0.0 );
    float *Hth_r = array_ini( Hth_elem, 0.0 );
    float *Hth_ph_d, *Hth_r_d;
    cudaMalloc( (void**)&Hth_ph_d, sizeof(float)*Hth_elem );
    cudaMalloc( (void**)&Hth_r_d, sizeof(float)*Hth_elem );
    cudaMemcpy( Hth_ph_d, Hth_ph, sizeof(float)*Hth_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Hth_r_d, Hth_r, sizeof(float)*Hth_elem, cudaMemcpyHostToDevice );

    int Hph_elem = 2*Nr*( (pml_Hph[0].p_j2 - pml_Hph[0].p_j1 + 1)*(pml_Hph[0].p_k2 - pml_Hph[0].p_k1 + 1) + 
        (pml_Hph[2].p_j2 - pml_Hph[2].p_j1 + 1)*(pml_Hph[2].p_k2 - pml_Hph[2].p_k1 + 1) );
    float *Hph_r = array_ini( Hph_elem, 0.0 );
    float *Hph_th = array_ini( Hph_elem, 0.0 );
    float *Hph_r_d, *Hph_th_d;
    cudaMalloc( (void**)&Hph_r_d, sizeof(float)*Hph_elem );
    cudaMalloc( (void**)&Hph_th_d, sizeof(float)*Hph_elem );
    cudaMemcpy( Hph_r_d, Hph_r, sizeof(float)*Hph_elem, cudaMemcpyHostToDevice );
    cudaMemcpy( Hph_th_d, Hph_th, sizeof(float)*Hph_elem, cudaMemcpyHostToDevice );

    // Allocate Ne, nyu //
    double *Nh = new double[ion_L];
    double *ny = new double[ion_L];
    double *Re = new double[ion_L];
    for( int i = 0; i < ion_L; i++ ){
        Nh[i] = 0.0;
        ny[i] = 0.0;
        Re[i] = 0.0;
    }

    Ne_allocate( Nh, Re );
    ny_allocate( ny, Re );
    
    float *Cmat = array_ini( ion_L*Nth*Nph*3*3, 0.0 );
    float *Fmat = array_ini( ion_L*Nth*Nph*3*3, 0.0 );
    
    set_matrix( Cmat, Fmat, Nh, ny );

    for( int i = 0; i < ion_L; i++ ){
        for( int j = 0; j < Nth; j++ ){
            for( int k = 0; k < Nph; k++ ){
                for( int m = 0; m < 3; m++ ){
                    for( int n = 0; n < 3; n++ ){
                        std::cout << Fmat[idx_mat(i, j, k, m, n)] << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    std::exit(0);

    std::complex <float> Z;
    float Z_real( 0.0 ), Z_imag( 0.0 );

    Z = surface_impe();

    Z_real = Z.real();
    Z_imag = Z.imag()/omega;

    // device allocate //
    float *Hr_d, *Hth_d, *Hph_d;
    cudaMalloc( (void**)&Hr_d, sizeof(float)*(Nr+1)*Nth*Nph );
    cudaMalloc( (void**)&Hth_d, sizeof(float)*Nr*(Nth+1)*Nph );
    cudaMalloc( (void**)&Hph_d, sizeof(float)*Nr*Nth*(Nph+1) );

    float *nEr_d, *nEth_d, *nEph_d;
    cudaMalloc( (void**)&nEr_d, sizeof(float)*Nr*(Nth+1)*(Nph+1) );
    cudaMalloc( (void**)&nEth_d, sizeof(float)*(Nr+1)*Nth*(Nph+1) );
    cudaMalloc( (void**)&nEph_d, sizeof(float)*(Nr+1)*(Nth+1)*Nph );

    float *oEr_d, *oEth_d, *oEph_d;
    cudaMalloc( (void**)&oEr_d, sizeof(float)*Nr*(Nth+1)*(Nph+1) );
    cudaMalloc( (void**)&oEth_d, sizeof(float)*(Nr+1)*Nth*(Nph+1) );
    cudaMalloc( (void**)&oEph_d, sizeof(float)*(Nr+1)*(Nth+1)*Nph );

    float *nDr_d, *nDth_d, *nDph_d;
    cudaMalloc( (void**)&nDr_d, sizeof(float)*Nr*(Nth+1)*(Nph+1) );
    cudaMalloc( (void**)&nDth_d, sizeof(float)*(Nr+1)*Nth*(Nph+1) );
    cudaMalloc( (void**)&nDph_d, sizeof(float)*(Nr+1)*(Nth+1)*Nph );

    float *oDr_d, *oDth_d, *oDph_d;
    cudaMalloc( (void**)&oDr_d, sizeof(float)*Nr*(Nth+1)*(Nph+1) );
    cudaMalloc( (void**)&oDth_d, sizeof(float)*(Nr+1)*Nth*(Nph+1) );
    cudaMalloc( (void**)&oDph_d, sizeof(float)*(Nr+1)*(Nth+1)*Nph );

    PML *pml_Dr_d, *pml_Dth_d, *pml_Dph_d;
    cudaMalloc( (void**)&pml_Dr_d, sizeof(PML)*4 );
    cudaMalloc( (void**)&pml_Dth_d, sizeof(PML)*4 );
    cudaMalloc( (void**)&pml_Dph_d, sizeof(PML)*4 );

    PML *pml_Hr_d, *pml_Hth_d, *pml_Hph_d;
    cudaMalloc( (void**)&pml_Hr_d, sizeof(PML)*4 );
    cudaMalloc( (void**)&pml_Hth_d, sizeof(PML)*4 );
    cudaMalloc( (void**)&pml_Hph_d, sizeof(PML)*4 );

    float *sig_th_d, *sig_ph_d, *sig_th_h_d, *sig_ph_h_d;
    cudaMalloc( (void**)&sig_th_d, sizeof(float)*(Nth+1) );
    cudaMalloc( (void**)&sig_ph_d, sizeof(float)*(Nph+1) );
    cudaMalloc( (void**)&sig_th_h_d, sizeof(float)*(Nth+1) );
    cudaMalloc( (void**)&sig_ph_h_d, sizeof(float)*(Nph+1) );
    
    float *Cmat_d, *Fmat_d;
    cudaMalloc( (void**)&Cmat_d, sizeof(float)*ion_L*Nth*Nph*3*3 );
    cudaMalloc( (void**)&Fmat_d, sizeof(float)*ion_L*Nth*Nph*3*3 );

    // E, D, H memory copy(H to D) //
    cudaMemcpy( Hr_d, Hr, sizeof(float)*(Nr+1)*Nth*Nph, cudaMemcpyHostToDevice );
    cudaMemcpy( Hth_d, Hth, sizeof(float)*Nr*(Nth+1)*Nph, cudaMemcpyHostToDevice );
    cudaMemcpy( Hph_d, Hph, sizeof(float)*Nr*Nth*(Nph+1), cudaMemcpyHostToDevice );

    cudaMemcpy( nEr_d, nEr, sizeof(float)*Nr*(Nth+1)*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( nEth_d, nEth, sizeof(float)*(Nr+1)*Nth*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( nEph_d, nEph, sizeof(float)*(Nr+1)*(Nth+1)*Nph, cudaMemcpyHostToDevice );

    cudaMemcpy( oEr_d, oEr, sizeof(float)*Nr*(Nth+1)*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( oEth_d, oEth, sizeof(float)*(Nr+1)*Nth*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( oEph_d, oEph, sizeof(float)*(Nr+1)*(Nth+1)*Nph, cudaMemcpyHostToDevice );

    cudaMemcpy( nDr_d, nDr, sizeof(float)*Nr*(Nth+1)*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( nDth_d, nDth, sizeof(float)*(Nr+1)*Nth*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( nDph_d, nDph, sizeof(float)*(Nr+1)*(Nth+1)*Nph, cudaMemcpyHostToDevice );

    cudaMemcpy( oDr_d, oDr, sizeof(float)*Nr*(Nth+1)*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( oDth_d, oDth, sizeof(float)*(Nr+1)*Nth*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( oDph_d, oDph, sizeof(float)*(Nr+1)*(Nth+1)*Nph, cudaMemcpyHostToDevice );

    cudaMemcpy( pml_Dr_d, pml_Dr, sizeof(PML)*4, cudaMemcpyHostToDevice );
    cudaMemcpy( pml_Dth_d, pml_Dth, sizeof(PML)*4, cudaMemcpyHostToDevice );
    cudaMemcpy( pml_Dph_d, pml_Dph, sizeof(PML)*4, cudaMemcpyHostToDevice );

    cudaMemcpy( pml_Hr_d, pml_Hr, sizeof(PML)*4, cudaMemcpyHostToDevice );
    cudaMemcpy( pml_Hth_d, pml_Hth, sizeof(PML)*4, cudaMemcpyHostToDevice );
    cudaMemcpy( pml_Hph_d, pml_Hph, sizeof(PML)*4, cudaMemcpyHostToDevice );

    // sigma th, ph memory copy (H to D) //
    cudaMemcpy( sig_th_d, sig_th, sizeof(float)*(Nth+1), cudaMemcpyHostToDevice );
    cudaMemcpy( sig_ph_d, sig_ph, sizeof(float)*(Nph+1), cudaMemcpyHostToDevice );
    cudaMemcpy( sig_th_h_d, sig_th_h, sizeof(float)*(Nth+1), cudaMemcpyHostToDevice );
    cudaMemcpy( sig_ph_h_d, sig_ph_h, sizeof(float)*(Nph+1), cudaMemcpyHostToDevice );

    // Cmatrix and Fmatrix memory copy //
    cudaMemcpy( Cmat_d, Cmat, sizeof(float)*ion_L*Nth*Nph*3*3, cudaMemcpyHostToDevice );
    cudaMemcpy( Fmat_d, Fmat, sizeof(float)*ion_L*Nth*Nph*3*3, cudaMemcpyHostToDevice );
    
    // define grid, block size //
    int block_x = 16;
    int block_y = 16;
    dim3 Db( block_x, block_y, 1 );
    int grid_r = 128;
    int grid_th = 128;
    int grid_ph = 1024;
    dim3 Dg( grid_r/block_x, grid_th*grid_ph/block_y, 1 );
    
    int time_step = 1700;

    std::chrono::system_clock::time_point start
     = std::chrono::system_clock::now();
    
    for( int n = 0; n < time_step; n++ ){

        float t = ( float(n)- 0.5 )*Dt;
        // Add J //
        add_J <<<Dg, Db>>> ( delta_r, delta_th, delta_ph, oEth_d, 
                    i_s, j_s, k_s, t, t0, sigma_t, R0, Nr, Nth, Nph );

        //cudaMemcpy( oEth, oEth_d, sizeof(float)*(Nr+1)*Nth*(Nph+1), cudaMemcpyDeviceToHost );
        //std::cout << " t : " << t << "  E_th :" << oEth[ idx_Eth(i_s, j_s, k_s ) ] << "\n";

        // D update //
        D_update <<<Dg, Db>>> ( Nr, Nth, Nph, nDr_d, nDth_d, nDph_d,
                    oDr_d, oDth_d, oDph_d, Hr_d, Hth_d, Hph_d, 
                    delta_r, delta_th, delta_ph, Dt, THETA0, R0, L );
        
        // D update in PML area //
        D_update_pml <<<Dg, Db>>> ( Nr, Nth, Nph, nDr_d, nDth_d, nDph_d,
                    Hr_d, Hth_d, Hph_d, Dr_th1_d, Dr_th2_d, Dr_ph_d,
                    Dth_ph_d, Dth_r_d, Dph_r_d, Dph_th_d, sig_th_d, sig_ph_d,
                    pml_Dr_d, pml_Dth_d, pml_Dph_d, delta_r, delta_th, delta_ph,
                    Dt, R0, THETA0 );

        // E update //
        E_update <<<Dg, Db>>> ( Nr, Nth, Nph, nEr_d, nEth_d, nEph_d, oEr_d, oEth_d, oEph_d,
                    nDr_d, nDth_d, nDph_d, oDr_d, oDth_d, oDph_d, Cmat_d, Fmat_d, 
                    ion_L, EPS0 );

        cudaDeviceSynchronize();

        H_update <<<Dg, Db>>> ( Nr, Nth, Nph, nEr_d, nEth_d, nEph_d, 
                    Hr_d, Hth_d, Hph_d, delta_r, delta_th, delta_ph, 
                    Dt, THETA0, R0, MU0, L );

        surface_H_update <<<Dg, Db>>> ( Nr, Nth, Nph, nEr_d, nEth_d, nEph_d, Hth_d, Hph_d,
            Z_real, Z_imag, delta_r, delta_th, delta_ph, Dt, R0, THETA0, MU0 );
        
        H_update_pml <<<Dg, Db>>> ( Nr, Nth, Nph, nEr_d, nEth_d, nEph_d,
                    Hr_d, Hth_d, Hph_d, Hr_th1_d, Hr_th2_d, Hr_ph_d,
                    Hth_ph_d, Hth_r_d, Hph_r_d, Hph_th_d, sig_th_h_d, sig_ph_h_d,
                    pml_Hr_d, pml_Hth_d, pml_Hph_d, delta_r, delta_th, delta_ph, Dt,
                    R0, THETA0, MU0 );
        
        cudaDeviceSynchronize();

        E_old_to_new <<<Dg, Db>>> (
            Nr, Nth, Nph, nEr_d, nEth_d, nEph_d, oEr_d, oEth_d, oEph_d
        );

        std::string fn = "./result/Er_" + std::to_string(n) + ".dat";
        std::ofstream ofs( fn.c_str() ); 
        cudaMemcpy( nEr, nEr_d, sizeof(float)*Nr*(Nth+1)*(Nph+1) , cudaMemcpyDeviceToHost );
        for( int i = 0; i < Nr; i++ ){
            for( int k = 0; k < Nph; k++ ){
                ofs << R0*ph(k)/1000.0 << " " << i*delta_r/1000.0 << " " << nEr[idx_Er(i, j_s, k)] << "\n";
            }
            ofs << "\n";
        }

        ofs.close();

    }
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    
    double total_time;
    total_time = std::chrono::duration_cast <std::chrono::milliseconds>
    (end - start).count();

    std::cout << "elapsed_time : " << total_time*1.e-3 << "\n";

    cudaMemcpy( nEr, nEr_d, sizeof(float)*Nr*(Nth+1)*(Nph+1), cudaMemcpyDeviceToHost);
    std::cout << "Source point : " << nEr[idx_Er(i_s, j_s, k_s)] << "\n";
    std::cout << "Receive point : " << nEr[idx_Er(i_r, j_r, k_r)] << "\n";

    cudaFree( nEr_d );
    cudaFree( nEth_d );
    cudaFree( nEph_d );
    cudaFree( oEr_d );
    cudaFree( oEth_d );
    cudaFree( oEph_d );
    cudaFree( Hr_d );
    cudaFree( Hth_d );
    cudaFree( Hph_d );
    cudaFree( nDr_d );
    cudaFree( nDth_d );
    cudaFree( nDph_d );
    cudaFree( oDr_d );
    cudaFree( oDth_d );
    cudaFree( oDph_d );
    cudaFree( pml_Dr_d );
    cudaFree( pml_Dth_d );
    cudaFree( pml_Dph_d );
    cudaFree( pml_Hr_d );
    cudaFree( pml_Hth_d );
    cudaFree( pml_Hph_d );
    cudaFree( Dr_th1_d );
    cudaFree( Dr_th2_d );
    cudaFree( Dr_ph_d );
    cudaFree( Dth_ph_d );
    cudaFree( Dth_r_d );
    cudaFree( Dph_r_d );
    cudaFree( Dph_th_d );
    cudaFree( Hr_th1_d );
    cudaFree( Hr_th2_d );
    cudaFree( Hr_ph_d );
    cudaFree( Hth_ph_d );
    cudaFree( Hth_r_d );
    cudaFree( Hph_r_d );
    cudaFree( Hph_th_d );
    cudaFree( sig_th_d );
    cudaFree( sig_ph_d );
    cudaFree( sig_th_h_d );
    cudaFree( sig_ph_h_d );
    cudaFree( Cmat_d );
    cudaFree( Fmat_d );

    delete[] nEr;
    delete[] nEth;
    delete[] nEph;
    delete[] oEr;
    delete[] oEth;
    delete[] oEph;
    delete[] Hr;
    delete[] Hth;
    delete[] Hph;
    delete[] nDr;
    delete[] nDth;
    delete[] nDph;
    delete[] oDr;
    delete[] oDth;
    delete[] oDph;
    delete[] pml_Dr;
    delete[] pml_Dth;
    delete[] pml_Dph;
    delete[] pml_Hr;
    delete[] pml_Hth;
    delete[] pml_Hph;
    delete[] Dr_th1;
    delete[] Dr_th2;
    delete[] Dr_ph;
    delete[] Dth_ph;
    delete[] Dth_r;
    delete[] Dph_r;
    delete[] Dph_th;
    delete[] Hr_th1;
    delete[] Hr_th2;
    delete[] Hr_ph;
    delete[] Hth_ph;
    delete[] Hth_r;
    delete[] Hph_r;
    delete[] Hph_th;
    delete[] sig_th;
    delete[] sig_ph;
    delete[] sig_th_h;
    delete[] sig_ph_h;
    delete[] Cmat;
    delete[] Fmat;

}