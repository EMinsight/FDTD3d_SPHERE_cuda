#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include "PML.h"

extern const float PI;
extern const float C0;
extern const float MU0;
extern const float EPS0;
extern const float R0;
extern const float THETA0;
extern const float E_Q;
extern const float E_M;

extern const int Nr;
extern const int Nth;
extern const int Nph;

extern const float delta_r;
extern const float delta_th;
extern const float delta_ph;
extern const float Dt;
extern const float inv_Dt;

// PML info //
extern const int L;
extern const float M;
extern const float R;
extern const float sig_th_max;
extern const float sig_ph_max;

// Iono info //
extern const float Alt_lower_ionosphere;
extern const int ion_L;
extern const float freq;
extern const float omega;

// Geomagnetic info //
extern const float B_abs;
extern const float Dec;
extern const float Inc;
extern const float Azim;

float* array_ini( int size, float ini );

void PML_idx_initialize(
    PML* idx_Dr, PML* idx_Dth, PML* idx_Dph,
    PML* idx_Hr, PML* idx_Hth, PML* idx_Hph
);

void PML_field_initialize(
    float*, float*, float*,
    float*, float*,
    float*, float*,
    float*, float*, float*,
    float*, float*,
    float*, float*,
    float*, float*, float*,
    float*, float*,
    float*, float*,
    float*, float*, float*,
    float*, float*,
    float*, float*,
    PML*, PML*, PML*,
    PML*, PML*, PML*
);

void sigma_calc(
    float *sigma_theta, float *sigma_phi,
    float *sigma_theta_half, float *sigma_phi_half
);

__global__ void add_J( 
    float del_r, float del_th, float del_ph, float *Eth,
    int i_s, int j_s, int k_s, float t, float t0, float sig, 
    float r0, int nr, int nth, int nph );

__global__ void D_update( 
    int nr, int nth, int nph, float *nDr, float *nDth, float *nDph,
    float *oDr, float *oDth, float *oDph, float *Hr, float *Hth, float *Hph,
    float del_r, float del_th, float del_ph, float dt, float theta0, float r0, int l );

__global__ void D_update_pml(
    int nr, int nth, int nph, float *nDr, float *nDth, float *nDph,
    float *Hr, float *Hth, float *Hph, float *Dr_th1, float *Dr_th2, float *Dr_ph,
    float *Dth_ph, float *Dth_r, float *Dph_r, float *Dph_th, float *sig_th, float *sig_ph,
    PML* pml_Dr, PML* pml_Dth, PML* pml_Dph, float del_r, float del_th, float del_ph, float dt,
    float r0, float theta0 );

__global__ void E_update( 
    int nr, int nth, int nph, float *nEr, float *nEth, float *nEph, float *oEr, float *oEth, float *oEph,
    float *nDr, float *nDth, float *nDph, float *oDr, float *oDth, float *oDph, float *Cmat, float *Fmat, 
    int ion_l, float eps) ;

__global__ void H_update(
    int nr, int nth, int nph, float *Er, float *eth, float *Eph,
    float *Hr, float *Hth, float *Hph, float del_r, float del_th, float del_ph,
    float dt, float th0, float r0, float mu, int l );

__global__ void H_update_pml(
    int nr, int nth, int nph, float *Er, float *Eth, float *Eph,
    float *Hr, float *Hth, float *Hph, float *Hr_th1, float *Hr_th2, float *Hr_ph,
    float *Hth_ph, float *Hth_r, float *Hph_r, float *Hph_th, float *sig_th_h, float *sig_ph_h,
    PML *pml_Hr, PML *pml_Hth, PML *pml_Hph, float del_r, float del_th, float del_ph, float dt,
    float r0, float theta0, float mu0 );

__global__ void surface_H_update(
    int nr, int nth, int nph, float *nEr, float *nEth, float *nEph, float *Hth, float *Hph,
    float z_real, float z_imag, float del_r, float del_th, float del_ph, float dt,
    float r0, float th0, float mu0
);

__global__ void E_old_to_new(
    int nr, int nth, int nph, float *nEr, float *nEth, float *nEph, float *oEr, float *oEth, float *oEph
);

inline float dist(float i){return R0 + i*delta_r;};
inline float th(float j){return THETA0 + j*delta_th;};
inline float ph(float k){return k*delta_ph;};

__device__ inline float dist_d( float i, float del_r, float r0 ){ return r0 + i*del_r; };
__device__ inline float th_d( float j, float del_th, float th0 ){ return th0 + j*del_th; };
__device__ inline float ph_d( float k, float del_ph ){ return k*del_ph; };

inline int idx_Er( int i, int j, int k ){ return i*((Nth+1)*(Nph+1)) + j*(Nph+1) + k; };
inline int idx_Eth( int i, int j, int k ){ return i*(Nth*(Nph+1)) + j*(Nph+1) + k; };
inline int idx_Eph( int i, int j, int k ){ return i*((Nth+1)*Nph) + j*Nph + k; };
inline int idx_Hr( int i, int j, int k ){ return i*(Nth*Nph) + j*Nph + k; };
inline int idx_Hth( int i, int j, int k ){ return i*((Nth+1)*Nph) + j*Nph + k; };
inline int idx_Hph( int i, int j, int k ){ return i*(Nth*(Nph+1)) + j*(Nph+1) + k; };

__device__ inline int idx_Er_d( int i, int j, int k,  int Nth, int  Nph){ return i*((Nth+1)*(Nph+1)) + j*(Nph+1) + k; };
__device__ inline int idx_Eth_d( int i, int j, int k, int Nth, int Nph ){ return i*(Nth*(Nph+1)) + j*(Nph+1) + k; };
__device__ inline int idx_Eph_d( int i, int j, int k, int Nth, int Nph ){ return i *((Nth+1)*Nph) + j*Nph + k; };
__device__ inline int idx_Hr_d( int i, int j, int k, int Nth, int Nph ){ return i*(Nth*Nph) + j*Nph + k; };
__device__ inline int idx_Hth_d( int i, int j, int k, int Nth, int Nph ){ return i*((Nth+1)*Nph) + j*Nph + k; };
__device__ inline int idx_Hph_d( int i, int j, int k, int Nth, int Nph ){ return i*(Nth*(Nph+1)) + j*(Nph+1) + k; };

__device__ inline float C1_d( float sig, float inv_Dt ){ return ( (inv_Dt - sig/2.0 )/( inv_Dt + sig/2.0 )); };
__device__ inline float C2_d( float r, float sig, float del_th, float inv_Dt ){ return 1.0/r/del_th/( inv_Dt + sig/2.0 ); };
__device__ inline float C3_d( float r, float th, float Dt ){ return Dt*std::cos(th)/std::sin(th)/2.0/r; };
__device__ inline float C4_d( float r, float th, float sig, float del_ph, float inv_Dt ){ return 1.0/r/std::sin(th)/del_ph/( inv_Dt + sig/2.0); };
__device__ inline float C5_d( float r, float del_r, float Dt ){ return Dt/r/del_r; };
__device__ inline float C6_d( float r, float sig, float del_th, float inv_Dt ){ return 1.0/( inv_Dt + sig/2.0 )/r/del_th; };

__device__ inline int idx_mat_d( int i, int j, int k, int m, int n, int ion_L, int Nth, int Nph ){
    return i*(Nth*Nph*3*3) + j*(Nph*3*3) + k*(3*3) + m*3 + n;
};


