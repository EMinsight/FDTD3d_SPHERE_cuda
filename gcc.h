#include <complex>

extern const float MU0;
extern const float EPSR;
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

// Surface Sigma info //
extern const float Z0;
extern const float SIGMA_PEC;
extern const float SIGMA_SEA;
extern const float SIGMA_WET_GROUND;
extern const float SIGMA_DRY_GROUND;
extern const float SIGMA_VERY_DRY_GROUND;
extern const float SIGMA_FRESH_WATER_ICE;

void Ne_allocate( double *Nh, double *Re );
void ny_allocate( double *ny, double *Re );
void set_matrix( float *Cmat, float *Fmat, double *Nh, double *ny );
std::complex <float> surface_impe( void );

inline int idx_mat( int i, int j, int k, int m, int n){ return i*(Nth*Nph*3*3) + j*(Nph*3*3) + k*(3*3) + m*3 + n; };
