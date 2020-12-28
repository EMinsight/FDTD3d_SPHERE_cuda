#define _USE_MATH_DEFINES
#include <cmath>
#include "fdtd3d.h"

void sigma_calc( float* sig_th, float* sig_ph, float* sig_th_h, float* sig_ph_h )
{
    for( int i = 0; i <= Nth; i++ ){
        float theta = float(i)*delta_th;
        float theta_h = ( float(i) + 0.5 )*delta_th;

        if( i <= L ){
            sig_th[i] = sig_th_max*std::pow( (L*delta_th - theta)/float(L)/delta_th, M);
            sig_th_h[i] = sig_th_max*std::pow( (L*delta_th - theta_h)/float(L)/delta_th, M);
        }

        else if( i >= Nth - L ){
            sig_th[i] = sig_th_max*std::pow( (theta - (Nth - L)*delta_th)/float(L)/delta_th, M );
            sig_th_h[i] = sig_th_max*std::pow( (theta_h - (Nth - L)*delta_th)/float(L)/delta_th, M );
        }

        else{
            sig_th[i] = 0.0;
            sig_th_h[i] = 0.0;
        }
    }

    for( int i = 0; i <= Nph; i++ ){
        float phi = float(i)*delta_ph;
        float phi_h = ( float(i) + 0.5 )*delta_ph;

        if( i <= L ){
            sig_ph[i] = sig_ph_max*std::pow( (L*delta_ph - phi)/float(L)/delta_ph, M );
            sig_ph_h[i] = sig_ph_max*std::pow( (L*delta_ph - phi_h)/float(L)/delta_ph, M );
        }

        else if( i >= Nph - L ){
            sig_ph[i] = sig_ph_max*std::pow( (phi - (Nph - L)*delta_ph)/float(L)/delta_ph, M );
            sig_ph_h[i] = sig_ph_max*std::pow( (phi_h - (Nph - L)*delta_ph)/float(L)/delta_ph, M);
        }
        else{
            sig_ph[i] = 0.0;
            sig_ph_h[i] = 0.0;
        }
    }

    sig_th_h[L] = 0.0;
    sig_ph_h[L] = 0.0;
}