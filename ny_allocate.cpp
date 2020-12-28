#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <nrlmsise-00.h>
#include "gcc.h"

void ny_allocate( double *ny, double *Re )
{
    double N_o( 0.0 );
    double N_n2( 0.0 );
    double N_o2( 0.0 );

    ap_array Xp;
    nrlmsise_input Input;
    nrlmsise_flags Flag;
    nrlmsise_output Output;

    for( int i = 0; i < 7; i++ ) Xp.a[i] = 100.0;

    Flag.switches[0] = 0;
    for( int i = 1; i < 24; i++ ){
        Flag.switches[i] = 1;
    }

    Input.year = 2016;
    Input.doy = 61;
    Input.sec = 43200;
    Input.g_lat = 35.0;
    Input.g_long = 135.0;
    Input.lst = (Input.sec/3600.0) + (Input.g_long/15.0);
    Input.f107A = 150;
    Input.f107 = 150;
    Input.ap = 4.0;
    Input.ap_a = &Xp;
    
    for( int i = 0; i < ion_L; i++ ){

        Input.alt = int( Nr - ion_L);
        gtd7(&Input, &Flag, &Output);

        N_o = Output.d[1]*1.e6;
        N_n2 = Output.d[2]*1.e6;
        N_o2 = Output.d[3]*1.e6;

        ny[i] = 4.6*N_n2*std::pow(Re[i], 0.90) + 4.3*N_o2*std::pow(Re[i], 0.55)
        + 1.5*N_o*std::pow(Re[i], 0.83)*1.0e-15;

    }
}
