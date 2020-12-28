#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <complex>
#include "gcc.h"

std::complex <float> surface_impe( void )
{
    std::complex <float> zj = ( 0.0, 0.0 );
    zj.real(0.0);
    zj.imag(1.0);

    float conduct = SIGMA_VERY_DRY_GROUND;

    std::complex <float> z = Z0/std::sqrt(EPSR - (zj*conduct/EPS0/omega));

    return z;
}