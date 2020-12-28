#include "PML.h"

void PML::set_point_1( int j1, int k1 ){
    p_j1 = j1;
    p_k1 = k1;
}

void PML::set_point_2( int j2, int k2 ){
    p_j2 = j2;
    p_k2 = k2;
}

void PML::set_point( int j1, int j2, int k1, int k2 ){
    set_point_1( j1, k1 );
    set_point_2( j2, k2 );
}