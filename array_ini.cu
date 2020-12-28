#include <iostream>
#include "fdtd3d.h"

float* array_ini( int size, float ini )
{
    float *array;
    array = new float[size];
    for( int i = 0; i < size; i++ )array[i] = ini;

    return array;
}