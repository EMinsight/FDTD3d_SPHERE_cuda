#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <cmath>
#include "gcc.h"

void Ne_allocate( double *Nh, double *Re )
{
    double trash[13];
    double Te( 0.0 );
    double height( 0.0 );
    int bufsize = 256;
    char buf[bufsize];
    int s = 25; // 読み始めの行 //

    std::ifstream ifs("./IRI-2016_for_cpp/fort.7");

    if(!ifs){
      std::cout << "file is not exist." << "\n";
      std::exit(0);
    }

    //空読み//
    for(int i = 0; i < s; i++ ){
      ifs.getline(buf, bufsize);
    }

    for( int i = 0; i < ion_L; i++ ){
        ifs >> height >> Nh[i];
        Nh[i] = Nh[i]*1.0e6;

        for( int m = 0; m < 3; m++ ) ifs >> trash[m];

        ifs >> Te;
        Re[i] = Te/300.0;

        for( int m = 0; m < 9; m++ ) ifs >> trash[m];

        if(Nh[i] < 0.0) Nh[i] = 0.0;
    }

    ifs.close();
}


