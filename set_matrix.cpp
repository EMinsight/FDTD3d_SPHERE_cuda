#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include "gcc.h"

void set_matrix(
    float *Cmat, float *Fmat, double *Nh, double *ny ){
    // float型のままだと値が大きいためinfとなるので //
    // double型で計算 -> float型にcast //

    double omg_p = 0.0;
    double omg_c = E_Q*B_abs/E_M;
    std::complex <double> zj = ( 0.0, 0.0 );
    
    zj.real(0.0);
    zj.imag(1.0);

    for(int ir = Nr - ion_L; ir < Nr; ir++){
        int i = ir - (Nr - ion_L);
        double Alt = ir*delta_r;
        for(int j = 0; j < Nth; j++){
            for(int k = 0; k < Nph; k++){
              
              omg_p = double(E_Q)*std::sqrt(Nh[i]/double(E_M)/double(EPS0));

              std::complex <double> omg = double(omega) - zj * ny[i];
              std::complex <double> diag_comp = double(omega)/(omg_c*omg_c - omg*omg);
              std::complex <double> offd_comp = zj * omg_c / (omg_c*omg_c - omg*omg);
              std::complex <double> coef = zj * double( EPS0 ) * omg_p * omg_p;
              /*if(j==Nth/2 && k == Nph/2) {
                    std::cout << E_Q << " " << Nh[i] << " " << E_M << " " << EPS0 << std::endl;
                    std::cout << omg_p << " " << omg << " " << diag_comp << " " << offd_comp << " " << coef << "\n";
                    }*/
                Eigen::Matrix3f Sigma = Eigen::Matrix3f::Zero(3, 3);
                Sigma(0, 0) = real( coef*diag_comp );
                Sigma(1, 1) = real( coef*diag_comp );
                Sigma(0, 1) = real( coef*offd_comp );
                Sigma(1, 0) = real( -1.0*coef*offd_comp );
                Sigma(2, 2) = real( -1.0*coef/omg );

                Eigen::Matrix3f A =
                  double(EPS0)/double(Dt)*Eigen::Matrix3f::Identity(3, 3) + 0.5*Sigma;
                Eigen::Matrix3f B =
                  double(EPS0)/double(Dt)*Eigen::Matrix3f::Identity(3, 3) - 0.5*Sigma;
                Eigen::Matrix3f C = A.inverse()*B;
                Eigen::Matrix3f F = 1./double(Dt)*A.inverse();

                for(int m = 0; m < 3; m++){
                    for(int n = 0; n < 3; n++){
                        Cmat[idx_mat(i, j, k, m, n)] = float( C(m, n) );
                        Fmat[idx_mat(i, j, k, m, n)] = float( F(m, n) );
                    }
                }

            }
        }

    }

}



