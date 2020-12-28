#include "fdtd3d.h"

void PML_idx_initialize(
    PML* idx_Dr, PML* idx_Dth, PML* idx_Dph,
    PML* idx_Hr, PML* idx_Hth, PML* idx_Hph)
{
    /* Define D field index in PML area */
    idx_Dr[0].set_point(1, L, 1, Nph - 1);
    idx_Dr[1].set_point(Nth - L, Nth - 1, 1, Nph - 1);
    idx_Dr[2].set_point(L + 1, Nth - L - 1, 1, L);
    idx_Dr[3].set_point(L + 1, Nth - L - 1, Nph - L, Nph - 1);

    idx_Dth[0].set_point(0, L - 1, 1, Nph - 1);
    idx_Dth[1].set_point(Nth - L, Nth - 1, 1, Nph - 1);
    idx_Dth[2].set_point(L, Nth - L - 1, 1, L);
    idx_Dth[3].set_point(L, Nth - L - 1, Nph - L, Nph - 1);

    idx_Dph[0].set_point(1, L, 0, Nph - 1);
    idx_Dph[1].set_point(Nth - L, Nth - 1, 0, Nph - 1);
    idx_Dph[2].set_point(L + 1, Nth - L - 1, 0, L - 1);
    idx_Dph[3].set_point(L + 1, Nth - L - 1, Nph - L, Nph - 1);

    /* Define H field index in PML area */
    idx_Hr[0].set_point(0, L - 1, 0, Nph - 1);
    idx_Hr[1].set_point(Nth - L, Nth - 1, 0, Nph - 1);
    idx_Hr[2].set_point(L, Nth - L - 1, 0, L - 1);
    idx_Hr[3].set_point(L, Nth - L - 1, Nph - L, Nph - 1);

    idx_Hth[0].set_point(1, L, 0, Nph - 1);
    idx_Hth[1].set_point(Nth - L, Nth - 1, 0, Nph - 1);
    idx_Hth[2].set_point(L + 1, Nth - L - 1, 0, L - 1);
    idx_Hth[3].set_point(L + 1, Nth - L - 1, Nph - L, Nph - 1);

    idx_Hph[0].set_point(0, L - 1, 1, Nph - 1);
    idx_Hph[1].set_point(Nth - L, Nth - 1, 1, Nph - 1);
    idx_Hph[2].set_point(L, Nth - L - 1, 1, L);
    idx_Hph[3].set_point(L, Nth - L - 1, Nph - L, Nph - 1);

}