/*
 * IRI (fortran) をC++ から利用
 */
#include <iostream>
#include <fstream>

#include "iri.h" /* このヘッダファイルをインクルードする */

int main(void){
  float *outf = new float [20*1000]; /* 出力を入れる配列 */
  float xlat = 35.0f, xlon = 135.0f; /* 緯度(-90〜+90)、経度(-180〜+180) */
  int iy = 2016, imd = 301; /* 2016年3月1日 */
  float hour = 3.0;  /* Universal Time (LT 18:00) */
  /* 60kmから100kmまで 1km毎 */
  float alt_beg = 60.0f, alt_end = 100.f, alt_step = 1.0f;

  /* IRIで計算
   * fort.7というファイルができる
   */
  iri_(&xlat, &xlon, &iy, &imd, &hour, &alt_beg, &alt_end, &alt_step, outf);

  /* 電子密度 [m^{-3}] は outf[i] に入る。 */
  /* 電子温度 [K] は outf[3000 + i] に入る。 */
  std::ofstream ofs("output.dat");
  for(int i = 0; i < int( (alt_end - alt_beg)/alt_step )+1; i++){
    ofs << alt_beg + i*alt_step << " " << outf[i] << " "
        << outf[3*1000 + i] << std::endl;
  }
  ofs.close();

}
