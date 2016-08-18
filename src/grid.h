#include <iostream>
#include <fstream>
#include <string.h>
#include <fftw3.h>
#include <cmath>
#include <vector>
#include <stdlib.h>

#define pi 3.1415926535897932385

const float twoPi=2*pi;
const int N = 16; // size of sq. grid in 1 dim. It must be even.
const int uniq=(N+4)*(N+2)/8; // the number of unique values of the magnitude of q; used to be =(N+4)*(N+2)/8 when lx=ly
const int uniq_Ny=N*(N+2)/8; // same as above, but excluding values at the Nyquist frequency; =N*(N+2)/8 when lx=ly




// function prototypes
void fullArray( float array1R[][N], float array1I[][N], fftwf_complex array2[], float lxy);
void qav(float array2D[][N], float array1D_uniq[], int Ny);
