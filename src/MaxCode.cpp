#include "grid.h"
using namespace std;

const int nl= 2000; // number of lipids per frame
const int frames=3981; // the number of frames to be analyzed, 495 for rep2cc
//nl=8192 for 16x16
//nl=512 for run2
//nfr=2501 for run2 and 16x16
//nfr=7893 for 16x16_2

const float cutang = cos(90*pi/180); // if the reference angle between the director and the z axis is greater than
										// this, throw out the lipid. When theta=pi/2
										// nothing is discarded.
float lipidx[3*nl]; // declaring these as global variables
float lipidy[3*nl]; // allocates more memory for these large arrays
float lipidz[3*nl]; // for lipidXMUA.out, there are 37 points per lipid

int main() {

int i,j,k,frame_num;
float lx_av=0; // average box length for x
float ly_av=0; // average box length for y

int DUMP=0; // =1 if real space data will be exported, =0 otherwise
int DUMPQ=1; // =1 if Fourier space averages will be exported, =0 otherwise
int TILT=1; // =1 if the tilt averages are to be calculated
int AREA=0; // =1 if FT of number densities is to be calculated, =0 otherwise
int AREA_tail=0; // when AREA==1, area fluctuations at the tails are measured if AREA_tail=1
				// if AREA_tail=0 the area fluctuations at the interfaces are measured

// binned quantities in real space
int nl1,nl2; // the number of lipids within each monolayer
int nt1,nt2; // number of lipids witin each monolayer that aren't too tilted
float z1[N][N], z2[N][N]; // coarse grained height field of each monolayer
float h[N][N], t[N][N]; // height and thickness

int nlg1[N][N], nlg2[N][N]; // number of lipids within each patch
int nlt1[N][N], nlt2[N][N]; // number of lipids used for tilt calculations
int nlb1[N][N], nlb2[N][N]; // number of bad lipids per patch

float psiRU[N][N]; // real and imaginary parts of
float psiIU[N][N]; // the Fourier transform of the number density of each monolayer "Up" & "Down"
float psiRD[N][N]; // when the number density is written as a sum of delta functions
float psiID[N][N]; // FT(phi-phi0in)
float rhoSigq2[N][N], rhoDelq2[N][N]; // the symmetric and antisymmetric parts of psi

float h_real[N][N], h_imag[N][N]; // real and imaginary parts of the non-grid based Fourier transform
				// of the height field
float hq2Ed[N][N]; // squared average of the non-grid based height field

float t1[N][N][3]; //  top tilt vector
float t2[N][N][3]; //  bottom tilt vector
float dm[N][N][2], dp[N][N][2]; // d vectors

float t1mol[3], t2mol[3]; //the tilt vector of an individual lipid
float u[3], v[3]; // basis vector in the plane perp to N with no component in y

int hist_t[100][100];
int hist_t2[100];
int histn, histt, histtx;
int hist_tcount=0;
float tmag,ut,vt;
float tmax=0;
int tproj1_cum[100]={0};
int tproj2_cum[100]={0};
float  ty_cum[100]={0};
float ty_cum2[100]={0};
float txq[frames]={0};
float rootgxinv, u_dot_t;
float tproj1[3], tproj2[3];
float tghist[100]={0};

float dot_cum=0;

float n1[N][N][2]; // top binned director field
float n2[N][N][2]; // bot binned director field
float um[N][N][2], up[N][N][2]; // u vectors


// 1D real quantities passed to fftw
float h1D[N*N], t1D[N*N];
float z1_1D[N*N], z2_1D[N*N];
float t1x1D[N*N], t1y1D[N*N], dmx1D[N*N], dmy1D[N*N], dpx1D[N*N], dpy1D[N*N]; // tilt fields
float umx1D[N*N], umy1D[N*N], upx1D[N*N], upy1D[N*N]; // symm and antisymm director fields
float dz1x1D[N*N], dz1y1D[N*N], dz2x1D[N*N], dz2y1D[N*N]; // derivatives passed from fftw
float norm_1[N*N][3]; // top normal vector
float norm_2[N*N][3]; // bottom normal vector

// Complex Fourier transforms. All quantities, before being transformed, are real, so the output is only given for the upper half
// of the complex plane. The output is also 1-dimensional. The function at the end of this file puts the output back into [N][N] form.
// here 'S' stands for 'small' since the output is not N*N
fftwf_complex hqS[N*(N/2+1)], tqS[N*(N/2+1)];
fftwf_complex z1qS[N*(N/2+1)], z2qS[N*(N/2+1)];  // FT of upper and lower monolayer surf.
fftwf_complex n1xqS[N*(N/2+1)], n1yqS[N*(N/2+1)];  // n1(q) - upper monolayer
fftwf_complex n2xqS[N*(N/2+1)], n2yqS[N*(N/2+1)];  // n1(q) - upper monolayer
fftwf_complex dz1xqS[N*(N/2+1)], dz1yqS[N*(N/2+1)], dz2xqS[N*(N/2+1)], dz2yqS[N*(N/2+1)]; //derivatives
fftwf_complex t1xqS[N*(N/2+1)], t1yqS[N*(N/2+1)]; // tilt of top monolayer
fftwf_complex dmxqS[N*(N/2+1)], dmyqS[N*(N/2+1)], dpxqS[N*(N/2+1)], dpyqS[N*(N/2+1)]; // d vectors
fftwf_complex umxqS[N*(N/2+1)], umyqS[N*(N/2+1)], upxqS[N*(N/2+1)], upyqS[N*(N/2+1)]; // u vectors

//average quantities and frequently used variables

float mag; // the magnitude of each director before it is normalized
float qx,qy; //  wave numbers used when calculating the phiXX's
int xj[nl], yj[nl]; // patch coordinates of each lipid
int xi, yi; // patch coordinates of a single lipid
float xx, yy; // xy coordinates of the portion of each lipid used to measure area fluctuations
int i1, i2, j1, j2; // neighboring cordinates to patch [i][j], used for interpolation
float nn; // 1.0/(the total number of lipids in the neighboring patches)
float t0_frame; // average monolayer thickness per frame
float t0=0; // average thickness
float t0in=18.08332825; // average thickness which is used to find the q=0 mode    (17.98627281 UA) (18.31448364 dppc)
float tq0=0; // <|t_q|^2> at q=0
float tq0_frame;
float tilt1_frame[2], tilt2_frame[2];
float tilt1_0=0;
float tilt2_0=0;
float phi0_frame=0;
float phi0=0; // mean lipid number density for monolayer
// float phi0in=0.0157489; // used to find q=0 mode when all lipids are kept
float phi0in=0.01546141133; // used to find q=0 mode
float nNav=1.007396817; // average (n.N)
float z1avg, z2avg;
float z1sq_av=0;
float z2sq_av=0;
float z1sq_av_frame, z2sq_av_frame;
float Lxy; // sqrt(lx[frame_num]*ly[frame_num])
float twoPiLx, twoPiLy; // 2*pi/lx[frame_num], 2*pi/ly[frame_num]
float invLx, invLy, invLxy; // 1/lx[frame_num], 1/ly[frame_num], 1/sqrt(lx[frame_num]*ly[frame_num])
float dlx, dly; // lx[frame_num]/N, ly[frame_num]/N ; the widths of each patch
float root_ginv1, root_ginv2; // 1/sqrt(1+(grad z)^2) for the top and bottom monolayers
float dot1, dot2; // (n.N)
float denom1,denom2; // 1/(n.N)
int empty_tot=0; // the number of instances where two neighboring patches are empty
int empty; // the number of empty neighboring patches within each frame

int q[N][N][2]; // full matrix of 2D q values
float cosq[N][N], sinq[N][N]; // = qx/q, qy/q, used for calculating the parallel and perp components of dm, dp
float q2[N][N]; // full matrix of the magnitude of q
float q2test[N][N]; // used to check if any values of q have changed over the course of the analysis
int qi,qj; // used when calculating derivatives in Fourier space

// real and imaginary parts of the Fourier transforms:
float hqR[N][N], hqI[N][N], tqR[N][N], tqI[N][N]; // the real and imaginary parts of the full hq and tq matrices
float t1xR[N][N], t1xI[N][N], t1yR[N][N], t1yI[N][N];
float dmxR[N][N], dmxI[N][N], dmyR[N][N], dmyI[N][N], dpxR[N][N], dpxI[N][N], dpyR[N][N], dpyI[N][N];
float umxR[N][N], umxI[N][N], umyR[N][N], umyI[N][N], upxR[N][N], upxI[N][N], upyR[N][N], upyI[N][N];

float t1xR_cum[N][N], t1xI_cum[N][N], t1yR_cum[N][N], t1yI_cum[N][N];

float dmparR[N][N], dmparI[N][N], dmperR[N][N], dmperI[N][N], dpparR[N][N], dpparI[N][N], dpperR[N][N], dpperI[N]
[N];
float umparR[N][N], umparI[N][N], umperR[N][N], umperI[N][N], upparR[N][N], upparI[N][N], upperR[N][N], upperI[N]
[N];
// magnitude of the Fourier transforms, which are accumulated over all frames
float hq2[N][N], tq2[N][N];
float t1xq2[N][N], t1yq2[N][N], dmq2[N][N], dpq2[N][N];
float dmparq2[N][N], dmperq2[N][N], dpparq2[N][N], dpperq2[N][N];
float hdmpar[N][N], tdppar[N][N];

float umparq2[N][N], umperq2[N][N], upparq2[N][N], upperq2[N][N];
float dum_par[N][N], dup_par[N][N];

float hq4[N][N],umparq4[N][N],umperq4[N][N]; // used for calculating variance


for (i=0; i<N; i++){
	for(j=0; j<N; j++){

		hq2[i][j]=0;		tq2[i][j]=0;

		rhoSigq2[i][j]=0;	rhoDelq2[i][j]=0;

		hq2Ed[i][j]=0;

		t1xR_cum[i][j]=0;	t1xI_cum[i][j]=0;	t1yR_cum[i][j]=0;	t1yI_cum[i][j]=0;

		t1xq2[i][j]=0;		t1yq2[i][j]=0;

		dmq2[i][j]=0;		dpq2[i][j]=0;

		dmparq2[i][j]=0;	dmperq2[i][j]=0;	dpparq2[i][j]=0;	dpperq2[i][j]=0;

		umparq2[i][j]=0;	umperq2[i][j]=0;	upparq2[i][j]=0;	upperq2[i][j]=0;

		hdmpar[i][j]=0;		tdppar[i][j]=0;

		dum_par[i][j]=0;	dup_par[i][j]=0;

		hq4[i][j]=0;		umparq4[i][j]=0;	umperq4[i][j]=0;
	}
}

for(i=0; i<100; i++){ hist_t2[i]=0;
	for(j=0; j<100; j++){
		hist_t[i][j]=0;
	}
}

float q2_uniq[uniq]={0.0};
float hq2_uniq[uniq]={0.0}; 		float tq2_uniq[uniq]={0.0};

float rhoSigq2_uniq[uniq]={0.0};	float rhoDelq2_uniq[uniq]={0.0};

float hq2Ed_uniq[uniq]={0.0};

float q2_uniq_Ny[uniq_Ny]={0.0};

float t1xq2_uniq[uniq_Ny]={0.0};	float t1yq2_uniq[uniq_Ny]={0.0};

float dmq2_uniq[uniq_Ny]={0.0};		float dpq2_uniq[uniq_Ny]={0.0};

float dmparq2_uniq[uniq_Ny]={0.0};	float dmperq2_uniq[uniq_Ny]={0.0};

float dpparq2_uniq[uniq_Ny]={0.0};	float dpperq2_uniq[uniq_Ny]={0.0};

float hdmpar_uniq[uniq_Ny]={0.0};	float tdppar_uniq[uniq_Ny]={0.0};

float umparq2_uniq[uniq_Ny]={0.0};	float umperq2_uniq[uniq_Ny]={0.0};

float upparq2_uniq[uniq_Ny]={0.0};	float upperq2_uniq[uniq_Ny]={0.0};

float dum_par_uniq[uniq_Ny]={0.0};	float dup_par_uniq[uniq_Ny]={0.0};

float hq4_uniq[uniq]={0.0};

float umparq4_uniq[uniq_Ny]={0.0};	float umperq4_uniq[uniq_Ny]={0.0};


ofstream buf1, buf2, buf4;

FILE *lboxpx, *lboxpy, *lboxpz, *lipidxp, *lipidyp, *lipidzp;

float lx[frames]; // array containing the box dimensions at each frame
float ly[frames];
float lz[frames];

float head[nl][3]; // each group has its 3 spatial components;
float end1[nl][3]; //
float end2[nl][3]; //
float dir[nl][3]; // the director for each molecule
				; // the fourth component is 1 for a good lipid and 0 for a bad one

int good[nl]; // =0 if the lipid is tilted too much, =1 if it's okay

float zavg[frames]; //the average z coordinate of the bilayer at each frame

lboxpx=fopen("./boxsizeX.out","r");
lboxpy=fopen("./boxsizeY.out","r");
lboxpz=fopen("./boxsizeZ.out","r");

lipidxp=fopen("./LipidX.out","r");
lipidyp=fopen("./LipidY.out","r");
lipidzp=fopen("./LipidZ.out","r");

// lboxpx=fopen("/usr/projects/nanoparticles/maxcw/max/rep2d/boxsizeX.out","r");
// lboxpy=fopen("/usr/projects/nanoparticles/maxcw/max/rep2d/boxsizeY.out","r");
// lboxpz=fopen("/usr/projects/nanoparticles/maxcw/max/rep2d/boxsizeZ.out","r");
//
// lipidxp=fopen("/usr/projects/nanoparticles/maxcw/max/rep2d/LipidX.out","r");
// lipidyp=ifopen("/usr/projects/nanoparticles/maxcw/max/rep2d/LipidY.out","r");
// lipidzp=fopen("/usr/projects/nanoparticles/maxcw/max/rep2d/LipidZ.out","r");

for(frame_num=0; frame_num< frames; frame_num++){

	fscanf(lboxpx,"%f",&lx[frame_num]);
	fscanf(lboxpy,"%f",&ly[frame_num]);
	fscanf(lboxpz,"%f",&lz[frame_num]);
}

fclose(lboxpx);		fclose(lboxpy);		fclose(lboxpz);

//calculate the average box size;
for(frame_num=0; frame_num<frames; frame_num++){

	lx_av += lx[frame_num];
	ly_av += ly[frame_num];
}
lx_av /= frame_num;
ly_av /= frame_num;

if(DUMP){

	buf1.open("./tq0Dyn.dat", ios::out);
// 	buf2.open("/home/maxcw/work/matlab/dmy.dat", ios::out);
	}

if(DUMPQ){buf4.open("./spectraMUA500.dat", ios:: out);}
// if(DUMPQ){buf4.open("/usr/projects/nanoparticles/maxcw/max/rep2d/spectra_st0.dat", ios:: out);}

//----------------------------------------------------------------------------------------------
//FOURIER SPACE SETUP////////////////////////////////////////////////////////////////////////////
//----------------------------------------------------------------------------------------------

	//constuct q matrix

for(i=0; i<N; i++){
	for(j=0; j<N; j++){

		q[i][j][0] =((i < N/2) ? i : i-N);
		q[i][j][1] =((j < N/2) ? j : j-N);


		if(i==0 && j==0){cosq[i][j]=0; sinq[i][j]=0;}
		else{

		mag=1.0/sqrt(q[i][j][0]*q[i][j][0] + q[i][j][1]*q[i][j][1]);

		cosq[i][j]=q[i][j][0]*mag; // x is the column and
		sinq[i][j]=q[i][j][1]*mag;} // y is the row
	}
}

static fftwf_plan spectrum_plan;
static fftwf_plan inv_plan;

const int m[2]={N,N};

//if(need_fftw_plan) {
    spectrum_plan = fftwf_plan_many_dft_r2c(2, m, 1,
                                           h1D, NULL, 1, 0,
                                           hqS, NULL, 1, 0, FFTW_MEASURE);

    inv_plan = fftwf_plan_many_dft_c2r(2, m, 1,
                                      dz1xqS, NULL, 1, 0,
                                      dz1x1D, NULL, 1, 0, FFTW_MEASURE);
//    need_fftw_plan=false;
//
//}
//----------------------------------------------------------------------------------------------
//LOOP OVER EACH FRAME////////////////////////////////////////////////////////////////////////////
//----------------------------------------------------------------------------------------------

for(frame_num=0; frame_num<frames; frame_num++){

	// initilize complex containers
	memset(hqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(tqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(z1qS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(z2qS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(dz1xqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(dz1yqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(dz2xqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(dz2yqS, 0, N*(N/2+1)*sizeof(fftwf_complex));

	memset(t1xqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(t1yqS, 0, N*(N/2+1)*sizeof(fftwf_complex));

	memset(dpxqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(dpyqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(dmxqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(dmyqS, 0, N*(N/2+1)*sizeof(fftwf_complex));

	memset(upxqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(upyqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(umxqS, 0, N*(N/2+1)*sizeof(fftwf_complex));
	memset(umyqS, 0, N*(N/2+1)*sizeof(fftwf_complex));

	/////initialize scalars
	zavg[frame_num]=0;
	t0_frame=0;
	tq0_frame=0;
	tilt1_frame[0]=0; tilt1_frame[1]=0;
	tilt2_frame[0]=0; tilt2_frame[1]=0;
	z1avg=0;
	z2avg=0;
	z1sq_av_frame=0;	z2sq_av_frame=0;
	nl1=0;	nl2=0;
	nt1=0;	nt2=0;
	twoPiLx=2*pi/lx[frame_num];
	twoPiLy=2*pi/lx[frame_num];
	invLx=1/lx[frame_num];
	invLy=1/ly[frame_num];
	invLxy=1/sqrt(lx[frame_num]*ly[frame_num]);
	dlx=lx[frame_num]/N;
	dly=ly[frame_num]/N;
	Lxy=sqrt(lx[frame_num]*ly[frame_num]);
	empty=0;

/////initialize arrays
	for(j=0; j<N; j++){
		for(k=0; k<N; k++){
			psiRU[j][k]=0;		psiIU[j][k]=0;	psiRD[j][k]=0;	psiRD[j][k]=0;

			h_real[j][k]=0;		h_imag[j][k]=0;

			z1[j][k]=0;		z2[j][k]=0;

			nlg1[j][k]=0;		nlg2[j][k]=0;

			nlt1[j][k]=0;		nlt2[j][k]=0;

			nlb1[j][k]=0;		nlb2[j][k]=0;

			hqR[j][k]=0;		hqI[j][k]=0;

			tqR[j][k]=0;		tqI[j][k]=0;

			//vector quantities

			t1[j][k][0]=0;		t1[j][k][1]=0;		t1[j][k][2]=0;

			t2[j][k][0]=0;		t2[j][k][1]=0;		t2[j][k][2]=0;

			n1[j][k][0]=0;		n1[j][k][1]=0;

			n2[j][k][0]=0;		n2[j][k][1]=0;

			t1xR[j][k]=0;		t1xI[j][k]=0;	t1yR[j][k]=0;	t1yI[j][k]=0;

			dmxR[j][k]=0;		dmxI[j][k]=0;

			dmyR[j][k]=0;		dmyI[j][k]=0;

			dpxR[j][k]=0;		dpxI[j][k]=0;

			dpyR[j][k]=0;		dpyI[j][k]=0;

			umxR[j][k]=0;		umxI[j][k]=0;

			umyR[j][k]=0;		umyI[j][k]=0;

			upxR[j][k]=0;		upxI[j][k]=0;

			upyR[j][k]=0;		upyI[j][k]=0;

		}
	}

//////assign each group to an array

	for(i=0; i < 3*nl; i++){
		fscanf(lipidxp,"%f",&lipidx[i]);
		fscanf(lipidyp,"%f",&lipidy[i]);
		fscanf(lipidzp,"%f",&lipidz[i]);
	}

///// fill the head, end1, end2, dir arrays with their coordinates for this frame

	for(i=0; i< nl; i++){
		//head[i][0]=(lipidx[37*i]+lipidx[37*i+1]+lipidx[37*i+2]+lipidx[37*i+3]+lipidx[37*i+4]+lipidx[37*i+5]+lipidx[37*i+6]+lipidx[37*i+7]+lipidx[37*i+8])/9;
		//head[i][1]=(lipidy[37*i]+lipidy[37*i+1]+lipidy[37*i+2]+lipidy[37*i+3]+lipidy[37*i+4]+lipidy[37*i+5]+lipidy[37*i+6]+lipidy[37*i+7]+lipidy[37*i+8])/9;
		//head[i][2]=(lipidz[37*i]+lipidz[37*i+1]+lipidz[37*i+2]+lipidz[37*i+3]+lipidz[37*i+4]+lipidz[37*i+5]+lipidz[37*i+6]+lipidz[37*i+7]+lipidz[37*i+8])/9;

		head[i][0]=lipidx[3*i];
		head[i][1]=lipidy[3*i];
		head[i][2]=lipidz[3*i];

		//end1[i][0]=(lipidx[37*i+9]+lipidx[37*i+10]+lipidx[37*i+11]+lipidx[37*i+12]+lipidx[37*i+13]+lipidx[37*i+14]+lipidx[37*i+15]+lipidx[37*i+16]+lipidx[37*i+17]+lipidx[37*i+18]+lipidx[37*i+19]+lipidx[37*i+20]+lipidx[37*i+21]+lipidx[37*i+22])/14;
		//end1[i][1]=(lipidy[37*i+9]+lipidy[37*i+10]+lipidy[37*i+11]+lipidy[37*i+12]+lipidy[37*i+13]+lipidy[37*i+14]+lipidy[37*i+15]+lipidy[37*i+16]+lipidy[37*i+17]+lipidy[37*i+18]+lipidy[37*i+19]+lipidy[37*i+20]+lipidy[37*i+21]+lipidy[37*i+22])/14;
		//end1[i][2]=(lipidz[37*i+9]+lipidz[37*i+10]+lipidz[37*i+11]+lipidz[37*i+12]+lipidz[37*i+13]+lipidz[37*i+14]+lipidz[37*i+15]+lipidz[37*i+16]+lipidz[37*i+17]+lipidz[37*i+18]+lipidz[37*i+19]+lipidz[37*i+20]+lipidz[37*i+21]+lipidz[37*i+22])/14;

		end1[i][0]=lipidx[3*i+1];
		end1[i][1]=lipidy[3*i+1];
		end1[i][2]=lipidz[3*i+1];

		//end2[i][0]=(lipidx[37*i+23]+lipidx[37*i+24]+lipidx[37*i+25]+lipidx[37*i+26]+lipidx[37*i+27]+lipidx[37*i+28]+lipidx[37*i+29]+lipidx[37*i+30]+lipidx[37*i+31]+lipidx[37*i+32]+lipidx[37*i+33]+lipidx[37*i+34]+lipidx[37*i+35]+lipidx[37*i+36])/14;
		//end2[i][1]=(lipidy[37*i+23]+lipidy[37*i+24]+lipidy[37*i+25]+lipidy[37*i+26]+lipidy[37*i+27]+lipidy[37*i+28]+lipidy[37*i+29]+lipidy[37*i+30]+lipidy[37*i+31]+lipidy[37*i+32]+lipidy[37*i+33]+lipidy[37*i+34]+lipidy[37*i+35]+lipidy[37*i+36])/14;
		//end2[i][2]=(lipidz[37*i+23]+lipidz[37*i+24]+lipidz[37*i+25]+lipidz[37*i+26]+lipidz[37*i+27]+lipidz[37*i+28]+lipidz[37*i+29]+lipidz[37*i+30]+lipidz[37*i+31]+lipidz[37*i+32]+lipidz[37*i+33]+lipidz[37*i+34]+lipidz[37*i+35]+lipidz[37*i+36])/14;

		end2[i][0]=lipidx[3*i+2];
		end2[i][1]=lipidy[3*i+2];
		end2[i][2]=lipidz[3*i+2];


		// ensure that the head coordinates are wrapped

//		cout << head[i][0]<< " "<< head[i][1]<< " "<< head[i][2]<< endl;
//		cout << end1[i][0]<< " "<< end1[i][1]<< " "<< end1[i][2]<< endl;
//		cout << end2[i][0]<< " "<< end2[i][1]<< " "<< end2[i][2]<< endl;
//		cout << endl;

		int counter=0;

		 // make sure the interface beads are within the box in terms of xy

		if(head[i][0] > lx[frame_num] || head[i][0] == lx[frame_num]){

//			cout << " x>= lx= " << lx[frame_num] << endl;
			head[i][0] = head[i][0] - lx[frame_num];
			end1[i][0] = end1[i][0] - lx[frame_num];
			end2[i][0] = end2[i][0] - lx[frame_num];
//			cout << " head fixed " << head[i][0] << " "  << head[i][1] << " "  << head[i][2] << endl;

			counter++;
			}

		if(head[i][1] > ly[frame_num] || head[i][1] == ly[frame_num]){
//			cout << " y>= ly= " << ly[frame_num] << endl;
//			cout << head[i][0] << " "  << head[i][1] << " "  << head[i][2] << endl;
			head[i][1] = head[i][1] - ly[frame_num];
			end1[i][1] = end1[i][1] - ly[frame_num];
			end2[i][1] = end2[i][1] - ly[frame_num];
//			cout << " head fixed " << head[i][0] << " "  << head[i][1] << " "  << head[i][2] << endl;
			counter++;
			}
		 // if head < 0
		if(head[i][0] < 0){
//			cout << " x<0 " << lx[frame_num] << endl;
//			cout << head[i][0] << " "  << head[i][1] << " "  << head[i][2] << endl;
			head[i][0] = head[i][0] + lx[frame_num];
			end1[i][0] = end1[i][0] + lx[frame_num];
			end2[i][0] = end2[i][0] + lx[frame_num];
//			cout << " head fixed " << head[i][0] << " "  << head[i][1] << " "  << head[i][2] << endl;
			counter++;
			}

		if(head[i][1] < 0){
//			cout << " y<0 " << ly[frame_num] << endl;
//			cout << head[i][0] << " "  << head[i][1] << " "  << head[i][2] << endl;
			head[i][1] = head[i][1] + ly[frame_num];
			end1[i][1] = end1[i][1] + ly[frame_num];
			end2[i][1] = end2[i][1] + ly[frame_num];
//			cout << " head fixed " << head[i][0] << " "  << head[i][1] << " "  << head[i][2] << endl;
			counter++;
			}

		//fix the tail beads which were carried to the other side of the box

		if(fabs(head[i][0]-end1[i][0])>0.5*lx_av){

// 			cout << "fabs(head[i][0]-end1[i][0])>0.5*lx_av  " << end1[i][0] << " " << head[i][0] << " " <<i ;
			end1[i][0]=(head[i][0]>end1[i][0] ? end1[i][0]+lx[frame_num] : end1[i][0] -lx[frame_num] );
// 			cout << "  " << end1[i][0] << endl;
		}

		if(fabs(head[i][0]-end2[i][0])>0.5*lx_av){

// 			cout << "fabs(head[i][0]-end2[i][0])>0.5*lx_av  " << end2[i][0] << " " << head[i][0] << " " <<i ;
			end2[i][0]=(head[i][0]>end2[i][0] ? end2[i][0]+lx[frame_num] : end2[i][0] -lx[frame_num] );
// 			cout << "  " << end2[i][0] << endl;
		}

		if(fabs(head[i][1]-end1[i][1])>0.5*lx_av){
// 			cout << "fabs(head[i][0]-end1[i][1])>0.5*lx_av  " << end1[i][1] << " " << head[i][1] << " " <<i ;
			end1[i][1]=(head[i][1]>end1[i][1] ? end1[i][1]+ly[frame_num] : end1[i][1] -ly[frame_num] );
// 			cout << "  " << end1[i][1] << endl;
		}

		if(fabs(head[i][1]-end2[i][1])>0.5*lx_av){
// 			cout << "fabs(head[i][0]-end2[i][1])>0.5*lx_av  " << end2[i][1] << " " << head[i][1] << " " <<i ;
			end2[i][1]=(head[i][1]>end2[i][1] ? end2[i][1]+ly[frame_num] : end2[i][1] -ly[frame_num] );
// 			cout << "  " << end2[i][1] << endl;
		}

//		if(counter>0){
//		cout << endl;
//		cout << head[i][0]<< " "<< head[i][1]<< " "<< head[i][2]<< endl;
//		cout << end1[i][0]<< " "<< end1[i][1]<< " "<< end1[i][2]<< endl;
//		cout << end2[i][0]<< " "<< end2[i][1]<< " "<< end2[i][2]<< endl;
//		cout << " ------- "<< endl;}


		//director fields
		dir[i][0]=0.5*(end1[i][0]+end2[i][0])-head[i][0];
		dir[i][1]=0.5*(end1[i][1]+end2[i][1])-head[i][1];
		dir[i][2]=0.5*(end1[i][2]+end2[i][2])-head[i][2];

		mag=1.0/sqrt(dir[i][0]*dir[i][0] + dir[i][1]*dir[i][1] + dir[i][2]*dir[i][2]);

		dir[i][0] *= mag;
		dir[i][1] *= mag; // normalize the director
		dir[i][2] *= mag;

                dir[i][0] /= fabs(dir[i][2]); // ADDED BY ZACH
                dir[i][1] /= fabs(dir[i][2]); // ADDED BY ZACH
                dir[i][2] /= fabs(dir[i][2]); // ADDED BY ZACH
//               cout << dir[i][2] << endl; // ADDED BY ZACH


		if( fabs(dir[i][2]) > cutang ){good[i]=1;} // if the lipid is within the cutoff angle
		else{
// 			cout << "_______"<< endl;
// 			cout << head[i][0]<< " "<< head[i][1]<< " "<< head[i][2]<< endl;
// 			cout << end1[i][0]<< " "<< end1[i][1]<< " "<< end1[i][2]<< endl;
// 			cout << end2[i][0]<< " "<< end2[i][1]<< " "<< end2[i][2]<< endl;
// 			cout <<dir[i][0]<< " "<< dir[i][1]<< " "<< dir[i][2]<< endl;
// 			cout << "_______"<< endl;
// 			cout << "bad lipid at "<< floor(head[i][0]/dlx)<< " "<< floor(head[i][1]/dly)<< endl;
			good[i]=0;
		}

		if(dir[i][2]<0){
			z1avg += head[i][2];
			nl1++;
		}

		if(dir[i][2]>0){
			z2avg += head[i][2];
			nl2++;
		}

	} // loop over nl

// 	zavg[frame_num]=0.5*(z1avg/nl1 + z2avg/nl2);

	// check if molecules were carried to other size in z

	for(i=0; i<nl; i++){

		if(dir[i][2]<0 && fabs(head[i][2]-z1avg/nl1)>20){ // stray molecule belongs in top monolayer

// 			cout << "mol "<<i<<" headz= "<<head[i][2]<<" z1avg= "<<z1avg/nl1<<" lz= "<<lz[frame_num];

			head[i][2] += lz[frame_num];
			end1[i][2] += lz[frame_num];
			end2[i][2] += lz[frame_num];

// 			cout<<" fixed: "<<head[i][2]<<endl;
		}

		if(dir[i][2]>0 && fabs(head[i][2]-z2avg/nl2)>20){ // stray molecule belongs in bottom monolayer

// 			cout << "mol "<<i<<" headz= "<<head[i][2]<<" z2avg= "<<z2avg/nl1<<" lz= "<<lz[frame_num];

			head[i][2] -= lz[frame_num];
			end1[i][2] -= lz[frame_num];
			end2[i][2] -= lz[frame_num];

// 			cout<<" fixed: "<<head[i][2]<<endl;
		}

	}

	for(i=0; i<nl; i++){ // calculate zavg after the lipids have been fixed

		zavg[frame_num] += head[i][2];
	}

	zavg[frame_num] /= nl;

	phi0_frame= 0.5*(nl1+nl2)/lx[frame_num]/ly[frame_num];
	phi0 += phi0_frame;
//----------------------------------------------------------------------------------------------
//CALCULATE NUMBER DENSITIES////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------
		// find phi_q, which is the sum over e^(iqx)
		if(AREA){

		for(i=0; i<nl; i++){
			for(j=0; j<N/2+1; j++){ // only take the upper half of the complex plane
				for(k=0; k<N; k++){// and use symmetries later

					qx = twoPi*q[j][k][0]*invLx;
					qy = twoPi*q[j][k][1]*invLy;
					// same as qmat but divided by the appropriate L

					if(AREA_tail){

						xx=0.5*(end1[i][0]+end2[i][0]);
						yy=0.5*(end1[i][1]+end2[i][1]);
					}
					else{
						xx=head[i][0];
						yy=head[i][1];
					}

					h_real[j][k] += (head[i][2]-zavg[frame_num])*cos(qx*xx + qy*yy);
					h_imag[j][k] -= (head[i][2]-zavg[frame_num])*sin(qx*xx + qy*yy);

					if(dir[i][2]<0 && good[i]){ // upper monolayer

						if(j==0 && k==0){ // treat q=0 separately
							psiRU[j][k] += 0;
							psiIU[j][k] += 0;}

						else{		psiRU[j][k] += cos(qx*xx + qy*yy);  // divide by phi0in at the end
								psiIU[j][k] -= sin(qx*xx + qy*yy);}
						}

					if(dir[i][2]>0 && good[i]){ // lower monolayer

						if(j==0 && k==0){ // treat q=0 separately for real part
							psiRD[j][k] += 0;
							psiID[j][k] += 0;}

						else{		psiRD[j][k] += cos(qx*xx + qy*yy);
								psiID[j][k] -= sin(qx*xx + qy*yy);}
						}

				}// 2 for loops
			} // 2 for loops

		} // loop over nl

	}  // if(AREA)
//
	if(AREA){
		for(j=0; j<N; j++){ // multiply by 1/L for correct dimensions
			for(k=0; k<N; k++){ // and take care of q=0 mode

				if(j==0 && k==0){
					psiRU[j][k]=nl1*invLxy - phi0in*Lxy; // (1/L) integral (phi-phi0in) dx dy
					psiIU[j][k]=0;
					psiRD[j][k]=nl2*invLxy - phi0in*Lxy; // =nl1/L-phi0in*L
					psiID[j][k]=0;
				}
				else{
				psiRU[j][k] *= invLxy;	psiIU[j][k] *= invLxy; // for dimensions in Edholm paper
				psiRD[j][k] *= invLxy;	psiID[j][k] *= invLxy;} // didn't worry about q=0
			}						// was *= invLxy for Seifert
		}
	} // if (AREA)

//----------------------------------------------------------------------------------------------
//HEIGHT & THICKNESS////////////////////////////////////////////////////////////////////////////
//----------------------------------------------------------------------------------------------

////////assign the lipids to coarse-grained fields
////////and calculate average height and thickness

	for(i=0; i<nl; i++){

		xj[i]= (int) floor(head[i][0]/dlx);
		yj[i]= (int) floor(head[i][1]/dly);

		if(xj[i]>N-1){

			if(head[i][0]==lx[frame_num]){xj[i]=N-1;} // this got through the wrapping filter because -1e-14<x<0
							// and x+lx is stored as lx
			if(head[i][0]!=lx[frame_num]){
				cout<<" xi>N-1 -> xi=" <<xj[i]<<" for x= " <<head[i][0]<<" lx= "<<lx[frame_num]<<" i= "<<i<<endl;
			}
		}

		if(yj[i]>N-1){

			if(head[i][1]==ly[frame_num]){yj[i]=N-1;} 	// this got through the wrapping filter because -1e-14<y<0
							// and y+ly is stored as ly
			if(head[i][0]!=ly[frame_num]){
				cout<<" yi>N-1 -> yi=" <<yj[i]<<" N= "<<N<<" for y= " <<head[i][1]<<" ly= "<<ly[frame_num]<<" i= "<<i<<endl;
			}
		}

		if(xj[i]<0){cout<<" xi<0 -> xi= "<< xj[i] <<" for x= "<< head[i][0] << " i= " << i <<endl;}
		if(yj[i]<0){cout<<" yi<0 -> yi= "<< yj[i] <<" for y= "<< head[i][1] << " i= " << i <<endl;}

		xi=xj[i];
		yi=yj[i];

		if(dir[i][2] < 0){  // upper monolayer
			if(good[i]){
				z1[xi][yi] += head[i][2]-zavg[frame_num];
				z1sq_av_frame += (head[i][2]-zavg[frame_num])*(head[i][2]-zavg[frame_num]);
// 				if(frame_num==183 && xi==20 && yi==15){cout<< "z1--> " <<xi<< " "<< yi << " "<<head[i][2] <<z1[xi][yi] << endl;}
				nlg1[xi][yi]++;
			}
			else{nlb1[xi][yi]++;}
		}


		if(dir[i][2] > 0){ //lower monolayer
			if(good[i]){
				z2[xi][yi] += head[i][2]-zavg[frame_num];
				z2sq_av_frame += (head[i][2]-zavg[frame_num])*(head[i][2]-zavg[frame_num]);
// 				if(frame_num==183 && xi==20 && yi==15){cout<<"z2-->" <<xi<< " "<< yi << " "<<head[i][2] <<z2[xi][yi] << endl;}
				nlg2[xi][yi]++;
			}
			else{nlb2[xi][yi]++;}
		}

	} // loop over nl

	z1sq_av_frame /=nl1;
	z2sq_av_frame /=nl2;

	z1sq_av += z1sq_av_frame;
	z2sq_av += z2sq_av_frame;

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			if(nlg1[i][j]>0){z1[i][j] /= nlg1[i][j];}
			if(nlg2[i][j]>0){z2[i][j] /= nlg2[i][j];}
		}
	}
	/////if a patch is empty, interpolate
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			i1 = ((i>0) ? (i-1) : (N-1));  i2 = ((i<N-1) ? (i+1) : 0); // periodic boundaries
			j1 = ((j>0) ? (j-1) : (N-1));  j2 = ((j<N-1) ? (j+1) : 0);

			if(nlg1[i][j]==0){

				if(nlg1[i1][j]==0 || nlg1[i2][j]==0 || nlg1[i][j1]==0 || nlg1[i][j2]==0 ){
				empty++;
				empty_tot++;}

				int nn= nlg1[i][j1] + nlg1[i][j2] + nlg1[i1][j] + nlg1[i2][j];

				z1[i][j] = (nlg1[i][j1]*z1[i][j1] + nlg1[i][j2]*z1[i][j2]
					+ nlg1[i1][j]*z1[i1][j] + nlg1[i2][j]*z1[i2][j])/nn;
			}

			if(nlg2[i][j]==0){

				if(nlg1[i1][j]==0 || nlg1[i2][j]==0 || nlg1[i][j1]==0 || nlg1[i][j2]==0 ){
				empty++;
				empty_tot++;}

				int nn= nlg2[i][j1] + nlg2[i][j2] + nlg2[i1][j] + nlg2[i2][j];

				z2[i][j] = (nlg2[i][j1]*z2[i][j1] + nlg2[i][j2]*z2[i][j2]
					+ nlg2[i1][j]*z2[i1][j] + nlg2[i2][j]*z2[i2][j])/nn;
			}


		}
	} // two for loops over (i,j)

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			h[i][j]=z1[i][j]+z2[i][j];
			t[i][j]=z1[i][j]-z2[i][j];

	/////////////calculate average thickness quantities

			t0_frame += t[i][j];
			tq0_frame += (t[i][j]-2*t0in);

		}
	}  // two for loops over (i,j)

	t0_frame = 0.5*t0_frame/(N*N);

	tq0_frame = lx[frame_num]*tq0_frame; // multiply by .25 at the end

	tq0_frame *=tq0_frame;

	t0 += t0_frame;

	tq0 += tq0_frame;
//----------------------------------------------------------------------------------------------
//CALCULATE NORMAL VECTORS//////////////////////////////////////////////////////////////////////
//----------------------------------------------------------------------------------------------
if(TILT){

	for(i=0; i<N; i++){
		for(j=0; j<N; j++) {

			z1_1D[i*N+j]=z1[i][j];
			z2_1D[i*N+j]=z2[i][j];
		}
	}

	fftwf_execute_dft_r2c(spectrum_plan, z1_1D, z1qS);
	fftwf_execute_dft_r2c(spectrum_plan, z2_1D, z2qS);

  //set wave vector: (2\pi/L){0, 1,..., N/2-1, -N/2,..., -1}
  //                  index: (0, 1,..., N/2-1,  N/2,... ,N-1}

	for(i=0; i<N; i++) {
		for( j=0; j<N/2+1; j++) {

//			qi = ((i < N/2) ? i : i-N);
//			qj = ((j < N/2) ? j : j-N);

			qi=q[i][j][0];
			qj=q[i][j][1];

			k = i*(N/2+1) + j;
			// handle Nyquist element (aliased between -N/2 and N/2
			// => set the N/2 term of the derivative to 0)

			if(i==N/2) {qi=0;}
			if(j==N/2) {qj=0;}

			dz1xqS[k][0] = -qi*z1qS[k][1]*twoPiLx;    dz1xqS[k][1] =  qi*z1qS[k][0]*twoPiLx;
			dz1yqS[k][0] = -qj*z1qS[k][1]*twoPiLy;    dz1yqS[k][1] =  qj*z1qS[k][0]*twoPiLy;
        //
			dz2xqS[k][0] = -qi*z2qS[k][1]*twoPiLx;    dz2xqS[k][1] =  qi*z2qS[k][0]*twoPiLx;
			dz2yqS[k][0] = -qj*z2qS[k][1]*twoPiLy;    dz2yqS[k][1] =  qj*z2qS[k][0]*twoPiLy;
		}
	}

    // backward transform to get derivatives in real space
    fftwf_execute_dft_c2r(inv_plan, dz1xqS, dz1x1D);
    fftwf_execute_dft_c2r(inv_plan, dz1yqS, dz1y1D);
    fftwf_execute_dft_c2r(inv_plan, dz2xqS, dz2x1D);
    fftwf_execute_dft_c2r(inv_plan, dz2yqS, dz2y1D);

    // normalize: f = (1/L) Sum f_q exp(iq.r)
    for(i=0; i<N; i++) {
		for(j=0; j<N; j++) {

			k = i*N + j;

			dz1x1D[k] *= invLx; dz1y1D[k] *= invLy;
			dz2x1D[k] *= invLx; dz2y1D[k] *= invLy;

			root_ginv1=1.0/sqrt(1.0 + dz1x1D[k]*dz1x1D[k] + dz1y1D[k]*dz1y1D[k]);
			root_ginv2=1.0/sqrt(1.0 + dz2x1D[k]*dz2x1D[k] + dz2y1D[k]*dz2y1D[k]);

// 			root_ginv1=1.0;
// 			root_ginv2=1.0;

			norm_1[k][0]= dz1x1D[k]*root_ginv1;
			norm_1[k][1]= dz1y1D[k]*root_ginv1;
			norm_1[k][2]= -root_ginv1;

			norm_2[k][0]= -dz2x1D[k]*root_ginv2;
			norm_2[k][1]= -dz2y1D[k]*root_ginv2; // signs are reversed
			norm_2[k][2]= root_ginv2;

 norm_1[k][0] /= fabs(norm_1[k][2]); // ADDED BY ZACH
 norm_1[k][1] /= fabs(norm_1[k][2]); // ADDED BY ZACH
 norm_1[k][2] /= fabs(norm_1[k][2]); // ADDED BY ZACH

 norm_2[k][0] /= fabs(norm_2[k][2]); // ADDED BY ZACH
 norm_2[k][1] /= fabs(norm_2[k][2]); // ADDED BY ZACH
 norm_2[k][2] /= fabs(norm_2[k][2]); // ADDED BY ZACH
		}

	}
//----------------------------------------------------------------------------------------------
//CALCULATE TILT VECTORS////////////////////////////////////////////////////////////////////////////
//----------------------------------------------------------------------------------------------

	for(i=0; i<nl; i++) {

		xi= xj[i];
		yi= yj[i];

		k = xi*N + yi;

		// tilt vector m = n/(n.N) - N

		if(dir[i][2] < 0) { // upper monolayer

			dot1=dir[i][0]*norm_1[k][0] + dir[i][1]*norm_1[k][1] + dir[i][2]*norm_1[k][2];

			dot_cum += dot1;


// 			if( fabs(dir[i][2]*norm_1[k][2]) > cutang){
// 			if( dot1>fabs(cutang)){

				denom1=1.0/dot1;

				nlt1[xi][yi]++;
				nt1++;
				//accumulate

				for(j=0; j<3; j++){

					t1mol[j]=dir[i][j] - norm_1[k][j]; //no denom

					t1[xi][yi][j] += t1mol[j];
				}

				n1[xi][yi][0] += dir[i][0];
				n1[xi][yi][1] += dir[i][1];

				rootgxinv=1.0/sqrt(1 + dz1x1D[k]*dz1x1D[k]);

				u[0]=rootgxinv;
				u[1]=0;
				u[2]=dz1x1D[k]*rootgxinv;

				v[0]=   u[1]*norm_1[k][2] - u[2]*norm_1[k][1];
				v[1]= -(u[0]*norm_1[k][2] - u[2]*norm_1[k][0]);
				v[2]=   u[0]*norm_1[k][1] - u[1]*norm_1[k][0];
//
				tmag=t1mol[0]*t1mol[0] + t1mol[1]*t1mol[1] + t1mol[2]*t1mol[2];

// 				cout<< (tproj2[0]*tproj2[0] + tproj2[1]*tproj2[1] + tproj2[2]*tproj2[2] + \
// 				tproj1[0]*tproj1[0] + tproj1[1]*tproj1[1] + tproj1[2]*tproj1[2])-tmag<<endl;

// 				histn=(int)floor((180/pi)*(100/90)*atan(sqrt(tmag)));

				ut=0; vt=0;

				for(j=0; j<3; j++){

					ut += t1mol[j] * u[j];
					vt += t1mol[j] * v[j];
				}

				if(abs(ut)<5){ tproj1_cum[ (int)floor(20*abs(ut)) ]++; }
				if(abs(vt)<5){ tproj2_cum[ (int)floor(20*abs(vt)) ]++; }

// 				tproj2_cum[frame_num] += v[0]*t1mol[0] + v[1]*t1mol[1] + v[2]*t1mol[2];

				if(sqrt(tmag)<1){
					ty_cum[ (int)floor(100*abs(sqrt(tmag))) ]++;
				}

//
				if(abs(t1mol[0])<1 && abs(t1mol[1])<1){
					hist_t[(int)floor(100*abs(t1mol[0]))][(int)floor(100*abs(t1mol[1]))]++;
				}

				if(abs(t1mol[0])<1){
					hist_t2[ (int) floor(100*abs(t1mol[0]))]++;
				}
//
// 				hist_theta[histn]++;

// 			} if( dot1>fabs(cutang)
		}


		if(dir[i][2] > 0) { // lower monolayer

			dot2=dir[i][0]*norm_2[k][0] + dir[i][1]*norm_2[k][1] + dir[i][2]*norm_2[k][2];

			dot_cum += dot2;

// 			if(dot2>fabs(cutang)){

				denom2=1.0/dot2;
// 				denom2=1.0;
// 				denom2=1/nNav;

				nlt2[xi][yi]++;
				nt2++;
				//accumulate

				for(j=0; j<3; j++){

					t2mol[j]=dir[i][j] - norm_2[k][j]; // no denom

					t2[xi][yi][j] += t2mol[j];
				}

				n2[xi][yi][0] += dir[i][0];
				n2[xi][yi][1] += dir[i][1];
// 			} if(dot2>fabs(cutang))
		}

	} // nl loop

// 	if(nt1 != nl/2 || nt2 != nl/2){goto there;}

	//average over each patch
	for(i=0; i<N; i++) {
		for(j=0; j<N; j++) {

			if(nlt1[i][j]>0) {t1[i][j][0] /= nlt1[i][j]; t1[i][j][1] /= nlt1[i][j];
					  n1[i][j][0] /= nlt1[i][j]; n1[i][j][1] /= nlt1[i][j];}

			if(nlt2[i][j]>0) {t2[i][j][0] /= nlt2[i][j]; t2[i][j][1] /= nlt2[i][j];
					  n2[i][j][0] /= nlt2[i][j]; n2[i][j][1] /= nlt2[i][j];}
		}
    }


// 	if(frame_num==36){
//
// 	int a,b;
// 		for(a=0; a<N; a++) {
// 			for(b=0; b<N; b++) {
//
// 				if(fabs(t1xR_cum[a][b]/frame_num)> 6.0){
// 					for(i=0; i<N; i++){
// 						for(j=0; j<N; j++){
//
// // 							cout << (t1xR_cum[i][j])/frame_num << " " ;
// 							cout << t1[i][j][0] << " " ;
// 						}
// 						cout << endl;
// 					}
// 				cout << "bad seed found at"<<a<<" "<<b<<endl;
// 				exit(-1);
// 				}
// -----------------------------------------------
// 			}
// 		}
// 	}

	///////////  if a patch is empty, interpolate
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			i1 = ((i>0) ? (i-1) : (N-1));  i2 = ((i<N-1) ? (i+1) : 0); // periodic boundaries
			j1 = ((j>0) ? (j-1) : (N-1));  j2 = ((j<N-1) ? (j+1) : 0);

			for(k=0; k<2; k++){ //  loop over {0,1}

				if(nlt1[i][j]==0){
					nn= 1.0/(nlt1[i][j1] + nlt1[i][j2] + nlt1[i1][j] + nlt1[i2][j]);

					t1[i][j][k] = (nlt1[i][j1]*t1[i][j1][k] + nlt1[i][j2]*t1[i][j2][k]
						+ nlt1[i1][j]*t1[i1][j][k] + nlt1[i2][j]*t1[i2][j][k])*nn;

					n1[i][j][k] = (nlt1[i][j1]*n1[i][j1][k] + nlt1[i][j2]*n1[i][j2][k]
						+ nlt1[i1][j]*n1[i1][j][k] + nlt1[i2][j]*n1[i2][j][k])*nn;
				}

				if(nlt2[i][j]==0){
					nn= 1.0/(nlt2[i][j1] + nlt2[i][j2] + nlt2[i1][j] + nlt2[i2][j]);

					t2[i][j][k] = (nlt2[i][j1]*t2[i][j1][k] + nlt2[i][j2]*t2[i][j2][k]
						+ nlt2[i1][j]*t2[i1][j][k] + nlt2[i2][j]*t2[i2][j][k])*nn;

					n2[i][j][k] = (nlt2[i][j1]*n2[i][j1][k] + nlt2[i][j2]*n2[i][j2][k]
						+ nlt2[i1][j]*n2[i1][j][k] + nlt2[i2][j]*n2[i2][j][k])*nn;
				}

			}
		} // interpolation
	}  	// loops

	for(i=0; i<N; i++) {
		for(j=0; j<N; j++) {

			// accumulate real space orientations
			t1xR_cum[i][j] += nlg1[i][j];
			t1xI_cum[i][j] += nlg2[i][j];
			t1yR_cum[i][j] += n1[i][j][0] - n2[i][j][0];
			t1yI_cum[i][j] += n1[i][j][1] - n2[i][j][1];

			if(abs(t1[i][j][0])<5){
			tghist[(int) floor(20*abs(t1[i][j][0])) ]++;
			}

			dp[i][j][0] = t1[i][j][0] + t2[i][j][0]; // m1 + m2
			dp[i][j][1] = t1[i][j][1] + t2[i][j][1];
			dm[i][j][0] = t1[i][j][0] - t2[i][j][0]; // m1 - m2
			dm[i][j][1] = t1[i][j][1] - t2[i][j][1]; // factors of two are added at the end

			up[i][j][0] = n1[i][j][0] + n2[i][j][0];
			up[i][j][1] = n1[i][j][1] + n2[i][j][1];
			um[i][j][0] = n1[i][j][0] - n2[i][j][0];
			um[i][j][1] = n1[i][j][1] - n2[i][j][1]; // factors of two are added at the end
		}
    }

} // if TILT

//----------------------------------------------------------------------------------------------
//ACCUMULATE SPECTRA////////////////////////////////////////////////////////////////////////////
//----------------------------------------------------------------------------------------------

	for(i=0; i<N; i++){
		for(j=0; j<N; j++) {

			h1D[i*N+j]=h[i][j];
			t1D[i*N+j]=t[i][j];

			if(TILT){

			t1x1D[i*N+j]=t1[i][j][0];
			t1y1D[i*N+j]=t1[i][j][1];

			dpx1D[i*N+j]=dp[i][j][0];
			dpy1D[i*N+j]=dp[i][j][1];
			dmx1D[i*N+j]=dm[i][j][0];
			dmy1D[i*N+j]=dm[i][j][1];

			upx1D[i*N+j]=up[i][j][0];
			upy1D[i*N+j]=up[i][j][1];
			umx1D[i*N+j]=um[i][j][0];
			umy1D[i*N+j]=um[i][j][1];}
		}
	}

	fftwf_execute_dft_r2c(spectrum_plan, h1D, hqS);
	fftwf_execute_dft_r2c(spectrum_plan, t1D, tqS);

	fullArray(hqR,hqI,hqS,Lxy); // multiply by lx/N^2 factor inside
	fullArray(tqR,tqI,tqS,Lxy);

	if(AREA){ // similar to 'fullArray,' use h_{-q}=h*_q to fill in the lower half of the complex plane

		for(i=1; i<N/2; i++){
			for(j=0; j<N; j++){

				psiRU[N-i][j]=psiRU[i][j];		psiIU[N-i][j]= -psiIU[i][j];
				psiRD[N-i][j]=psiRD[i][j];		psiID[N-i][j]= -psiID[i][j];
				h_real[N-i][j]=h_real[i][j];		h_imag[N-i][j]=h_imag[i][j];
			}
		}
	} // if(AREA)

	if(TILT){
		fftwf_execute_dft_r2c(spectrum_plan, t1x1D, t1xqS);
		fftwf_execute_dft_r2c(spectrum_plan, t1y1D, t1yqS);

		fftwf_execute_dft_r2c(spectrum_plan, dpx1D, dpxqS);
		fftwf_execute_dft_r2c(spectrum_plan, dpy1D, dpyqS);
		fftwf_execute_dft_r2c(spectrum_plan, dmx1D, dmxqS);
		fftwf_execute_dft_r2c(spectrum_plan, dmy1D, dmyqS);

		fftwf_execute_dft_r2c(spectrum_plan, upx1D, upxqS);
		fftwf_execute_dft_r2c(spectrum_plan, upy1D, upyqS);
		fftwf_execute_dft_r2c(spectrum_plan, umx1D, umxqS);
		fftwf_execute_dft_r2c(spectrum_plan, umy1D, umyqS);

		fullArray(t1xR,t1xI,t1xqS,Lxy);
		fullArray(t1yR,t1yI,t1yqS,Lxy);

		fullArray(dpxR,dpxI,dpxqS,Lxy);
		fullArray(dpyR,dpyI,dpyqS,Lxy);
		fullArray(dmxR,dmxI,dmxqS,Lxy);
		fullArray(dmyR,dmyI,dmyqS,Lxy);

		fullArray(upxR,upxI,upxqS,Lxy);
		fullArray(upyR,upyI,upyqS,Lxy);
		fullArray(umxR,umxI,umxqS,Lxy);
		fullArray(umyR,umyI,umyqS,Lxy);

	////// decompose into parallel and perpendicular components

		for(i=0; i<N; i++){
			for(j=0; j<N; j++){

				if(i==0 && j==0){ // the perp and par components are not defined at q=0

					/*dmparR[i][j] = (dmxR[i][j] + dmyR[i][j]);
					dmperR[i][j] = (dmxR[i][j] + dmyR[i][j]);

					dpparR[i][j] = (dpxR[i][j] + dpyR[i][j]);
					dpperR[i][j] = (dpxR[i][j] + dpyR[i][j]);*/

					dmparR[i][j] = 0.0; dmperR[i][j] = 0.0;
					dpparR[i][j] = 0.0; dpperR[i][j] = 0.0;
					dmparI[i][j] = 0.0;	dmperI[i][j] = 0.0;
					dpparI[i][j] = 0.0;	dpperI[i][j] = 0.0;

					/* umparR[i][j] = (umxR[i][j] + umyR[i][j]);
					umperR[i][j] = (umxR[i][j] + umyR[i][j]);

					upparR[i][j] = (upxR[i][j] + upyR[i][j]);
					upperR[i][j] = (upxR[i][j] + upyR[i][j]); */

					umparR[i][j] = 0.0; umperR[i][j] = 0.0;
					upparR[i][j] = 0.0; upperR[i][j] = 0.0;
					umparI[i][j] = 0.0;	umperI[i][j] = 0.0;
					upparI[i][j] = 0.0;	upperI[i][j] = 0.0;
				}
				else{
					dmparR[i][j]=  dmxR[i][j]*cosq[i][j] + dmyR[i][j]*sinq[i][j];
					dmperR[i][j]= -dmxR[i][j]*sinq[i][j] + dmyR[i][j]*cosq[i][j];
					dmparI[i][j]=  dmxI[i][j]*cosq[i][j] + dmyI[i][j]*sinq[i][j];
					dmperI[i][j]= -dmxI[i][j]*sinq[i][j] + dmyI[i][j]*cosq[i][j];

					dpparR[i][j]=  dpxR[i][j]*cosq[i][j] + dpyR[i][j]*sinq[i][j];
					dpperR[i][j]= -dpxR[i][j]*sinq[i][j] + dpyR[i][j]*cosq[i][j];
					dpparI[i][j]=  dpxI[i][j]*cosq[i][j] + dpyI[i][j]*sinq[i][j];
					dpperI[i][j]= -dpxI[i][j]*sinq[i][j] + dpyI[i][j]*cosq[i][j];

					umparR[i][j]=  umxR[i][j]*cosq[i][j] + umyR[i][j]*sinq[i][j];
					umperR[i][j]= -umxR[i][j]*sinq[i][j] + umyR[i][j]*cosq[i][j];
					umparI[i][j]=  umxI[i][j]*cosq[i][j] + umyI[i][j]*sinq[i][j];
					umperI[i][j]= -umxI[i][j]*sinq[i][j] + umyI[i][j]*cosq[i][j];

					upparR[i][j]=  upxR[i][j]*cosq[i][j] + upyR[i][j]*sinq[i][j];
					upperR[i][j]= -upxR[i][j]*sinq[i][j] + upyR[i][j]*cosq[i][j];
					upparI[i][j]=  upxI[i][j]*cosq[i][j] + upyI[i][j]*sinq[i][j];
					upperI[i][j]= -upxI[i][j]*sinq[i][j] + upyI[i][j]*cosq[i][j];
					}
			}
		}

	} // if (TILT)

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			hq2[i][j] += hqR[i][j]*hqR[i][j] + hqI[i][j]*hqI[i][j];
			tq2[i][j] += tqR[i][j]*tqR[i][j] + tqI[i][j]*tqI[i][j];

			hq4[i][j] += pow(hqR[i][j]*hqR[i][j] + hqI[i][j]*hqI[i][j],2); // for variance calculation

			if(TILT){

				t1xq2[i][j] += t1xR[i][j]*t1xR[i][j] + t1xI[i][j]*t1xI[i][j];
				t1yq2[i][j] += t1yR[i][j]*t1yR[i][j] + t1yI[i][j]*t1yI[i][j];

// 				txq[frame_num]=t1xR[0][0];
// 			txq[frame_num]=dpxR[0][0]*dpxR[0][0] + dpxI[0][0]*dpxI[0][0] + dpyR[0][0]*dpyR[0][0] + dpyI[0][0]*dpyI[0][0];

				dpq2[i][j] += dpxR[i][j]*dpxR[i][j] + dpxI[i][j]*dpxI[i][j] + dpyR[i][j]*dpyR[i][j] + dpyI[i][j]*dpyI[i][j];
				dmq2[i][j] += dmxR[i][j]*dmxR[i][j] + dmxI[i][j]*dmxI[i][j] + dmyR[i][j]*dmyR[i][j] + dmyI[i][j]*dmyI[i][j];

				dpparq2[i][j] += dpparR[i][j]*dpparR[i][j] + dpparI[i][j]*dpparI[i][j];
				dpperq2[i][j] += dpperR[i][j]*dpperR[i][j] + dpperI[i][j]*dpperI[i][j];

				dmparq2[i][j] += dmparR[i][j]*dmparR[i][j] + dmparI[i][j]*dmparI[i][j];
				dmperq2[i][j] += dmperR[i][j]*dmperR[i][j] + dmperI[i][j]*dmperI[i][j];

				hdmpar[i][j] += -dmparI[i][j]*hqR[i][j] + dmparR[i][j]*hqI[i][j];
				tdppar[i][j] += -dpparI[i][j]*tqR[i][j] + dpparR[i][j]*tqI[i][j];

// 				hdmpar[i][j] += -umparI[i][j]*hqR[i][j] + umparR[i][j]*hqI[i][j]; // Im(h(q) dm(-q))
// 				tdppar[i][j] += -upparI[i][j]*tqR[i][j] + upparR[i][j]*tqI[i][j]; // Im(t(q) dp(-q))

				// these are the imaginary components of the cross correlations
				// I checked that the real parts are virtually zero

				upparq2[i][j] += upparR[i][j]*upparR[i][j] + upparI[i][j]*upparI[i][j];
				upperq2[i][j] += upperR[i][j]*upperR[i][j] + upperI[i][j]*upperI[i][j];

				umparq2[i][j] += umparR[i][j]*umparR[i][j] + umparI[i][j]*umparI[i][j];
				umperq2[i][j] += umperR[i][j]*umperR[i][j] + umperI[i][j]*umperI[i][j];

				umparq4[i][j] += pow(umparR[i][j]*umparR[i][j] + umparI[i][j]*umparI[i][j],2);
				umperq4[i][j] += pow(umperR[i][j]*umperR[i][j] + umperI[i][j]*umperI[i][j],2);

				dum_par[i][j] += dmparR[i][j]*umparR[i][j] + dmparI[i][j]*umparI[i][j];
				dup_par[i][j] += dpparR[i][j]*upparR[i][j] + dpparI[i][j]*upparI[i][j];
				// real parts


			}

		if(AREA){

		rhoSigq2[i][j] += (psiRD[i][j]+psiRU[i][j])*(psiRD[i][j]+psiRU[i][j]) + (psiID[i][j]+psiIU[i][j])*(psiID[i][j]+psiIU[i][j]);
		rhoDelq2[i][j] += (psiRD[i][j]-psiRU[i][j])*(psiRD[i][j]-psiRU[i][j]) + (psiID[i][j]-psiIU[i][j])*(psiID[i][j]-psiIU[i][j]);

		hq2Ed[i][j] += (h_real[i][j])*(h_real[i][j]) + (h_imag[i][j])*(h_imag[i][j]);
		}

		// |psiU +/- psiD|^2 =  Re(psiU +/- psiD)*Re(psiU +/- psiD) + Im(psiU +/- psiD)*Im(psiU +/- psiD)
		}
	}

	//////////////// export data
	if(DUMP){

// 		for(i=0; i<N; i++){
// 			for(j=0; j<N; j++){
//
// 		buf1 << dm[i][j][0] << " ";
// 		buf2 << dm[i][j][1] << " ";
// 			}
// 		buf1 << endl;
// 		buf2 << endl;
// 		}
		buf1 << sqrt(tq2[1][0])<<endl;

	}

	//print info
	cout << frame_num+1<< "  ";
	cout << lx[frame_num]<< "  ";
	cout << ly[frame_num]<< "  ";
 	cout << zavg[frame_num] << "  " ;
	cout << z1avg/nl1<< "  ";
	cout << z2avg/nl2<< "  ";
	cout << t0_frame << "  " ;
	cout << nt1 << "  " ;
	cout << nt2 << "  " ;
	cout << nl1 << "  " ;
	cout << nl2 << "  " ;
	cout << empty << " ";
	cout << endl;


} // end of loop over frames
//---------------------------------------------------------------------------------------------------------------
//END OF LOOP OVER ALL FRAMES//////////////////////////////////////////////////////////////////////////////////
//---------------------------------------------------------------------------------------------------------------

if(AREA==1){

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			cout << (rhoSigq2[i][j])/400/frame_num/phi0in/phi0in << " " ;
		}
	cout << endl;
	}

	cout << endl;

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			cout << (rhoDelq2[i][j])/400/frame_num/phi0in/phi0in << " " ;
		}
	cout << endl;
	}

	cout << endl;

}

if(TILT==1){
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			cout << (t1xR_cum[i][j])/frame_num << " " ;
		}
	cout << endl;
	}
	cout << "--------------------------"<<endl;

	cout << endl;

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			cout << (t1xI_cum[i][j])/frame_num << " " ;
		}
	cout << endl;
	}
	cout << "--------------------------"<<endl;
	cout << endl;

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			cout << (t1yR_cum[i][j])/frame_num << " " ;
		}
	cout << endl;
	}
	cout << "--------------------------"<<endl;
	cout << endl;

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){

			cout << (t1yI_cum[i][j])/frame_num << " " ;
		}
	cout << endl;
	}
	cout << "--------------------------"<<endl;
	cout << endl;

} // if(TILT==1)

//contruct q2 matrix
for(i=0; i<N; i++){
	for(j=0; j<N; j++){

		int q_x =((i < N/2) ? i : i-N);
		int q_y =((j < N/2) ? j : j-N);

		q2test[i][j]=q_x*q_x + q_y*q_y;
		q2[i][j]=q[i][j][0]*q[i][j][0] + q[i][j][1]*q[i][j][1];

		if(q2[i][j] != q2test[i][j]){cout << "The values of q have been improperly accessed" << endl;}

		q2[i][j]=(2*pi)*sqrt( pow(q[i][j][0]/lx_av,2) + pow(q[i][j][1]/lx_av,2));

	}
}

//	for(i=0; i<N; i++){
//		for(j=0; j<N; j++){
//			cout << q2[i][j] << " ";
//		}
//		cout << endl;
//	}
//	cout << endl;

qav(q2,q2_uniq,1);
qav(q2,q2_uniq_Ny,0);

qav(hq2,hq2_uniq,0); //changed from 1
qav(tq2,tq2_uniq,0); // changed from 1

qav(hq4,hq4_uniq,0);

if(AREA){
qav(rhoSigq2,rhoSigq2_uniq,0); //changed from 1
qav(rhoDelq2,rhoDelq2_uniq,0);
qav(hq2Ed,hq2Ed_uniq,0);
} //changed from 1

// when printing results, convert to nm, divide by frame_num, and divide by 4 for symmetric and antisymmetric quantities

cout << "q2=" << endl;
for(i=0; i<uniq; i++){cout << 10*q2_uniq[i] << ", ";}
cout << endl;
cout << endl;

//to keep the values at the Nyquist frequency, change uniq_Ny to uniq when printing the averages

cout << "hq2=" << endl;
for(i=0; i<uniq_Ny; i++){cout << hq2_uniq[i]/40000/frame_num << ", ";}
cout << endl;
cout << endl;

cout << "tq2=" << endl;

tq2_uniq[0]=tq0/(N*N*N*N);

for(i=0; i<uniq_Ny; i++){cout << tq2_uniq[i]/40000/frame_num << ", ";}
cout << endl;	cout << endl;

cout <<"__________ *error bars* ____________"<<endl;

cout << "sqrt(var(hq2))="<< endl;
for(i=0; i<uniq_Ny; i++){cout << sqrt( hq4_uniq[i]/frame_num - pow(hq2_uniq[i]/frame_num,2))/40000 << ", ";}
cout << endl;
cout << endl;


cout << "q2_tilt=" << endl;
for(i=0; i<uniq_Ny; i++){cout << 10*q2_uniq_Ny[i] << ", ";}
cout << endl; 	cout << endl;

if(TILT){

// 	cout << "t1xq2==========================================================="<< endl;
// 	for(i=0; i<N; i++){
// 		for(j=0; j<N; j++){
//
// 			//k = i*N + j;
//
// 			cout << t1xq2[i][j]/400/frame_num << " " ;
// 		}
// 		cout << endl;
// 	}

	qav(t1xq2,t1xq2_uniq,0);	qav(t1yq2,t1yq2_uniq,0);

	qav(dmq2,dmq2_uniq,0);		qav(dpq2,dpq2_uniq,0);

	qav(dpparq2,dpparq2_uniq,0);	qav(dpperq2,dpperq2_uniq,0);

	qav(dmparq2,dmparq2_uniq,0);	qav(dmperq2,dmperq2_uniq,0);

	qav(hdmpar,hdmpar_uniq,0);		qav(tdppar,tdppar_uniq,0);

	qav(upparq2,upparq2_uniq,0);	qav(upperq2,upperq2_uniq,0);

	qav(umparq2,umparq2_uniq,0);	qav(umperq2,umperq2_uniq,0);

	qav(dum_par,dum_par_uniq,0);	qav(dup_par,dup_par_uniq,0);

	qav(umparq4,umparq4_uniq,0);	qav(umperq4,umperq4_uniq,0);

	cout << endl; 	cout << endl;

cout<<"_________________  *Tilt* __________________"<<endl;

	cout << "t1xq2=" << endl;
	for(i=0; i<uniq_Ny; i++){cout << t1xq2_uniq[i]/100/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "t1yq2=" << endl;
	for(i=0; i<uniq_Ny; i++){cout << t1yq2_uniq[i]/100/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "dpq2=" << endl;
	for(i=0; i<uniq_Ny; i++){cout << dpq2_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "dmq2=" << endl;
	for(i=0; i<uniq_Ny; i++){cout << dmq2_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "dpparq2=" << endl;
	dpparq2_uniq[0] = 0.5*dpq2_uniq[0];
	for(i=0; i<uniq_Ny; i++){cout << dpparq2_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "dpperq2=" << endl;
	dpperq2_uniq[0] = 0.5*dpq2_uniq[0];
	for(i=0; i<uniq_Ny; i++){cout << dpperq2_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "dmparq2=" << endl;
	dmparq2_uniq[0] = 0.5*dmq2_uniq[0];
	for(i=0; i<uniq_Ny; i++){cout << dmparq2_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "dmperq2=" << endl;
	dmperq2_uniq[0] = 0.5*dmq2_uniq[0];
	for(i=0; i<uniq_Ny; i++){cout << dmperq2_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "Im(hdmpar)=" << endl;
	for(i=0; i<uniq_Ny; i++){cout << hdmpar_uniq[i]/4000/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "Im(tdppar)=" << endl;
	for(i=0; i<uniq_Ny; i++){cout << tdppar_uniq[i]/4000/frame_num << ", ";}
	cout << endl; 	cout << endl;

cout<<"_________________  *Directors* __________________"<<endl;

	cout << "umparq2=" << endl;
	umparq2_uniq[0] = 0.5*dmq2_uniq[0];
	for(i=0; i<uniq_Ny; i++){cout << umparq2_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "umperq2=" << endl;
	umperq2_uniq[0] = 0.5*dmq2_uniq[0];
	for(i=0; i<uniq_Ny; i++){cout << umperq2_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "upparq2=" << endl;
	upparq2_uniq[0] = 0.5*dpq2_uniq[0];
	for(i=0; i<uniq_Ny; i++){cout << upparq2_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "upperq2=" << endl;
	upperq2_uniq[0] = 0.5*dpq2_uniq[0];
	for(i=0; i<uniq_Ny; i++){cout << upperq2_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "Real(dum_par)=" << endl;
	dum_par_uniq[0] *= 0.5;
	for(i=0; i<uniq_Ny; i++){cout << dum_par_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout << "Real(dup_par)=" << endl;
	dup_par_uniq[0] *= 0.5;
	for(i=0; i<uniq_Ny; i++){cout << dup_par_uniq[i]/400/frame_num << ", ";}
	cout << endl; 	cout << endl;

	cout <<"__________ *error bars* ____________"<<endl;

	cout << "sqrt(var(umparq2))="<< endl;
	for(i=0; i<uniq_Ny; i++){cout << sqrt( umparq4_uniq[i]/frame_num - pow(umparq2_uniq[i]/frame_num,2))/400 << ", ";}
	cout << endl;
	cout << endl;

	cout << "sqrt(var(umperq2))="<< endl;
	for(i=0; i<uniq_Ny; i++){cout << sqrt( umperq4_uniq[i]/frame_num - pow(umperq2_uniq[i]/frame_num,2))/400 << ", ";}
	cout << endl;
	cout << endl;

// 	cout<<"tilt angle histogram"<<endl;
// 	for(i=0; i<100; i++){cout << hist_theta[i] << " ";}
// 	cout << endl;
//
// 	cout<<"tilt^2 histogram"<<endl;
// 	for(i=0; i<100; i++){
// 		for(j=0; j<100; j++){
// 			 cout << hist_t[i][j] << " ";}
// 		cout << endl;}
// 	cout << endl;
//
// 	cout<<"tx"<<endl;
// 	for(i=0; i<100; i++){cout << hist_t2[i] << " ";}
// 	cout << endl;
// //
 	cout<<"tmag"<<endl;
 	for(i=0; i<100; i++){cout << ty_cum[i] << " ";}
 	cout << endl;
//
// 	cout << "---------------------"<< endl;
// //
// 	cout<<"tproj1_cum"<<endl;
// 	for(i=0; i<100; i++){cout << tproj1_cum[i] << " ";}
// 	cout << endl;
//
// 	cout<<"tproj2_cum"<<endl;
// 	for(i=0; i<100; i++){cout << tproj2_cum[i] << " ";}
// 	cout << endl;

// 	cout<<"txq_hist"<<endl;
// 	for(i=0; i<frames; i++){cout << txq[i] << " ";}
// 	cout << endl;

// 	cout<<"t_grid_hist"<<endl;
// 	for(i=0; i<100; i++){cout << tghist[i] << " ";}
// 	cout << endl;

} // if (TILT)

if(DUMPQ){

	for(i=0; i<uniq_Ny; i++){buf4 << 10*q2_uniq_Ny[i] << " ";}
	buf4<<endl;

	for(i=0; i<uniq_Ny; i++){buf4 << hq2_uniq[i]/40000/frame_num << " ";}
	buf4<<endl;

	for(i=0; i<uniq_Ny; i++){buf4 << tq2_uniq[i]/40000/frame_num << " ";}
	buf4<<endl;

	if(TILT){

		for(i=0; i<uniq_Ny; i++){buf4 << dmparq2_uniq[i]/400/frame_num << " ";}
		buf4<<endl;

		for(i=0; i<uniq_Ny; i++){buf4 << dpparq2_uniq[i]/400/frame_num << " ";}
		buf4<<endl;

		for(i=0; i<uniq_Ny; i++){buf4 << dmperq2_uniq[i]/400/frame_num << " ";}
		buf4<<endl;

		for(i=0; i<uniq_Ny; i++){buf4 << dpperq2_uniq[i]/400/frame_num << " ";}
		buf4<<endl;

		for(i=0; i<uniq_Ny; i++){buf4 << hdmpar_uniq[i]/4000/frame_num << " ";}
		buf4<<endl;

		for(i=0; i<uniq_Ny; i++){buf4 << tdppar_uniq[i]/4000/frame_num << " ";}
		buf4<<endl;

		for(i=0; i<uniq_Ny; i++){buf4 << t1xq2_uniq[i]/100/frame_num << " ";}
		buf4<<endl;

		for(i=0; i<uniq_Ny; i++){buf4 << umparq2_uniq[i]/400/frame_num << " ";}
		buf4<<endl;

		for(i=0; i<uniq_Ny; i++){buf4 << upparq2_uniq[i]/400/frame_num << " ";}
		buf4<<endl;

		for(i=0; i<uniq_Ny; i++){buf4 << umperq2_uniq[i]/400/frame_num << " ";}
		buf4<<endl;

		for(i=0; i<uniq_Ny; i++){buf4 << upperq2_uniq[i]/400/frame_num << " ";}
		buf4<<endl;
	}

}

if(AREA){
cout << "rhoSigq2=" << endl;
for(i=0; i<uniq_Ny; i++){

cout << rhoSigq2_uniq[i]/400/frame_num/phi0in/phi0in << " ";
}

cout << endl;
cout << endl;

cout << "hq2_Edholm=" << endl;
for(i=0; i<uniq_Ny; i++){

float Srho=(nl/2)/phi0in/phi0in*(z1sq_av+z2sq_av)/(2*frame_num)*(rhoSigq2_uniq[i]/4/frame_num/(lx_av*ly_av)); // in Angstroms

cout << (hq2Ed_uniq[i]/(4*frame_num*nl/2) -Srho)/(phi0/frame_num)/10000 << " ";
}



// cout << "rhoDelq2=" << endl;
// for(i=0; i<uniq_Ny; i++){
//
// cout << rhoDelq2_uniq[i]/400/frame_num/phi0in/phi0in << " ";
// }
//
// cout << endl;
cout << endl;} // if(AREA)


if(DUMP){

		buf1.close();
		buf2.close();
}

if(DUMPQ){

	buf4.close();
}

fftwf_destroy_plan(spectrum_plan);
fftwf_destroy_plan(inv_plan);

fclose(lipidxp);
fclose(lipidyp);
fclose(lipidzp);


cout << "Average Box Size= "<< lx_av << " Angstroms" << endl;

cout << "Total Number of Neighboring Empty Patches= "<< empty_tot << endl;
cout << "<z1^2>= "<<z1sq_av/frame_num <<" Angstroms^2" << endl;
cout << "<z2^2>= "<<z2sq_av/frame_num <<" Angstroms^2" << endl;

cout.precision(10);
cout << "Average Number Density= "<< phi0/frame_num << " Angstroms^(-2)" << endl;
cout << "Average monolayer thickness=" << t0/frame_num << " Angstroms" << endl;
cout << "Average (n.N) = " << dot_cum/(frame_num*nl) << endl;

cout << endl;

if(t0in > t0/frame_num+0.001 || t0in < t0/frame_num-0.001){cout << "The input and output thickness are not the same! The q=0 point will not be accurate "<< endl;}
cout << endl;

if(phi0in > phi0/frame_num+0.001 || phi0in < phi0/frame_num-0.001){cout << "The input and output phi0's are not the same! The q=0 point will not be accurate "<< endl;}
cout << endl;

return 0;
} // end of main function

//---------------------------------------------------------------------------------------------------------------
//DEFINE FUNCTIONS//////////////////////////////////////////////////////////////////////////////////
//---------------------------------------------------------------------------------------------------------------

void fullArray( float array1R[][N], float array1I[][N], fftwf_complex array2[], float lxy)
{ // filling the right half of the full array
 // using the property h_{-q}=h*_{q}=h_{N-q}. This was checked against MATLAB
int a,b,c;

float factor=lxy/(N*N); // lx[frame_num]/N^2 prefactor allowing the FT to have the correct units

for(c=0; c<N*(N/2+1); c++){
	array2[c][0] *= factor;  array2[c][1] *= factor;
}

for(a=0; a<N; a++){
	for(b=0; b<N; b++){


		if(a==0){// top row
			if(b <= N/2){
				array1R[a][b]=array2[a*(N/2+1) + b][0];
				array1I[a][b]=array2[a*(N/2+1) + b][1];
			}
			else{array1R[a][b]=  array2[a*(N/2+1) + (N-b)][0];
				 array1I[a][b]= -array2[a*(N/2+1) + (N-b)][1];
			}
		}

		if(a !=0 && b==0){  // leftmost column
			array1R[a][b]=array2[a*(N/2+1) + b][0];
			array1I[a][b]=array2[a*(N/2+1) + b][1];
		}

		if(a>0 && b>0){ // rest of the array

			if(b<=N/2){ // b= 0...N/2 portion of block
				array1R[a][b]=array2[a*(N/2+1) + b][0];
				array1I[a][b]=array2[a*(N/2+1) + b][1];
			}

			else{ // b=N/2+1...N portion of block
				array1R[a][b]=  array2[(N-a)*(N/2+1) + (N-b)][0];
				array1I[a][b]= -array2[(N-a)*(N/2+1) + (N-b)][1];
			}

		}

	} // two for loops
}


} // end of function
//////////////////////////////////////////////////////
void qav(float array2D[][N], float array1D_uniq[], int Ny)
// takes the full 2D Fourier transform array
// and averages the components which have the same magnitude of q
// the argument array1D_uniq should be initialized to zero
// after this function is called, array1D_uniq will contain
// the values in array2D averaged over each value of q
// when array2D is of dimension NxN, array1D is of [1][(N+4)*(N+2)/8]
{
// int uniq=(N+4)*(N+2)/8; // this is the number of unique values in qmat, which = sum_{i=1}^{N/2+1} {i}
// int uniq_Ny=N*(N+2)/8; // this is the number of unique values in qmat, which = sum_{i=1}^{N/2} {i}

float qs[N][N][2];
int a1,a2,b1,b2;

int count_out=0;

int *count_in;

if(Ny){ count_in=  (int *) calloc(uniq,sizeof(int *));}
else  { count_in=  (int *) calloc(uniq_Ny,sizeof(int *));} //toss Nyquist values
// the arrays are initialized to zero

for(a1=0; a1<N; a1++){
	for(a2=0; a2<N; a2++){

		qs[a1][a2][0]=((a1 < N/2) ? a1 : N-a1); // this definition is slightly different from q[N][N][2]
		qs[a1][a2][1]=((a2 < N/2) ? a2 : N-a2);	// since it has (N-a1) instead of (a1-N), but we only
												//care about |q| and keep all values positive

	}
}
//// 4 nested loops
for(a1=0; a1<N/2+Ny; a1++){      // these two loops scan through the unique
	for(a2=a1; a2<N/2+Ny; a2++){ // values of |q|; when lx=ly it used to be (a2=a1; a2<N/2+Ny; a2++)

	// For tilt quantities, Ny=0.

		for(b1=0; b1<N; b1++){ // these loops scan through all values of qvals
			for(b2=0; b2<N; b2++){

//				if(qs[a1][a2][0] == qs[b1][b2][0] && qs[a1][a2][1] == qs[b1][b2][1]){

				if((qs[a1][a2][0] == qs[b1][b2][0] && qs[a1][a2][1] == qs[b1][b2][1]) || \
				   (qs[a1][a2][1] == qs[b1][b2][0] && qs[a1][a2][0] == qs[b1][b2][1])){ // when lx=ly

					array1D_uniq[count_out] += array2D[b1][b2];
					count_in[count_out]++;}
			}
		}
	count_out++;
	}
}//4 nested loops



if((Ny==0) && count_out != uniq_Ny){cout << "count_out != uniq_Ny " << count_out << " "<< uniq_Ny <<endl;}

for(a1=0; a1<count_out; a1++){array1D_uniq[a1] /= count_in[a1]; //cout << count_in[a1] << " " ;
}

free(count_in);
} // end of function
//////////////////////////////////////////////////////

