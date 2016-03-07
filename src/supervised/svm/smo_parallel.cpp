/**************************************************
 * smooneloop.cpp
 *
 * A variant of Modification 2 in the Keerthi et al. (2000) paper:
 * Improvements to Platt's SMO algorithm for SVM classifier
 * design
 * 
 * Author: Jeyanthi Salem Narasimhan 
 * Created: January 2007.
 *
 * *************************************************/
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#define EPS 1e-20
#define ZERO 1e-12

#include <mpi.h>
#include <ga.h>

#include <iostream>
#include <mpi.h>
#include "ga.h"
#include <string>
#include <iomanip>
#include <locale>
#include <sstream>
#include <vector>
#include <map>

#include <cassert>
#include <fstream>
#include <armci.h>

using namespace std;

#define DEBUG0
//#define DUAL
#define PRINT

struct allred_ds{
    double val;
    int index;
} lowin, lowout, upin, upout;
/* function declarations */
int tokenize (string token, vector<string>&retarray);
int64_t read_file(char *fname, int64_t wcl, int64_t wcw, 
        map<string, int>& classmap);
double par_addFcache(void);
double par_normw(void);
double par_addarrele(void);
void calc_wc(char *, int64_t*, int64_t*);
int64_t gathersvs(int64_t, int64_t);
int read_testfile(char *fname, int64_t wcl, int64_t wcw, 
        const map<string, int>& classmap);

double **inputvecmat, *yarr, *fcache, *alpharr, **testvecmat, *testy;
double b_up, b_low, C, sigmasqr, thresh;
int *setinfo, dim, numtestsamp;
int64_t i_up, i_low, numtrsamp;
double pone = 1.0, mone = -1.0;
double *precompdotprod, *precompdotprod_test, TOL;

int me, nproc;
int dataset_handle;
int svset_handle;

double *sample_i, *local_sample;
int64_t first_pone = -1, first_mone = -1;
int64_t max_sample_size = 0;
int64_t *row_ptr;
int64_t *temp_sv_row_ptr;

/*  test file variables */
int testdataset_handle;
int64_t *test_row_ptr;
double *testlocal_sample;
int64_t test_max_sample_size = 0;

void calc_wc(char *fname, int64_t *wcl, int64_t *wcw)
{
    /*  variables for calculating wc output */
    char wcout[500];
    FILE *fp;
    string token, cmd_name = "wc ";
    vector<string> retarray;

    assert(fname);

    cmd_name += fname;
    cout<<"cmd name: "<<cmd_name<<endl;
    fp = popen(cmd_name.c_str(), "r");
    if (fp == NULL)
    {
        printf("Failed to run wc command");
        exit(0);
    }
    cout<<"fp "<<fp<<endl;
    assert(fgets(wcout, sizeof(wcout), fp)!= NULL);
    cout<<"wcout: "<<wcout<<" ";
    token.assign(wcout);/*  gets the entire wc output */
    tokenize(token, retarray);
    *wcl = atol(retarray[0].data());
    *wcw = atol(retarray[1].data());
    cout<<"wcl: "<<*wcl<<" wcw: "<<*wcw<<endl;
    pclose(fp);
}
void precomputedp ()
{
	int i, j;
	double sum  = 0.0;
	for (j  = 0; j < numtrsamp; j++)
	{
		sum = 0.0;
		for(i = 0; i <dim; i++)
		{
		if ( fabs(inputvecmat[j][i]) < ZERO )
			continue;
		sum += inputvecmat[j][i] * inputvecmat[j][i];
		}
		precompdotprod[j] = sum;
	}
}
void precomputedp_test ()
{
	int i, j;
	double sum  = 0.0;
	for (j  = 0; j < numtestsamp; j++)
	{
		sum = 0.0;
		for(i = 0; i <dim; i++)
		{
		if ( fabs(testvecmat[j][i]) < ZERO )
			continue;
		sum += testvecmat[j][i] * testvecmat[j][i];
		}
		precompdotprod_test[j] = sum;
	}
}

double testkernelfunc (int i1, int i2)
{
	double twosigmasqr, xx = 0, xy = 0, yy = 0, normsqr;
	int i;
	twosigmasqr = 2 * sigmasqr;
/*	for (i = 0; i < dim; i++)
	{
		xx += inputvecmat[i1][i] * inputvecmat[i1][i];
	}*/
	xx += precompdotprod[i1];
	yy += precompdotprod_test[i2];
	/*for (i = 0; i < dim; i++)
	{
		if ( testvecmat[i2][i] < ZERO )
			continue;
		yy += testvecmat[i2][i] * testvecmat[i2][i];
	}*/
	for (i = 0; i < dim; i++)
	{
		if ( fabs(inputvecmat[i1][i]) < ZERO ||
			fabs(testvecmat[i2][i]) < ZERO )
			continue;
		xy += inputvecmat[i1][i] * testvecmat[i2][i];
	}
	normsqr = (2 * xy -xx -yy) / twosigmasqr; 
	return( exp(normsqr) );
}

double testevali(int idx)
{
	double testkernelfunc (int i1, int i2);
	double scalarsum = 0.0, scalar;
	int i;
	for ( i = 0; i < numtrsamp; i++)
	{
		if (alpharr[i] < ZERO)
			continue;
		scalar = alpharr[i] * yarr[i];
		scalar *= testkernelfunc(i, idx);
		scalarsum += scalar;
	}
	return scalarsum;
}
int testsamples(void)
{
	double testval;
	int mclass = 0, i;
	for ( i = 0; i < numtestsamp; i++)
	{
		testval = testevali(i) - thresh;
		if (testval  < ZERO)
			{
				if (testy[i] > ZERO) 
					mclass++;
			} 
		if (testval > ZERO )
		{
			if (testy[i] < ZERO)
				mclass ++;
		}
	}
	return mclass;
}


#if 0
void setslackarr(double *slackarr)
{
	double arg;
	int i;
	for ( i = 0; i < numtrsamp; i++)
	{
		/*if ( fabs(alpharr[i] - C) < ZERO)
		{*/
			arg = (evali(i) - thresh) * yarr[i];
			arg = pone - arg;
			slackarr[i] = (arg > ZERO) ? arg : 0.0;
	/*	}
		else
			slackarr[i] = 0.0;*/
	}
}
#endif
int rfile(char *fname, double **mat, double *acty, int samplesize)
{
	FILE *fp;
	int dummy;
	int i, icol;
	fp = fopen(fname,"r");
	if (!fp)
		return 0;
	for (i = 0; i < samplesize; i++)
	{
		/*fscanf(fp, "%lf", (acty + i));*/
		fscanf(fp, "%lf", (acty + i));
		for (icol = 0; icol < dim; icol++)
		{
			dummy = fgetc(fp);
			fscanf(fp, "%lf", &(mat[i][icol]) );
		}
		dummy = fgetc(fp);
	}
	fclose(fp);
	return 1;
}


// elems contains the total number of elements including the columns and
// values
double mydpfunc(double *sample1, int64_t elems1, double *sample2, int64_t elems2)
{
    double value = 0;
    int64_t  start1 = 0, start2 = 0;

    while (start1 < elems1 && start2 < elems2) {
        /* FIXME: was previously equality check*/
        if (fabs(sample1[start1] - sample2[start2])<1e-12) { // same column
            value += sample1[start1 + 1] * sample2[start2 + 1];
            start1 += 2;
            start2 += 2;
        }
        else if (sample1[start1] < sample2[start2]) {
            start1 += 2;
        } 
        else {
            start2 += 2;
        }
    }

    return value;
}


double mykernelfunc(double *sample1, int64_t elems1, double *sample2, int64_t elems2)
{
	double twosigmasqr, xx, xy, yy, normsqr;
	twosigmasqr = 2 * sigmasqr;
	xx = mydpfunc(sample1, elems1, sample1, elems1);
	yy = mydpfunc(sample2, elems2, sample2, elems2);
	xy = mydpfunc(sample1, elems1, sample2, elems2);
	normsqr = (2 * xy -xx -yy) / twosigmasqr; 
	return ( exp(normsqr));
}
    
    
    /*tuned for sparse non binary datasets */
double dp(int i1, int i2)
{
	int i;
	double sum = 0.0;
	for (i = 0; i < dim; i++)
	{
		if ( fabs(inputvecmat[i1][i]) < ZERO ||
			fabs(inputvecmat[i2][i]) < ZERO )
			continue;
		sum += inputvecmat[i1][i] * inputvecmat[i2][i];
	}
	return sum;
}
double kernelfunc (int i1, int i2)
{
	double twosigmasqr, xx, xy, yy, normsqr;
	twosigmasqr = 2 * sigmasqr;
	xx = precompdotprod[i1];
	yy = precompdotprod[i2];
	xy = dp (i1, i2);
	normsqr = (2 * xy -xx -yy) / twosigmasqr; 
	return( exp(normsqr) );
}
void updateFI(double delalph1, double delalph2, int i1, int i2)
{
    double withone, withtwo;
    int i;
    for ( i = 0; i < numtrsamp; i++)
    {
        withone = kernelfunc(i1, i);
        withtwo = kernelfunc(i2, i);
        fcache[i] += yarr[i1] * delalph1 * withone + 
            yarr[i2] * delalph2 * withtwo;
    }
}
void setset(int idx)
{
	double alphval, yval;
	alphval = alpharr[idx];
	yval = yarr[idx];
	if(alphval > ZERO && fabs(alphval - C ) > ZERO)
		setinfo[idx] = 0;
	if( alphval < ZERO )
	{
		if ( fabs(yval - pone) < ZERO )
			setinfo[idx] = 1;
		else
			setinfo[idx] = 4;
	}
	if ( fabs(alphval - C) < ZERO)
	{
		if ( fabs(yval - pone) < ZERO)
			setinfo[idx] = 3;
		else
			setinfo[idx] = 2;

	}
}
void computeb_i_up()
{
	double upmin = 1e12;
	int i, sinfo;
	for ( i = 0; i < numtrsamp; i++)
	{
		sinfo = setinfo[i];
		if (sinfo == 0 || sinfo == 1 || sinfo == 2 )
		{
			if ( fcache[i] < upmin)
			{
				upmin = fcache[i];
				b_up = upmin;
				i_up = i;
			}
		}
	}
}
void computeb_i_low()
{
	double lowmax = -1e12;
	int i, sinfo;
	for ( i = 0; i < numtrsamp; i++)
	{
		sinfo = setinfo[i];
		if (sinfo == 0 || sinfo == 3 || sinfo == 4 )
		{
			if ( fcache[i] > lowmax)
			{
				lowmax = fcache[i];
				b_low = lowmax;
				i_low = i;
			}
		}
	}
}

int local_takestep(int i1, int i2)
{
    // perform GA_Get for the i2 and i1
    int64_t ld = -1, mylo[1], myhi[1];
    int64_t num_elems = 0;
    
    // Get the first sample 

    mylo[0] = row_ptr[i1];
    myhi[0] = row_ptr[i1 + 1] - 1;

    if (0 == me)
        NGA_Get64(dataset_handle, mylo, myhi, sample_i, &ld);  
    
    num_elems += row_ptr[i1 + 1] - row_ptr[i1];
//    cout<<"local take step: first sample: "<<num_elems<<endl;
    // Get the second sample
    mylo[0] = row_ptr[i2];
    myhi[0] = row_ptr[i2 + 1] - 1;
    
    if (0 == me)
        NGA_Get64(dataset_handle, mylo, myhi, sample_i + num_elems, 
                &ld);  
    
    num_elems += row_ptr[i2 + 1] - row_ptr[i2];
 //   cout<<"local take step: toal size: "<<num_elems<<endl;

    GA_Sync();

    // Broadcast the local samples FIXME, was previously MPI_LONG
    MPI_Bcast(sample_i, num_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); 
    // subtract the num elems 
    num_elems -= (row_ptr[i2 + 1] - row_ptr[i2]);

    return num_elems;
}/*  end function local_takestep */

int mydsupdate(int64_t i1, int64_t i2, double a1, double a2, double delalph1, 
        double delalph2, int64_t num_elems_i1, int64_t num_elems_i2,
        double y1, double y2)
{
    int64_t start_index = me * numtrsamp / nproc;
    int64_t end_index = (me + 1) * numtrsamp / nproc;

    if (me == nproc - 1)
        end_index = numtrsamp;

    int64_t i, sinfo;
    int64_t ld = -1, mylo[1], myhi[1], nelems;

    double upmin = 1e12;
    double lowmax = -1e12;
    /*  the following two lines are important when the number of samples is
     *  not a large number and there is a possibility that some processors
     *  might get exactly one sample or no sample at all*/
     b_up = upmin;
     b_low = lowmax;
    for(i = start_index; i < end_index; i++)
    {
        mylo[0] = row_ptr[i];
        myhi[0] = row_ptr[i + 1] - 1;
        nelems = row_ptr[i + 1] - row_ptr[i];
        NGA_Get64(dataset_handle, mylo, myhi, local_sample, &ld);  
        
        // alpha update 
        if (i == i1) {
            local_sample[0] = a1;
        }
        
        // alpha update
        if (i == i2) {
            local_sample[0] = a2;
        }

        // setinfo update
        if (i == i1 || i == i2) {
            double alphval, yval;
            alphval = local_sample[0];
            yval = local_sample[3];

            if (alphval > ZERO && fabs(alphval - C ) > ZERO)
                local_sample[1] = 0;
            
            if (alphval < ZERO ) {
                if ( fabs(yval - pone) < ZERO )
                    local_sample[1] = 1;
                else
                    local_sample[1] = 4;
            }

            if ( fabs(alphval - C) < ZERO) {
                if ( fabs(yval - pone) < ZERO)
                    local_sample[1] = 3;
                else
                    local_sample[1] = 2;
            }
        }

        // Kernel calculation and fcache update
        double withone, withtwo;
        withone = mykernelfunc(&(sample_i[4]), num_elems_i1 - 4, &(local_sample[4]),
                    nelems - 4); 
        withtwo = mykernelfunc(&(sample_i[num_elems_i1 + 4]), num_elems_i2 - 4,
                    &(local_sample[4]), nelems - 4); 
/*         cout<<"y1: "<<y1<<" y2: "<<y2<<" delalph1: "<<delalph1<<" delalph2: "<<
            delalph2<<endl;*/
        local_sample[2] += y1 * delalph1 * withone + 
            y2 * delalph2 * withtwo;
/*      if (local_sample[2] > 10)
        {
            cout<<"fcache["<<i+1<<"]: "<<local_sample[2]<<endl;
            cout<<"local sample nelems: "<<nelems - 4 <<endl;
            cout<<"withone: "<<withone<<" withtwo: "<<withtwo<<endl;
            cout<<"The local sample itself:"<<endl;
            for (int ii = 0; ii < nelems; ii++ )
            {
                cout<<local_sample[ii]<<" ";
            }
            cout<<endl;
        }*/


        // bup and blow calculation
        sinfo = local_sample[1];
        if (sinfo == 0 || sinfo == 1 || sinfo == 2 )
        {
            if (local_sample[2] < upmin)
            {
                upmin = local_sample[2];
                b_up = upmin;
                i_up = i;
            }
        }

        if (sinfo == 0 || sinfo == 3 || sinfo == 4 )
        {
            if (local_sample[2] > lowmax)
            {
                lowmax = local_sample[2];
                b_low = lowmax;
                i_low = i;
            }
        }

        // Update the value FIXME:should verify the correctness of mylo and hi
        mylo[0] = row_ptr[i];
        myhi[0] = row_ptr[i] + 2;
        NGA_Put64(dataset_handle, mylo, myhi, local_sample, &ld);  
    }/*  end for i = start_index */

    GA_Sync();

    // Allreduction for MPI
#if 0
    lowin.val = b_low;
    lowin.index = i_low;

    upin.val = b_up;
    upin.index = i_up;

    MPI_Allreduce(&lowin, &lowout, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    MPI_Allreduce(&upin, &upout, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    b_up = upout.val;
    /*FIXME */
    i_up = (int64_t)upout.index;

    b_low = lowout.val;
    /*FIXME */
    i_low = (int64_t)lowout.index;
#else
    double out_low, out_up; 
//     cout << "b_up " << b_up << ", b_low " << b_low << endl;
    MPI_Allreduce(&b_low, &out_low, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&b_up, &out_up, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if (fabs(out_low - b_low) > 1e-10) {
        i_low = numtrsamp;
    }
    /* if more than one processor holds b_low, then return the sample index that
     * is the lowest
     * */ 
    MPI_Allreduce(MPI_IN_PLACE, &i_low, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);

    if (fabs(out_up - b_up) > 1e-10) {
        i_up = numtrsamp;
    }
    MPI_Allreduce(MPI_IN_PLACE, &i_up, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    b_low = out_low;
    b_up = out_up;
#if 1
        static int smo_counter = 0;
    if (0 == me && (smo_counter % 500 == 0))
    {
        cout << "iter:" << smo_counter++<< " , b_up " << out_up << ", b_low " << out_low << endl;
    }
#endif
#endif
    return 0;
}/* end my dsupdate*/

int takestep( int i1, int i2)
{
    double s, alph1, alph2, F1, F2, L, H, k11, k22, k12, y1, y2 ;
    double eta, a2, a1, delalph1, delalph2, compart, Lobj, Hobj;
    double t;
    /*sepw = (double *) malloc(sizeof(double) * dim);*/
    if ( i1 == i2)
        return 0;

#if 0
    if (0 == me)
        cout << "in take step: i1: "<<i1+1 << " i2: " << i2+1 << endl;
#endif

    int64_t num_elems_i1;

    // num elems is number of elements in i1
    num_elems_i1 = local_takestep(i1, i2);

    alph1 = sample_i[0];
    //setinfo is ignored
    F1 = sample_i[2];
    y1  = sample_i[3];  

    alph2 = sample_i[num_elems_i1];
    //setinfo is ignored
    F2 = sample_i[num_elems_i1 + 2];
    y2 = sample_i[num_elems_i1 + 3];

    // variable calculations
    s = y1 * y2; 
    if ( fabs(y1 - y2) > ZERO)
    {
        L = ( (alph2 - alph1) > ZERO )? alph2 - alph1 : 0.0; 
        H = ( (C + alph2 - alph1) < C) ? 
            C + alph2 - alph1 : C ;
    }
    else
    {
        L = ( (alph2 + alph1 - C) > ZERO )? 
            alph2 + alph1 - C : 0.0; 
        H = ( (alph2 + alph1) < C) ? alph2 + alph1 : C ;
    }
    if ( fabs(L - H) < ZERO )
    {
        return 0;
    }

    int64_t num_elems_i2;

    num_elems_i2 = row_ptr[i2 + 1] - row_ptr[i2];

    k12 = mykernelfunc(&(sample_i[4]), num_elems_i1 - 4,
            &(sample_i[num_elems_i1 + 4]), num_elems_i2 - 4); 

    k11 = mykernelfunc(&(sample_i[4]), num_elems_i1 - 4,
            &(sample_i[4]), num_elems_i1 - 4); 

    k22 = mykernelfunc(&(sample_i[num_elems_i1 + 4]), num_elems_i2 - 4,
            &(sample_i[num_elems_i1 + 4]), num_elems_i2 - 4); 
    /*cout<<"K12: "<<k12<<" K11: "<<k11<<" K22: "<<k22<<endl;*/
    eta = 2 * k12 - k11 - k22;
    if ( eta < ZERO)
    {
        a2 = alph2 - y2 * (F1 - F2) / eta;
        if (a2 < L)
        {
            a2 = L;
        }
        else if (a2 > H)
            a2 = H;
    }
    else
    {
        compart = y2 * (F1 - F2) - eta * alph2;
        Lobj = 0.5 * eta * L * L + compart * L;
        Hobj = 0.5 * eta * H * H + compart * H;
        if (Lobj > Hobj + EPS)
            a2 = L;
        else if (Lobj < Hobj - EPS)
            a2 = H;
        else
            a2 = alph2;
    }
    if ( a2 < ZERO)
    {
        a2 = 0.0;
    }
    else if (a2 > (C - ZERO))
        a2 = C;

    delalph2 = a2 - alph2;

    if ( fabs(delalph2) < EPS * (a2 + alph2 + EPS ))
    {
        fprintf(stdout,"fabs quit\n");
        fprintf(stdout, "delal:%lf a2+alph2%lf L:%lf H:%lf ",
                delalph2,
                a2 + alph2, L, H);
        return 0;
    }
    a1 = alph1 + s * (alph2 - a2);
    if ( a1 < ZERO)
    {
        a2 += s * a1;
        a1 = 0.0;
    }
    else if( a1 >(C - ZERO) )
    {
        t = a1 - C;
        a2 += s * t;
        a1 = C;
    }

    delalph1 = a1 - alph1;
    /*cout<<"takestep end :a1 "<<a1<<" a2: "<<a2<<endl;*/
#if 0
    alpharr[i1] = a1;
    alpharr[i2] = a2;
    setset(i1);
    setset(i2);
    updateFI(delalph1, delalph2, i1, i2);
    computeb_i_low();
    computeb_i_up();
#else
    //My DS update

    mydsupdate(i1, i2, a1, a2, delalph1, delalph2, num_elems_i1, num_elems_i2, y1, y2);
#endif
    return 1;
}

int main(int argc, char *argv[])
{
    int64_t i, iupsetflag, ilowsetflag;
    int i1, i2;
    unsigned int svcnt = 0, bsvcnt = 0, zsvcnt = 0;
    double primal,dual, *slackarr, wval, secelap;
    struct timeval t[2];
    int64_t wcl, wcw; 
    map<string, int> classmap;
    MPI_Init(&argc, &argv);
    GA_Initialize();

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &me); 

    if (0 == me)
        cout << "smo_parallel " << nproc << endl;

    //    cout << "Argc: " <<argc << endl;

    if (argc != 12)
    {
        fprintf(stderr, "Usage: # of training samples\n");
        fprintf(stderr, "training file\n");
        fprintf(stderr, "dim\n");
        fprintf(stderr, "C\n");
        fprintf(stderr, "sigmasqr\n");
        fprintf(stderr, "Tol\n");
        fprintf(stderr, "wc -l\n");
        fprintf(stderr, "wc -w\n");
        fprintf(stderr, "# of test samples\n");
        fprintf(stderr, "Test file\n");
        fprintf(stderr, "test wc -w\n");
        exit(0);
    }

    numtrsamp = atoi(argv[1]);
    dim = atoi(argv[3]);
    C = atof (argv[4]);
    sigmasqr = atof(argv[5]);
    TOL = atof(argv[6]);
    numtestsamp = atoi(argv[9]);
    /*  precompdotprod = (double *) malloc(sizeof(double) * numtrsamp);
        precompdotprod_test = (double *) malloc(sizeof(double) * 
        numtestsamp);*/
    /*
     * calculating wc -l and wc -w from inside the program
     */
    //    calc_wc(argv[2], &wcl, &wcw);

    /* read the dataset */
    read_file(argv[2], atol(argv[7]), atol(argv[8]), classmap);

    /* Dataset */    
    /* Start parallelization */
    /* training dataset */

    /* FIXME  - nothing to fix, unused variable*/
    int64_t mynumsamples = numtrsamp / nproc;

    int64_t dims[1], myhi[1], mylo[1], ld=-1;

#if 0
    precomputedp();
#endif

    b_up = mone; b_low = pone;

    double t1, t2;

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    while ( b_up < b_low - 2 * TOL)
    {
        i2 = i_low;
        i1 = i_up;
        if (!takestep ( i1, i2))
            break;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();

    if (me == 0)
        cout << "Total Time " << (t2 - t1)* 1.0e3 << " ms" << endl;
    thresh = par_addFcache();
    if (0 == me)
    {
        cout <<"Threshold: "<<thresh<<endl; 
        unsigned int svcnt =0, bsvcnt = 0, zsvcnt = 0;
        double myalpha;
        for (i = 0; i < numtrsamp; i++)
        {
            mylo[0] = row_ptr[i];
            myhi[0] = row_ptr[i]+0;
            /*  change myalpha to myvals if fcache value is also required */
            NGA_Get64(dataset_handle, mylo, myhi, &myalpha, &ld);  
            /*             myfcache = myvals[2];
                           cout<<"A["<<i<<"]:"<<myalpha<<" F:"<<myfcache<<endl;*/
            if (myalpha > ZERO && fabs(myalpha - C) > ZERO)
            {
                //                cout<<i<<" nsv"<<endl;
                svcnt ++;
            }
            else if(fabs(myalpha - C) < ZERO)
            {
                //               cout<<i<<" bsv"<<endl;
                bsvcnt++;
            }
            else
            {
                //              cout<<i<<" zsv"<<endl;
                zsvcnt++;
            }
        }
        cout<<"Nsv: "<<svcnt<<" Bsv: "<<bsvcnt<<" Zsv: "<<zsvcnt<<endl;
    }
    GA_Sync();
#ifdef DUAL
    double wtw = par_normw();
    double addalpha = par_addarrele();
    if (0 == me)
    {
        dual = addalpha - wtw;
        cout<<"dual addalpha: "<<addalpha<<" wtw: "<<wtw<<endl;
        cout<<"The dual value : "<<dual<<endl;
    }
#endif
    /*  int64_t totsvscount = gathersvs(atol(argv[7]), atol(argv[8]));
        if (0 == me)
        cout<<"totsvscount: "<<totsvscount<<endl;
        if (!read_testfile(argv[10], numtestsamp, atol(argv[11]), classmap))
        {
        cout<<"Failed reading test file\n"<<endl;
        exit(0);
        }*/


    MPI_Barrier(MPI_COMM_WORLD);
    GA_Terminate();

    MPI_Finalize();
}/*  end main */

double fcalc(int idx)
{
    double retf = 0.0, tempscalar;
    int i;
    for ( i = 0; i < numtrsamp; i++)
    {
        tempscalar = alpharr[i] * yarr[i];
        tempscalar *= kernelfunc(i, idx);
        retf += tempscalar;
    }
    retf = retf - yarr[idx];	
    return retf;
}

/* Reading the Dataset part */
int tokenize (string token, vector<string>&retarray)
{
    const string SPACES= " \t\r\n\f";
    void remove_space(string&, const string);
    string tempstr;
    const string singlespace=" ";
    bool flag = false;
    size_t tab_place, char_place;
    remove_space(token, SPACES);
    if (token.empty())
        return 0;
    tab_place = token.find_first_of(SPACES);
    char_place = 0;
    while(tab_place != string::npos)
    {
        flag = true;
        tempstr = token.substr(char_place,tab_place-char_place);
        retarray.push_back(tempstr);
        char_place = token.find_first_not_of(SPACES, tab_place);
        tab_place = token.find_first_of(SPACES, char_place);

    }
    if(!flag)
        retarray.push_back(token);
    else
        retarray.push_back(token.substr(char_place));
    return 1;
}

void remove_space(string &node, const string spaces)
{
    size_t spacestart, spaceend;

    spacestart = node.find_first_not_of(spaces);
    spaceend = node.find_last_not_of(spaces);
    if (spacestart == string::npos)
        node="";
    else
        node = node.substr(spacestart, spaceend+1-spacestart);
}

int64_t read_file(char *fname, int64_t wcl, int64_t wcw, map<string, int>&classmap) 
{
    int64_t dims[1];
    ifstream ifs;
    vector<string> retarray;
    map <string, int>::iterator classmap_it;
    int classval[2] = {-1, 1};
    int classval_cnt = 0;
    string token, ttoken, tempy;
    int j;
    size_t colonplace, retarray_sz;
    int64_t nnz = wcw - wcl;

    static int pone_flag = 0, mone_flag = 0;   
    int64_t trans_count = 0;

    assert(nnz > 0);

    row_ptr = (int64_t *)malloc(sizeof(int64_t) * (wcl+1));
    row_ptr[0] = 0;
    assert(row_ptr);

    int64_t reqspace = 2 * nnz + 4 * wcl;
    /* y, a, f, sinfo */

    dims[0] = reqspace;
    /*FIXME: why dataset handle not int64_t*/
    dataset_handle = NGA_Create64(C_DBL, 1, dims, const_cast<char*>("dataset_handle"), NULL);

    /*  linearized sparse matrix  in csr format*/
    assert(row_ptr);


    if (0 == me) {
        double *onedarr;
        //double *onedarr = (double*) malloc (sizeof(double) * reqspace);
        onedarr = (double *)ARMCI_Malloc_local(sizeof(double) * reqspace);

        assert(onedarr);
        int64_t accum_wcw = 0, curr_num_nzs = 0;
        ifs.open(fname);

        while (!ifs.eof())
        {
            getline(ifs, token);

            if (!tokenize(token, retarray))
            {
                if(ifs.eof())
                {
                    break;
                }
                else {
                    cerr <<"Empty line in file....exiting"<<endl;
                    return 0;
                }
            }

            retarray_sz = retarray.size();

            // check for max trans size:FIXME
            if (retarray_sz > (size_t)max_sample_size)
                max_sample_size = retarray_sz;

            accum_wcw += (int64_t)retarray_sz;

            // alpha
            onedarr[curr_num_nzs++] = 0.0;

            tempy = retarray[0];
            /*  FIXME: TODO currently handles only two classes. For number of
             *  classes > 2, have to change the input file maually */
            classmap_it = classmap.find(tempy);
            if (classmap_it == classmap.end() && classval_cnt < 2)/* new class id */
            {
                classmap[tempy] = classval[classval_cnt];
                if (classval[classval_cnt]== 1 && !pone_flag)
                {
                    // Get the first sample with +1 yclass
                    pone_flag = 1;
                    first_pone = trans_count;
                }
                if (classval[classval_cnt]== -1 && !mone_flag)
                {
                    // Get  the first sample with -1 yclass
                    mone_flag = 1;
                    first_mone = trans_count;
                }
                classval_cnt++;
            }
            else
            {
                if(classmap_it == classmap.end() && classval_cnt == 2)
                {
                    cout<<"More than 2 class labels not allowed: "<<tempy<<endl;
                    cout<<"Available classes: "<<endl;
                    for(classmap_it = classmap.begin(); classmap_it!=classmap.end(); classmap_it++)
                       cout<<classmap_it->first<<"=>"<<classmap_it->second<<" ";
                cout<<endl<<"Quitting"<<endl;
                exit(0);
                }
            }
            if (classmap[tempy] == 1) {
                onedarr[curr_num_nzs++] = 1.0;//sinfo
                onedarr[curr_num_nzs++] = -1.0;//fcache, -y_i initially
                onedarr[curr_num_nzs++] = 1.0;//y

            }

            if (classmap[tempy] == -1) {
                onedarr[curr_num_nzs++] = 4.0;
                onedarr[curr_num_nzs++] = 1.0;
                onedarr[curr_num_nzs++] = -1.0;
            }


            for (j = 1; j < (int) retarray.size(); j++)
            {
                ttoken = retarray[j];
                colonplace = ttoken.find_first_of(':');
                if (colonplace == string::npos)
                {
                    cerr <<"File not in sparse format" <<endl;
                    assert(0);
                }
                onedarr[curr_num_nzs++] = atof(ttoken.substr(0, colonplace).c_str())-1.0;
                onedarr[curr_num_nzs++] = atof(ttoken.substr(colonplace+1, 
                            string::npos).c_str());
            }

            row_ptr[trans_count+1] = row_ptr[trans_count] 
                + (int64_t)(retarray_sz-1)*2
                + 4;
            ++trans_count;
            retarray.clear();
            /*  position after padded information */
        }
        assert (row_ptr[trans_count] == reqspace);
        assert(trans_count == wcl);/*  number of lines read should be same as wcl */
        assert(curr_num_nzs == reqspace);
        assert(accum_wcw == wcw);/*  total words encountered so far */

        ifs.close();

        // Finally write to the dataset

        assert(first_pone != -1 && first_mone != -1);
        int64_t mylo[1], myhi[1];
        mylo[0] = 0;
        myhi[0] = reqspace - 1;
        NGA_Put64(dataset_handle, mylo, myhi, onedarr, NULL); 
        /*        cout<<"The onedarr: "<<endl;
                  for(int64_t ii = 0; ii < curr_num_nzs; ii++)
                  cout<<onedarr[ii]<<" ";
                  cout<<endl;*/
        ARMCI_Free_local(onedarr);
        /*        cout<<"The rowptr: "<<endl;
                  for(int64_t ii = 0; ii < (trans_count+1); ii++)
                  cout<<row_ptr[ii]<<" ";
                  cout<<endl;*/
        // Update the max sample size to 2 * max_sample_size + 4
        max_sample_size = 2 * (max_sample_size - 1) + 4;
        /*        cout<<"The max sample size: "<<max_sample_size<<endl;*/
    }

    GA_Sync();
#if 0
    GA_Print(dataset_handle);
#endif

    // row pointer 
    MPI_Bcast(row_ptr, sizeof(int64_t) * (wcl + 1), MPI_BYTE, 0, MPI_COMM_WORLD); 

    // maximum sample size
    MPI_Bcast(&max_sample_size, sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    // Allocate the memory for two samples (i2 and i1)
    sample_i = (double *)malloc(sizeof(double) *  2 * max_sample_size);
    assert(sample_i);

    local_sample = (double *)malloc(sizeof(double) *  max_sample_size);
    assert(local_sample);

    // send the initial values of i2 and i1
    int64_t i2i1[2];
    i2i1[0] = first_pone;
    i2i1[1] = first_mone;

    MPI_Bcast(i2i1, sizeof(int64_t) * 2, MPI_BYTE, 0, MPI_COMM_WORLD);
    i_up = i2i1[0];
    i_low = i2i1[1];

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == me) {
        cout << "Finished reading file" << endl;
        cout << "i_up: "<<i_up << " i_low: " << i_low << endl;
    }
    return 1;
}/* end read_file function */

/* function
 * addFcache : each processor adds its Fcache locally and through a collective
 * operation (MPI_Reduce) final Fcache sum is calculated
 */
double par_addFcache()
{
    int64_t start_index = me * numtrsamp / nproc;
    int64_t end_index = (me + 1) * numtrsamp / nproc;

    if (me == nproc - 1)
        end_index = numtrsamp;

    int64_t i;
    int64_t ld = -1, mylo[1], myhi[1];
    double sum = 0, myalpha;
    int64_t mycnt = 0;

    for(i = start_index; i < end_index; i++) 
    {
        mylo[0] = row_ptr[i];
        myhi[0] = row_ptr[i + 1] - 1;
        NGA_Get64(dataset_handle, mylo, myhi, local_sample, &ld);  

        myalpha = local_sample[0]; 
        if (myalpha > 1e-12 && fabs(myalpha - C) > 1e-12)
        {
            sum += local_sample[2];
            mycnt += 1;
        }
    }

    /* FIXME - should I uncomment the following ? */
    GA_Sync();

    int64_t out_cnt; 
    double out_sum;
    MPI_Reduce(&mycnt, &out_cnt, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum, &out_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (0 == me)
    {
        if (out_cnt == 0)
            out_sum = (b_low + b_up)*0.5;
        else
            out_sum = out_sum/(double)out_cnt;
    }
    /* FIXME: is barrier essential here ? */
    MPI_Barrier(MPI_COMM_WORLD);
    return out_sum;
}/* end par_addFcache*/

/*  
 * Calculate W'X. X here is the sample projected on W)
 *  */
double par_evali(double* sample, int64_t nelems)
{
    int64_t j, urlo[1], urhi[1], urnelems, ld = -1;
    double innernormwsum = 0.0, Kij, ll_alpha;
    /*  local local sample */
    double *lls = (double*) malloc(sizeof(double)* max_sample_size);
    /*  for each entry in rowptr */
    for(j = 0; j < numtrsamp; j++)
    {
        urlo[0] = row_ptr[j];
        urhi[0] = row_ptr[j+1] - 1;
        urnelems = urhi[0] + 1 - urlo[0];
        NGA_Get64(dataset_handle, urlo, urhi, lls, &ld);
        ll_alpha = lls[0];
        if (ll_alpha < ZERO)
            continue;
        Kij = mykernelfunc(sample + 4, nelems - 4, 
                lls + 4, urnelems - 4);
        innernormwsum += Kij * lls[3] * ll_alpha; 
    }
    return innernormwsum;

}/*  end par_evali */

double par_normw()
{
    int64_t start_index = me * numtrsamp / nproc;
    int64_t end_index = (me + 1) * numtrsamp / nproc;

    if (me == nproc - 1)
        end_index = numtrsamp;

    int64_t i;
    int64_t ld = -1, mylo[1], myhi[1], nelems;
    double normwsum = 0.0, myalpha;
    for(i = start_index; i < end_index; i++)
    {
        mylo[0] = row_ptr[i];
        myhi[0] = row_ptr[i + 1] - 1;
        nelems = row_ptr[i + 1] - row_ptr[i];
        NGA_Get64(dataset_handle, mylo, myhi, local_sample, &ld);  
        myalpha = local_sample[0];
        if (myalpha < ZERO)
            continue;
        normwsum += par_evali(local_sample, nelems) * myalpha * local_sample[3];
    }
    GA_Sync();
    double out_normwsum=0;
    cout<<"normwsum:"<<normwsum<<endl;
    MPI_Reduce(&normwsum, &out_normwsum, 1, MPI_DOUBLE, 
            MPI_SUM, 0, MPI_COMM_WORLD);
    return out_normwsum*0.5;
}/* end par_normw*/

/*  currently adds only alpha arr, but can be changed to add slackarr too */
double par_addarrele()
{
    int64_t start_index = me * numtrsamp / nproc;
    int64_t end_index = (me + 1) * numtrsamp / nproc;

    if (me == nproc - 1)
        end_index = numtrsamp;

    int64_t ld = -1, mylo[1], myhi[1], i;
    double sum = 0.0;
    for(i = start_index; i < end_index; i++)
    {
        mylo[0] = row_ptr[i];
        myhi[0] = row_ptr[i] + 1;/*  just a single element is enough */
        NGA_Get64(dataset_handle, mylo, myhi, local_sample, &ld);  
        sum += local_sample[0];
    }
    GA_Sync();
    double out_sum;
    MPI_Reduce(&sum, &out_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return out_sum;
}/*  end par_addarrele */

int64_t gathersvs(void)
{
    int64_t reqspace = 2 * wcw + 2 * wcl;
    double *onedarr = (double*) ARMCI_Malloc_local(sizeof(double) * reqspace);
    int64_t start_index = me * numtrsamp / nproc;
    int64_t end_index = (me + 1) * numtrsamp / nproc;
    int64_t i, allelems, ld = -1, mylo[1], myhi[1];
    int64_t accum_nelems = 0;/*  to adjust the capacity of sendbuf finally */
    int64_t localsvcount = 0, nelems, totsvscount;
    double myalpha, myy;

    if (me == nproc - 1)
        end_index = numtrsamp;
    allelems =  end_index - start_index;/*  local count of samples */
    double *sendbuf = (double*) ARMCI_Malloc_local(sizeof(double) * 
            allelems * max_sample_size);
    int64_t *local_sv_row_ptr = (int64_t *) malloc(sizeof(int64_t) * allelems);
    local_sv_row_ptr[0] = 0;
    int sendbuf_idx = 0;
    for (i = start_index; i < end_index; i++)
    {
        mylo[0] = row_ptr[i];
        myhi[0] = row_ptr[i+1] - 1;
        NGA_Get64(dataset_handle, mylo, myhi, local_sample, &ld);
        myalpha = local_sample[0];
        if (myalpha < ZERO)
            continue;
        nelems = myhi[0] + 1 - mylo[0];
        accum_nelems += nelems - 2;/*  do not need sinfo and Fcache values */
        myy = local_sample[3];
        localsvcount++;
        sendbuf[sendbuf_idx++] = myalpha;
        sendbuf[sendbuf_idx++] = myy;
        memcpy(sendbuf + sendbuf_idx, local_sample + 4, 
                sizeof(double) * (nelems - 4));
        sendbuf_idx += nelems - 4;/*  count for alpha and y is already included */
        local_sv_row_ptr[localsvcount] = local_sv_row_ptr[localsvcount-1] 
            + (nelems-2);
    }
    if(0 == me)
        cout<<"last location of local_sv_row_ptr: "<<local_sv_row_ptr[localsvcount]<<endl;
    /*  total elements to be sent(nnz) should be the last element in rowptr */
    assert(local_sv_row_ptr[localsvcount]==(int64_t)sendbuf_idx);
    /*
     * at this point, localsvcount has the correct value of sv count, but local_sv_row_ptr
     * should accommodate one additional space for nnz
     */
    /*  realloc row_ptr to correct number of elements */
    local_sv_row_ptr = (int64_t*) realloc(local_sv_row_ptr,
            sizeof(int64_t) * (localsvcount+1));
    assert(local_sv_row_ptr != NULL);/*  don't want this to happen */
    GA_Destroy(dataset_handle);
    GA_Sync();
    /* TODO:just a cross-verify operation. remove accum_nelems once the check is successful */
    assert(accum_nelems == (int64_t)sendbuf_idx);
    /* realloc sendbuf, otherwise too much wastage. Use localsvcount or sendbuf_idx*/
    double *tempsendbuf = (double *)ARMCI_Malloc_local(sizeof(double) * sendbuf_idx);
    /*  FIXME: is there an ARMCI memcpy */
    memcpy(tempsendbuf, sendbuf, sizeof(double) * sendbuf_idx);
    ARMCI_Free_local(sendbuf);
    /*  collect the number of rows in each processor */
    MPI_Allreduce(&localsvcount, &totsvscount, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    int64_t totsvscnt_space=0;
    int64_t tempsendbuf_idx = (int64_t)sendbuf_idx;
    MPI_Reduce(&tempsendbuf_idx, &totsvscnt_space, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if ( 0 == me )
    {
        cout<<"total svs: "<<totsvscount<<endl;
        cout<<"totalsvcount space:"<<totsvscnt_space<<endl;
    }
    double *onedsvarr;
    int *displs; int64_t* sv_row_ptr;
    int* rcounts;
    if (0 == me)
    {
        totsvscount += (int64_t)nproc;/*  add the last 1 missed out from each processor */
        onedsvarr = (double *) ARMCI_Malloc_local( sizeof(double) * totsvscnt_space);
        rcounts = (int *) malloc ( nproc * sizeof(int) );
        displs = (int *) malloc ( nproc * sizeof(int) );
        sv_row_ptr = (int64_t *) malloc (sizeof(int64_t) * totsvscount);
    }
    /*  every processor needs to reserve this space */
    temp_sv_row_ptr = (int64_t *) malloc(sizeof(int64_t) * (totsvscount-nproc+1));
    assert(temp_sv_row_ptr);
    /*  gather counts and calculate strides for the sv samples*/
    MPI_Gather(&sendbuf_idx, 1, MPI_LONG, rcounts, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (0==me)
    {
        displs[0] = 0;
        for (i = 1; i < nproc; i++)
        {
            displs[i] = displs[i-1] + rcounts[i-1];
        }
    }
    /* gather sv vectors all in one array FIXME: rcounts and displs are forced
     * int TODO: check for dataloss */
    MPI_Gatherv(tempsendbuf, sendbuf_idx, MPI_DOUBLE, onedsvarr, rcounts, 
            displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /*  gather localsvcount and calculate stride for the row pointer */
    MPI_Gather(&localsvcount, 1, MPI_LONG, rcounts, 1, MPI_LONG, 0, MPI_COMM_WORLD); 
    if (0==me)
    {
        mylo[0] = 0;
        myhi[0] = totsvscnt_space-1;  
        /*  write svs to GA - call FIXME: TODO NGA_Create before this line*/
        NGA_Put64(svset_handle, mylo, myhi, onedsvarr, NULL);

        displs[0] = 0;
        for (i = 1; i < nproc; i++)
        {
            displs[i] = displs[i-1] + rcounts[i-1] + 1;/*  adjust the missing one */
        }
    }
    localsvcount = localsvcount + 1;/*  space for last element in rowptr */
    MPI_Gatherv(local_sv_row_ptr, localsvcount, MPI_LONG, sv_row_ptr, rcounts, displs, 
            MPI_LONG, 0, MPI_COMM_WORLD);
    int64_t rowj;
    /*  adjust the boundaries of received row pointers */
    if (0 == me)
    {
        rowj = 0;
        int nproc_idx = 1;
        i = 0;
        int64_t increment = 0;
        /*  totsvscount includes nproc value, skip the nnz of last proc alone*/
        while (i < (totsvscount-1))        
        {
            if (i == displs[nproc_idx]-1 && nproc_idx < nproc)
            {
                increment += sv_row_ptr[i];/*  accumulate the increment at the boundary */
                nproc_idx++;
                i = i + 1 ;
            }
            temp_sv_row_ptr[rowj++] = increment + sv_row_ptr[i];
            i = i + 1;
        }
        increment += sv_row_ptr[i];/*  take the nnz of the last proc*/
        temp_sv_row_ptr[rowj] = increment;/*  assign the cumulative nnz */ 
        assert(rowj == totsvscount - nproc);
        assert(increment == totsvscnt_space);
    }
    MPI_Bcast(temp_sv_row_ptr, sizeof(int64_t) * (rowj+1), MPI_BYTE, 0, MPI_COMM_WORLD);
    ARMCI_Free_local(tempsendbuf);
    ARMCI_Free_local(onedsvarr);
    ARMCI_Free_local(onedarr);
    free(rcounts);
    free(displs);
    free(sv_row_ptr);
    free(local_sv_row_ptr);
    /*  get the actual value back */
    totsvscount -= (int64_t)nproc;
    MPI_Barrier(MPI_COMM_WORLD);
    return totsvscount;
}/*  end function gathersvs */

int read_testfile(char *fname, int64_t wcl, int64_t wcw, 
        const map<string, int>& classmap) 
{
    int64_t dims[1];
    ifstream ifs;
    vector<string> retarray;
    map <string, int>::const_iterator classmap_it;
    string token, ttoken, tempy;
    int j;
    size_t colonplace, retarray_sz;
    int64_t nnz = wcw - wcl;

    int64_t trans_count = 0;

    assert(nnz > 0);

    test_row_ptr = (int64_t *)malloc(sizeof(int64_t) * (wcl+1));
    test_row_ptr[0] = 0;
    assert(test_row_ptr);

    int64_t reqspace = 2 * nnz + wcl;/*  class label and the samples */

    dims[0] = reqspace;
    /*FIXME: why dataset handle not int64_t*/
    testdataset_handle = NGA_Create64(C_DBL, 1, dims, const_cast<char*>("testdataset_handle"), NULL);

    /*  linearized sparse matrix  in csr format*/
    assert(testdataset_handle);


    if (0 == me) {
        double *onedarr;
        onedarr = (double *)ARMCI_Malloc_local(sizeof(double) * reqspace);

        assert(onedarr);
        int64_t accum_wcw = 0, curr_num_nzs = 0;
        ifs.open(fname);

        while (!ifs.eof())
        {
            getline(ifs, token);

            if (!tokenize(token, retarray))
            {
                if(ifs.eof())
                {
                    break;
                }
                else {
                    cerr <<"Empty line in file....exiting"<<endl;
                    return 0;
                }
            }

            retarray_sz = retarray.size();

            // check for max trans size:FIXME
            if (retarray_sz > (size_t)test_max_sample_size)
                test_max_sample_size = retarray_sz;

            accum_wcw += (int64_t)retarray_sz;

            tempy = retarray[0];
            classmap_it = classmap.find(tempy);
            assert(classmap_it != classmap.end());/* new class id */
            if (classmap_it->second == 1) {
                onedarr[curr_num_nzs++] = 1.0;//y
            }

            if (classmap_it->second == -1) {
                onedarr[curr_num_nzs++] = -1.0;
            }

            for (j = 1; j < (int) retarray.size(); j++)
            {
                ttoken = retarray[j];
                colonplace = ttoken.find_first_of(':');
                if (colonplace == string::npos)
                {
                    cerr <<"File not in sparse format" <<endl;
                    assert(0);
                }
                onedarr[curr_num_nzs++] =
                    atof(ttoken.substr(0, colonplace).c_str())-1.0;
                onedarr[curr_num_nzs++] = atof(ttoken.substr(colonplace+1, 
                            string::npos).c_str());
            }

            test_row_ptr[trans_count+1] =test_row_ptr[trans_count] 
                + (int64_t)(retarray_sz-1)*2
                + 1;
            ++trans_count;
            retarray.clear();
            /*  position after padded information */
        }
        assert (test_row_ptr[trans_count] == reqspace);
        assert(trans_count == wcl);/*  number of lines read should be same as wcl */
        assert(curr_num_nzs == reqspace);
        assert(accum_wcw == wcw);/*  total words encountered so far */

        ifs.close();

        // Finally write to the dataset

        int64_t mylo[1], myhi[1];
        mylo[0] = 0;
        myhi[0] = reqspace - 1;
        NGA_Put64(testdataset_handle, mylo, myhi, onedarr, NULL); 
#ifdef PRINT
        cout<<"The onedarr: "<<endl;
        for(int64_t ii = 0; ii < curr_num_nzs; ii++)
            cout<<onedarr[ii]<<" ";
        cout<<endl;
#endif
        ARMCI_Free_local(onedarr);
#ifdef PRINT
        cout<<"The rowptr: "<<endl;
        for(int64_t ii = 0; ii < (trans_count+1); ii++)
            cout<<test_row_ptr[ii]<<" ";
        cout<<endl;
#endif
        test_max_sample_size = 2 * (test_max_sample_size - 1) + 1;
    }

    GA_Sync();
    // row pointer 
    MPI_Bcast(test_row_ptr, sizeof(int64_t) * (wcl + 1), MPI_BYTE, 0, MPI_COMM_WORLD); 
    // maximum sample size
    MPI_Bcast(&test_max_sample_size, sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    testlocal_sample = (double *)malloc(sizeof(double) *  test_max_sample_size);
    assert(testlocal_sample);

    MPI_Barrier(MPI_COMM_WORLD);
    return 1;
}/* end read_testfile function */

/*************************************************************/
int64_t gathersvs_jan10_2013(void)
{
    int64_t start_index = me * numtrsamp / nproc;
    int64_t end_index = (me + 1) * numtrsamp / nproc;
    int64_t i, allelems, ld = -1, mylo[1], myhi[1];
    int64_t accum_nelems = 0;/*  to adjust the capacity of sendbuf finally */
    int64_t localsvcount = 0, nelems, totsvscount;
    double myalpha, myy;

    if (me == nproc - 1)
        end_index = numtrsamp;
    allelems =  end_index - start_index;/*  local count of samples */
    double *sendbuf = (double*) ARMCI_Malloc_local(sizeof(double) * 
            allelems * max_sample_size);
    int64_t *local_sv_row_ptr = (int64_t *) malloc(sizeof(int64_t) * allelems);
    local_sv_row_ptr[0] = 0;
    int sendbuf_idx = 0;
    for (i = start_index; i < end_index; i++)
    {
        mylo[0] = row_ptr[i];
        myhi[0] = row_ptr[i+1] - 1;
        NGA_Get64(dataset_handle, mylo, myhi, local_sample, &ld);
        myalpha = local_sample[0];
        if (myalpha < ZERO)
            continue;
        nelems = myhi[0] + 1 - mylo[0];
        accum_nelems += nelems - 2;/*  do not need sinfo and Fcache values */
        myy = local_sample[3];
        localsvcount++;
        sendbuf[sendbuf_idx++] = myalpha;
        sendbuf[sendbuf_idx++] = myy;
        memcpy(sendbuf + sendbuf_idx, local_sample + 4, 
                sizeof(double) * (nelems - 4));
        sendbuf_idx += nelems - 4;/*  count for alpha and y is already included */
        local_sv_row_ptr[localsvcount] = local_sv_row_ptr[localsvcount-1] 
            + (nelems-2);
    }
    if(0 == me)
        cout<<"last location of local_sv_row_ptr: "<<local_sv_row_ptr[localsvcount]<<endl;
    /*  total elements to be sent(nnz) should be the last element in rowptr */
    assert(local_sv_row_ptr[localsvcount]==(int64_t)sendbuf_idx);
    /*
     * at this point, localsvcount has the correct value of sv count, but local_sv_row_ptr
     * should accommodate one additional space for nnz
     */
    /*  realloc row_ptr to correct number of elements */
    local_sv_row_ptr = (int64_t*) realloc(local_sv_row_ptr,
            sizeof(int64_t) * (localsvcount+1));
    assert(local_sv_row_ptr != NULL);/*  don't want this to happen */
    GA_Destroy(dataset_handle);
    GA_Sync();
    /* TODO:just a cross-verify operation. remove accum_nelems once the check is successful */
    assert(accum_nelems == (int64_t)sendbuf_idx);
    /* realloc sendbuf, otherwise too much wastage. Use localsvcount or sendbuf_idx*/
    double *tempsendbuf = (double *)ARMCI_Malloc_local(sizeof(double) * sendbuf_idx);
    /*  FIXME: is there an ARMCI memcpy */
    memcpy(tempsendbuf, sendbuf, sizeof(double) * sendbuf_idx);
    ARMCI_Free_local(sendbuf);
    /*  collect the number of rows in each processor */
    MPI_Allreduce(&localsvcount, &totsvscount, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    int64_t totsvscnt_space=0;
    int64_t tempsendbuf_idx = (int64_t)sendbuf_idx;
    MPI_Reduce(&tempsendbuf_idx, &totsvscnt_space, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if ( 0 == me )
    {
        cout<<"total svs: "<<totsvscount<<endl;
        cout<<"totalsvcount space:"<<totsvscnt_space<<endl;
    }
    double *onedsvarr;
    int *displs; int64_t* sv_row_ptr;
    int* rcounts;
    if (0 == me)
    {
        totsvscount += (int64_t)nproc;/*  add the last 1 missed out from each processor */
        onedsvarr = (double *) ARMCI_Malloc_local( sizeof(double) * totsvscnt_space);
        rcounts = (int *) malloc ( nproc * sizeof(int) );
        displs = (int *) malloc ( nproc * sizeof(int) );
        sv_row_ptr = (int64_t *) malloc (sizeof(int64_t) * totsvscount);
    }
    /*  every processor needs to reserve this space */
    temp_sv_row_ptr = (int64_t *) malloc(sizeof(int64_t) * (totsvscount-nproc+1));
    assert(temp_sv_row_ptr);
    /*  gather counts and calculate strides for the sv samples*/
    MPI_Gather(&sendbuf_idx, 1, MPI_LONG, rcounts, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (0==me)
    {
        displs[0] = 0;
        for (i = 1; i < nproc; i++)
        {
            displs[i] = displs[i-1] + rcounts[i-1];
        }
    }
    /* gather sv vectors all in one array FIXME: rcounts and displs are forced
     * int TODO: check for dataloss */
    MPI_Gatherv(tempsendbuf, sendbuf_idx, MPI_DOUBLE, onedsvarr, rcounts, 
            displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /*  gather localsvcount and calculate stride for the row pointer */
    MPI_Gather(&localsvcount, 1, MPI_LONG, rcounts, 1, MPI_LONG, 0, MPI_COMM_WORLD); 
    if (0==me)
    {
        mylo[0] = 0;
        myhi[0] = totsvscnt_space-1;  
        /*  write svs to GA - call FIXME: TODO NGA_Create before this line*/
        NGA_Put64(svset_handle, mylo, myhi, onedsvarr, NULL);

        displs[0] = 0;
        for (i = 1; i < nproc; i++)
        {
            displs[i] = displs[i-1] + rcounts[i-1] + 1;/*  adjust the missing one */
        }
    }
    localsvcount = localsvcount + 1;/*  space for last element in rowptr */
    MPI_Gatherv(local_sv_row_ptr, localsvcount, MPI_LONG, sv_row_ptr, rcounts, displs, 
            MPI_LONG, 0, MPI_COMM_WORLD);
    int64_t rowj;
    /*  adjust the boundaries of received row pointers */
    if (0 == me)
    {
        rowj = 0;
        int nproc_idx = 1;
        i = 0;
        int64_t increment = 0;
        /*  totsvscount includes nproc value, skip the nnz of last proc alone*/
        while (i < (totsvscount-1))        
        {
            if (i == displs[nproc_idx]-1 && nproc_idx < nproc)
            {
                increment += sv_row_ptr[i];/*  accumulate the increment at the boundary */
                nproc_idx++;
                i = i + 1 ;
            }
            temp_sv_row_ptr[rowj++] = increment + sv_row_ptr[i];
            i = i + 1;
        }
        increment += sv_row_ptr[i];/*  take the nnz of the last proc*/
        temp_sv_row_ptr[rowj] = increment;/*  assign the cumulative nnz */ 
        assert(rowj == totsvscount - nproc);
        assert(increment == totsvscnt_space);
    }
    MPI_Bcast(temp_sv_row_ptr, sizeof(int64_t) * (rowj+1), MPI_BYTE, 0, MPI_COMM_WORLD);
    ARMCI_Free_local(tempsendbuf);
    ARMCI_Free_local(onedsvarr);
    free(rcounts);
    free(displs);
    free(sv_row_ptr);
    free(local_sv_row_ptr);
    /*  get the actual value back */
    totsvscount -= (int64_t)nproc;
    MPI_Barrier(MPI_COMM_WORLD);
    return totsvscount;
}/*  end function gathersvs_jan10_2013 */



/*
 * for testing, alternatives are 
 * 1) sparse matrix-matrix multiplication packages
 * like sparselib++ 
 *
 * 2) broadcast the sv indices and split the test set among
 * processes. Rearrange the sv indices such that they are contiguous. Adapt the
 * idea of norw calculation. Since number of svs are small, sharing the
 * ownership of test samples might equally divide the work rather than the other
 * way.
 *
 * creating a new onedarr to hold svs exclusively?
 *
 * 3) sparsekit from Saad
 */
