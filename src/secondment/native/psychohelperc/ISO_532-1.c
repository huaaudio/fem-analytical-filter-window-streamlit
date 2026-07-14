/************************************************************************/
/*  Loudness calculation according to ISO 532-1,                        */
/*  methods for stationary and time varying signals                     */
/************************************************************************/

#include "ISO_532-1.h"

#include <math.h>     /* for sqrt, exp, pow, floor, log10               */
#include <stdlib.h>   /* for memory allocation                          */

/*  Number of lower critical bands consisting of more than one          */
/*  third octave band                                                   */
#define N_LCBS          3

/*  Number of third octave bands for N_LCBS lower bands                 */
#define N_LCB_BANDS     11

/*  Number of core loudness values                                      */
#define N_CORE_LOUDN    21

/*  Number of level ranges for consideration of equal loudness contours */
#define N_RAP_RANGES    8

/*  Number of loudness and critical band ranges for calculating         */
/*  steepness of upper slopes in the specific loudness -                */
/*  critical-band-rate pattern                                          */
#define N_RNS_RANGES    18
#define N_CB_RANGES     8

/*  Constants for third octave filters                                  */
#define N_FILTER_STAGES 3
#define N_FILTER_COEFS  6

/*  Factors for virtual upsampling/inner iterations                     */
#define NL_ITER         24
#define LP_ITER         24

/*  Time constants for non-linear temporal decay                        */
#define TSHORT          0.005
#define TLONG           0.015
#define TVAR            0.075

//////////////////////////////////////////////////////////////////////////
//  BLOCK   Memory management
//////////////////////////////////////////////////////////////////////////

static void freeRaggedArray(double** raggedArray, int numArrays)
{
    int i;
    for (i = 0; i < numArrays; i++)
    {
        if (raggedArray[i] != NULL)
        {
            free(raggedArray[i]);
        }
    }
}

static int callocRaggedArray(double** raggedArray, int numArrays, int lengthArray)
{
    int i;
    for (i = 0; i < numArrays; i++)
    {
        raggedArray[i] = (double*)calloc(lengthArray, sizeof(double));
        if (raggedArray[i] == NULL)
        {
            freeRaggedArray(raggedArray, i - 1);
            return LoudnessErrorMemoryAllocFailed;
        }
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////
//  BLOCK   NL
//////////////////////////////////////////////////////////////////////////

/* Inputfile structure                                                  */
struct NlLpData
{
    double B[6]         /*  coefficients    */
         , UoLast       /*  Uo, last sample */
         , U2Last;      /*  U2, last sample */
};

/*  Initializes constants B and states of capacitors C1 and C2,
    1/delta_t = sampling rate                                           */
struct NlLpData f_init_nl_lp(double SampleRate)
{
    double Lambda1, Lambda2, P, Q, Den, E1, E2, DeltaT;
    double Tvar     = TVAR;
    double Tshort   = TSHORT;
    double Tlong    = TLONG;

    struct NlLpData NlLp;
    
    DeltaT      = 1 / SampleRate;
    P           = (Tvar + Tlong) / (Tvar*Tshort);
    Q           = 1/(Tshort*Tvar);
    Lambda1     =-P/2 + sqrt(P*P/4 - Q);
    Lambda2     =-P/2 - sqrt(P*P/4 - Q);
    Den         = Tvar * (Lambda1 - Lambda2);
    E1          = exp(Lambda1 * DeltaT);
    E2          = exp(Lambda2 * DeltaT);

    NlLp.B[0]   = (E1 - E2) / Den;
    NlLp.B[1]   =((Tvar * Lambda2 + 1) * E1 - (Tvar * Lambda1 + 1) * E2) / Den;
    NlLp.B[2]   =((Tvar * Lambda1 + 1) * E1 - (Tvar * Lambda2 + 1) * E2) / Den;
    NlLp.B[3]   = (Tvar * Lambda1+1) * (Tvar * Lambda2 + 1) * (E1-E2) / Den;
    NlLp.B[4]   = exp(-DeltaT / Tlong);
    NlLp.B[5]   = exp(-DeltaT / Tvar);

    /*  At beginning capacitors C1 and C2 are discharged    */
    NlLp.UoLast = 0; 
    NlLp.U2Last = 0;

    return NlLp;
}

/*  NL: Calculates Uo(t) from Ui(t) using UoLast and U2Last             */
double f_nl_lp(double Ui, struct NlLpData *pNlLp)
{
    double Uo, U2;

    if (Ui < pNlLp->UoLast)                 /*  case 1          */
	{
        if (pNlLp->UoLast > pNlLp->U2Last)  /*  case 1.1        */
        {
            U2 = pNlLp->UoLast*pNlLp->B[0] - pNlLp->U2Last*pNlLp->B[1];
            Uo = pNlLp->UoLast*pNlLp->B[2] - pNlLp->U2Last*pNlLp->B[3];
            if (Uo < Ui)                    /*  Uo can't become */
                Uo = Ui;                    /*  lower than Ui   */
            if (U2 > Uo)                    /*  U2 can't become */
                U2 = Uo;                    /*  higher than Uo  */
        }
        else                                /*  Case 1.2        */
        {
            Uo = pNlLp->UoLast*pNlLp->B[4];
            if (Uo < Ui)                    /*  Uo can't become */
                Uo = Ui;                    /*  lower than Ui   */
            U2 = Uo;
        }
	}
    else
    {
        if (fabs(Ui - pNlLp->UoLast)<1e-5)            /*  Case 2          */
        {
            Uo = Ui;
            if (Uo > pNlLp->U2Last)         /*  Case 2.1        */
                U2 = (pNlLp->U2Last - Ui)*pNlLp->B[5] +  Ui;
            else                            /*  Case 2.2        */
                U2 = Ui;
        }
        else                                /*  Case 3          */
        {
            Uo = Ui;
            U2 = (pNlLp->U2Last - Ui)*pNlLp->B[5] + Ui;
        }
    }

    pNlLp->UoLast = Uo;                     /*  Preparation for */
    pNlLp->U2Last = U2;                     /*  next step       */

    return(Uo);
}

/*  Non-linear temporal decay, block NL, uses inner iterations/linear
    interpolation for increased precision                               */
void f_nl(double **CoreLoudness, double SampleRate, int NumSamples)
{
    int     IdxCL, IdxTime, IdxI;
    double  *pCoreL, NextInput, Delta, Ui, Uo;
    struct  NlLpData    NlLp = f_init_nl_lp(SampleRate * NL_ITER);
        
    for (IdxCL = 0; IdxCL < N_CORE_LOUDN; IdxCL++)
    {
        pCoreL = CoreLoudness[IdxCL];

        /*  instead of calling f_init_nl_lp */
        NlLp.UoLast = 0;
        NlLp.U2Last = 0;

        for (IdxTime = 0; IdxTime < NumSamples-1; IdxTime++)
        {
            /*  next sample                                             */
            NextInput = *(pCoreL + 1);
            /*  interpolation steps between current and next sample     */
            Delta = (NextInput - *pCoreL) / (double)NL_ITER;

            Ui = *pCoreL;
            *pCoreL = f_nl_lp(Ui, &NlLp);
            Ui += Delta;

            /*  inner iterations                                        */
            for (IdxI = 1; IdxI < NL_ITER; IdxI++)
            {
                Uo = f_nl_lp(Ui, &NlLp);
                Ui += Delta;
            }
            pCoreL++;
        }
        *pCoreL = f_nl_lp(*pCoreL, &NlLp);
    }
}

//////////////////////////////////////////////////////////////////////////
//  BLOCK   Third-octave filtering
//////////////////////////////////////////////////////////////////////////

/*  Reference filter coefficient table  */
/*  b0r, b1r, b2r, a0r, a1r, a2r        */
static double ThirdOctaveFilterRef[N_FILTER_STAGES][N_FILTER_COEFS]=
{   {1, 2, 1, 1, -2, 1},
    {1, 0,-1, 1, -2, 1},
    {1,-2, 1, 1, -2, 1}
};

/*  Filter tables of difference from reference table for third octave 
    filter for sampling rate = 48kHz                                    */
/*  b0d, b1d, b2d, a0d, a1d, a2d                                        */
static double ThirdOctaveFilters[N_LEVEL_BANDS][N_FILTER_STAGES][N_FILTER_COEFS]=
{   { {0,0,0,0,-6.70260e-004,6.59453e-004},
      {0,0,0,0,-3.75071e-004,3.61926e-004},
      {0,0,0,0,-3.06523e-004,2.97634e-004} },

    { {0,0,0,0,-8.47258e-004,8.30131e-004},
      {0,0,0,0,-4.76448e-004,4.55616e-004},
      {0,0,0,0,-3.88773e-004,3.74685e-004} },

    { {0,0,0,0,-1.07210e-003,1.04496e-003},
      {0,0,0,0,-6.06567e-004,5.73553e-004},
      {0,0,0,0,-4.94004e-004,4.71677e-004} },

    { {0,0,0,0,-1.35836e-003,1.31535e-003},
      {0,0,0,0,-7.74327e-004,7.22007e-004},
      {0,0,0,0,-6.29154e-004,5.93771e-004} },

    { {0,0,0,0,-1.72380e-003,1.65564e-003},
      {0,0,0,0,-9.91780e-004,9.08866e-004},
      {0,0,0,0,-8.03529e-004,7.47455e-004} },

    { {0,0,0,0,-2.19188e-003,2.08388e-003},
      {0,0,0,0,-1.27545e-003,1.14406e-003},
      {0,0,0,0,-1.02976e-003,9.40900e-004} },

    { {0,0,0,0,-2.79386e-003,2.62274e-003},
      {0,0,0,0,-1.64828e-003,1.44006e-003},
      {0,0,0,0,-1.32520e-003,1.18438e-003} },

    { {0,0,0,0,-3.57182e-003,3.30071e-003},
      {0,0,0,0,-2.14252e-003,1.81258e-003},
      {0,0,0,0,-1.71397e-003,1.49082e-003} },

    { {0,0,0,0,-4.58305e-003,4.15355e-003},
      {0,0,0,0,-2.80413e-003,2.28135e-003},
      {0,0,0,0,-2.23006e-003,1.87646e-003} },

    { {0,0,0,0,-5.90655e-003,5.22622e-003},
      {0,0,0,0,-3.69947e-003,2.87118e-003},
      {0,0,0,0,-2.92205e-003,2.36178e-003} },

    { {0,0,0,0,-7.65243e-003,6.57493e-003},
      {0,0,0,0,-4.92540e-003,3.61318e-003},
      {0,0,0,0,-3.86007e-003,2.97240e-003} },

    { {0,0,0,0,-1.00023e-002,8.29610e-003},
      {0,0,0,0,-6.63788e-003,4.55999e-003},
      {0,0,0,0,-5.15982e-003,3.75306e-003} },

    { {0,0,0,0,-1.31230e-002,1.04220e-002},
      {0,0,0,0,-9.02274e-003,5.73132e-003},
      {0,0,0,0,-6.94543e-003,4.71734e-003} },

    { {0,0,0,0,-1.73693e-002,1.30947e-002},
      {0,0,0,0,-1.24176e-002,7.20526e-003},
      {0,0,0,0,-9.46002e-003,5.93145e-003} },

    { {0,0,0,0,-2.31934e-002,1.64308e-002},
      {0,0,0,0,-1.73009e-002,9.04761e-003},
      {0,0,0,0,-1.30358e-002,7.44926e-003} },

    { {0,0,0,0,-3.13292e-002,2.06370e-002},
      {0,0,0,0,-2.44342e-002,1.13731e-002},
      {0,0,0,0,-1.82108e-002,9.36778e-003} },

    { {0,0,0,0,-4.28261e-002,2.59325e-002},
      {0,0,0,0,-3.49619e-002,1.43046e-002},
      {0,0,0,0,-2.57855e-002,1.17912e-002} },

    { {0,0,0,0,-5.91733e-002,3.25054e-002},
      {0,0,0,0,-5.06072e-002,1.79513e-002},
      {0,0,0,0,-3.69401e-002,1.48094e-002} },

    { {0,0,0,0,-8.26348e-002,4.05894e-002},
      {0,0,0,0,-7.40348e-002,2.24476e-002},
      {0,0,0,0,-5.34977e-002,1.85371e-002} },

    { {0,0,0,0,-1.17018e-001,5.08116e-002},
      {0,0,0,0,-1.09516e-001,2.81387e-002},
      {0,0,0,0,-7.85097e-002,2.32872e-002} },

    { {0,0,0,0,-1.67714e-001,6.37872e-002},
      {0,0,0,0,-1.63378e-001,3.53729e-002},
      {0,0,0,0,-1.16419e-001,2.93723e-002} },

    { {0,0,0,0,-2.42528e-001,7.98576e-002},
      {0,0,0,0,-2.45161e-001,4.43370e-002},
      {0,0,0,0,-1.73972e-001,3.70015e-002} },

    { {0,0,0,0,-3.53142e-001,9.96330e-002},
      {0,0,0,0,-3.69163e-001,5.53535e-002},
      {0,0,0,0,-2.61399e-001,4.65428e-002} },

    { {0,0,0,0,-5.16316e-001,1.24177e-001},
      {0,0,0,0,-5.55473e-001,6.89403e-002},
      {0,0,0,0,-3.93998e-001,5.86715e-002} },

    { {0,0,0,0,-7.56635e-001,1.55023e-001},
      {0,0,0,0,-8.34281e-001,8.58123e-002},
      {0,0,0,0,-5.94547e-001,7.43960e-002} },

    { {0,0,0,0,-1.10165e+000,1.91713e-001},
      {0,0,0,0,-1.23939e+000,1.05243e-001},
      {0,0,0,0,-8.91666e-001,9.40354e-002} },

    { {0,0,0,0,-1.58477e+000,2.39049e-001},
      {0,0,0,0,-1.80505e+000,1.28794e-001},
      {0,0,0,0,-1.32500e+000,1.21333e-001} },

    { {0,0,0,0,-2.50630e+000,1.42308e-001},
      {0,0,0,0,-2.19464e+000,2.76470e-001},
      {0,0,0,0,-1.90231e+000,1.47304e-001} },
};

/*  Filter gains/scale values                                           */
static double FilterGain[N_LEVEL_BANDS][N_FILTER_STAGES]=
{   {4.30764e-011,1,1},
    {8.59340e-011,1,1},
    {1.71424e-010,1,1},
    {3.41944e-010,1,1},
    {6.82035e-010,1,1},
    {1.36026e-009,1,1},
    {2.71261e-009,1,1},
    {5.40870e-009,1,1},
    {1.07826e-008,1,1},
    {2.14910e-008,1,1},
    {4.28228e-008,1,1},
    {8.54316e-008,1,1},
    {1.70009e-007,1,1},
    {3.38215e-007,1,1},
    {6.71990e-007,1,1},
    {1.33531e-006,1,1},
    {2.65172e-006,1,1},
    {5.25477e-006,1,1},
    {1.03780e-005,1,1},
    {2.04870e-005,1,1},
    {4.05198e-005,1,1},
    {7.97914e-005,1,1},
    {1.56511e-004,1,1},
    {3.04954e-004,1,1},
    {5.99157e-004,1,1},
    {1.16544e-003,1,1},
    {2.27488e-003,1,1},
    {3.91006e-003,1,1} 
};

/*  1st order low-pass                                                  */
void f_lowpass( double *pInput, double *pOutput, double Tau,
                double SampleRate, int NumSamples)
{
    int     IdxTime;
    double  A1, B0, Y1 = 0;
    double  *pX = pInput
         ,  *pY = pOutput;

    A1 = exp(-1 / (SampleRate * Tau));
    B0 = 1 - A1;

    for (IdxTime = 0; IdxTime < NumSamples; IdxTime++)
    {
        *pY = B0 * *pX + A1 * Y1;
        Y1  = *pY;
        pX++;
        pY++;
    }
}

/*  2nd order filtering                                                 */
void f_filter_2ndOrder( double *pInput, double *pOutput, double *Coeffs,
                        int NumSamples, double Gain)
{
    int     IdxTime;
    double  *pX = pInput;
    double  *pY = pOutput;
    double  Wn0 = 0
         ,  Wn1 = 0
         ,  Wn2 = 0;

    for (IdxTime = 0; IdxTime < NumSamples; IdxTime++)
    {
        Wn0 = *pX*Gain - Coeffs[4]*Wn1 - Coeffs[5]*Wn2; /* coeffs[3]=1  */
        *pY = Coeffs[0]*Wn0 + Coeffs[1]*Wn1 + Coeffs[2]*Wn2;
        Wn2 = Wn1;
        Wn1 = Wn0;
        pX++;
        pY++;
    }
}

/*  Squaring and smoothing by three 1st order lowpass filters           */
int f_square_and_smooth(    double *pInput, double CenterFrequency,
                            double SampleRate, int NumSamples,
                            int Method, double TimeSkip)
{
    int     IdxTime, IdxFS, NumSkip;
    double  *pX = pInput, out;
    double  Tau;
    
    if (Method == LoudnessMethodTimeVarying)
    {
        /*  Frequency dependant time-constant                           */
        if (CenterFrequency <= 1000)
        {
            Tau = 2/(3*CenterFrequency);
        }
        else
        {
            Tau = 2/(3*1000.);
        }

        /*  Squaring                                                    */
        for (IdxTime = 0; IdxTime < NumSamples; IdxTime++)
        {
             *pX = pow(*pX,2);
             pX++;
        }

        /*  Three smoothing low-pass filters                            */
        for (IdxFS = 0; IdxFS < 3; IdxFS++)
        {
            f_lowpass (pInput,pInput,Tau,SampleRate,NumSamples);
        }
    }
    else
    {
        out = 0;
        NumSkip = (int)floor(TimeSkip * SampleRate);
        if (NumSkip >= NumSamples)
            return LoudnessErrorSignalTooShort;

        pX += NumSkip;
        for (IdxTime = NumSkip; IdxTime < NumSamples; IdxTime++)
        {
            out += pow(*pX,2);
            pX++;
        }
        *pInput = out / (NumSamples-NumSkip);
    }
    return 0;
}

/*  3rd octave filtering, squaring, smoothing, level calculation and
    downsampling by DecFactor to SR_LEVEL                               */

#define TINY_VALUE  1e-12

int f_calc_third_octave_levels( struct InputData *pSignal,
                                double **ThirdOctaveLevel,
                                int DecFactor, int Method,
                                double TimeSkip)
{
    short   IdxFB, IdxFS, IdxC;
    int     NumSamples, NumDecSamples;
    int     IdxTime, retval;
    double  Coeffs[N_FILTER_COEFS], Gain, CenterFrequency;
    double  *pInput, *pThirdOctaveLevel, *pOutput, *pSPL;
    
    NumSamples      = pSignal->NumSamples;
    NumDecSamples   = NumSamples/DecFactor;

    pOutput     = (double*)calloc(NumSamples,sizeof(double));
    if (pOutput == NULL)
    {
        return LoudnessErrorMemoryAllocFailed;
    }

    for (IdxFB = 0; IdxFB < N_LEVEL_BANDS; IdxFB++)     /*  All filters */
    {
        pInput  = pSignal->pData;
        
        /*  Filter stages                                               */
        for (IdxFS = 0; IdxFS < N_FILTER_STAGES; IdxFS++)
        {
            for (IdxC = 0; IdxC < N_FILTER_COEFS; IdxC++)
            {
                Coeffs[IdxC] =  ThirdOctaveFilterRef[IdxFS][IdxC] - 
                                ThirdOctaveFilters[IdxFB][IdxFS][IdxC];
            }
            Gain    = FilterGain[IdxFB][IdxFS];

            /*  2nd order filtering                                     */
            f_filter_2ndOrder(pInput,pOutput,Coeffs,NumSamples,Gain);
            pInput  = pOutput;
        }

        /*  Calculate center frequency of filter                        */
        CenterFrequency = pow(10, ((double)IdxFB-16)/10.) * 1000;

        /*  Squaring and smoothing of filtered signal                   */
        retval = f_square_and_smooth(pInput,CenterFrequency,
            pSignal->SampleRate,NumSamples,Method,TimeSkip);
        if (retval < 0)
        {
            if (pOutput != NULL)
                free (pOutput);
            return retval;
        }

        pThirdOctaveLevel   = ThirdOctaveLevel[IdxFB];

        /*  SPL calculation                                             */
        pSPL    = pInput;

        for (IdxTime = 0; IdxTime < NumDecSamples; IdxTime++)
        {
            *pThirdOctaveLevel = 10*log10((*pSPL + TINY_VALUE) / I_REF);
            pSPL    += DecFactor;
            pThirdOctaveLevel++;
        }
    }
    if (pOutput != NULL)
        free (pOutput);
    return 0;
}

//////////////////////////////////////////////////////////////////////////
//  BLOCK   Temporal weighting of loudness
//////////////////////////////////////////////////////////////////////////

/*  1st order low-pass with linear interpolation of signal for
    increased precision                                                 */
void f_lowpass_intp(double *pInput, double *pOutput, double Tau,
                    double SampleRate, int NumSamples)
{
    int     IdxTime, IdxI;
    double  A1, B0, X0, Xd, Y1  = 0;
    double  *pX = pInput
         ,  *pY = pOutput;

    A1 = exp(-1 / (SampleRate * LP_ITER * Tau));
    B0 = 1 - A1;

    for (IdxTime = 0; IdxTime < NumSamples; IdxTime++)
    {
        X0  = *pX;
        pX++;

        Y1  = B0 * X0 + A1 * Y1;

        *pY = Y1;
        pY++;

        /*  Linear interpolation steps between current and next sample  */
        if (IdxTime < NumSamples - 1)
        {
            Xd  = (*pX - X0) / (double)LP_ITER;

            /*  Inner iterations/interpolation                          */
            for (IdxI = 1; IdxI < LP_ITER; IdxI++)
            {
                X0  += Xd;
                Y1  = B0 * X0 + A1 * Y1;
            }
        }
    }
}

/*  Two 1st order lowpasses (with interpolation) and weighted summation 
    for duration dependent behavior of loudness perception              */
int f_temporal_weight_loudness( double *Loudness, double SampleRate,
                                int NumSamples)
{
    int     IdxTime;
    double  Tau;
    double  *pLoudness = Loudness
          , *pLoudness_1, *pLoudness_2, *pLoudness_t1, *pLoudness_t2;

    /*  Memory allocation                                               */
    pLoudness_t1 = (double*)calloc(NumSamples, sizeof(double));
    if (pLoudness_t1 == NULL)
    {
        return LoudnessErrorMemoryAllocFailed;
    }
    pLoudness_t2 = (double*)calloc(NumSamples, sizeof(double));
    if (pLoudness_t2 == NULL)
    {
        if (pLoudness_t1 != NULL)
            free (pLoudness_t1);

        return LoudnessErrorMemoryAllocFailed;
    }

    pLoudness_1 = pLoudness_t1;
    pLoudness_2 = pLoudness_t2;

    Tau = 3.5e-3;
    f_lowpass_intp(pLoudness,pLoudness_1,Tau,SampleRate,NumSamples);

    Tau = 70e-3;
    f_lowpass_intp(pLoudness,pLoudness_2,Tau,SampleRate,NumSamples);

    for (IdxTime = 0; IdxTime < NumSamples; IdxTime++)
    {
        *pLoudness = 0.47 * (*pLoudness_1) + 0.53 * (*pLoudness_2);
        pLoudness++;
        pLoudness_1++;
        pLoudness_2++;
    }

    /*  Memory deallocation                                             */
    if (pLoudness_t1 != NULL)
        free(pLoudness_t1);

    if (pLoudness_t2 != NULL)
        free(pLoudness_t2);
    return 0;
}

//////////////////////////////////////////////////////////////////////////
//  BLOCK   Loudness calculation
//////////////////////////////////////////////////////////////////////////

/*  Third octave band level range for correction of low frequencies
    according to the equal loudness contours (RAP, LT + ∆L)             */
static double RAP[] = {45.f, 55.f, 65.f,71.f,80.f,90.f,100.f,120.f};

/*  Third octave band level reduction at low frequencies according
    to the equal loudness contours within the eight sectors
    defined by RAP(DLL)                                                 */
static double DLL[N_RAP_RANGES][N_LCB_BANDS] =
{   {-32.f,-24.f,-16.f,-10.f,-5.f,0.f,-7.f,-3.f,0.f,-2.f,0.f},
    {-29.f,-22.f,-15.f,-10.f,-4.f,0.f,-7.f,-2.f,0.f,-2.f,0.f},
    {-27.f,-19.f,-14.f, -9.f,-4.f,0.f,-6.f,-2.f,0.f,-2.f,0.f},
    {-25.f,-17.f,-12.f, -9.f,-3.f,0.f,-5.f,-2.f,0.f,-2.f,0.f},
    {-23.f,-16.f,-11.f, -7.f,-3.f,0.f,-4.f,-1.f,0.f,-1.f,0.f},
    {-20.f,-14.f,-10.f, -6.f,-3.f,0.f,-4.f,-1.f,0.f,-1.f,0.f},
    {-18.f,-12.f, -9.f, -6.f,-2.f,0.f,-3.f,-1.f,0.f,-1.f,0.f},
    {-15.f,-10.f, -8.f, -4.f,-2.f,0.f,-3.f,-1.f,0.f,-1.f,0.f}
};

/*  Critical band level at the threshold in quiet without consideration
    of the ear's transmission characteristics (LTQ)                     */
static double LTQ[] =
{   30.f,18.f,12.f,8.f,7.f,6.f,5.f,4.f,3.f,
    3.f,3.f,3.f,3.f,3.f,3.f,3.f,3.f,3.f,3.f,3.f };

/*  Level correction in accordance with the ear's transmission
    characteristics (AO)                                                */
static double A0[] =
{   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
    -0.5f,-1.6f,-3.2f,-5.4f,-5.6f,-4.0f,-1.5f,2.0f,5.0f,12.0f
};

/*  Level difference between free and diffuse sound
    field (DDF)                                                         */
static double DDF[] = 
{   0.0f, 0.0f, 0.5f, 0.9f, 1.2f, 1.6f, 2.3f, 2.8f,
    3.0f, 2.0f, 0.0f,-1.4f,-2.0f,-1.9f,-1.0f, 0.5f,
    3.0f, 4.0f, 4.3f, 4.0f
};

/*  Adaptation of the third octave band levels to the corresponding
    critical band levels due to different bandwidth (DCB)               */
static double DCB[] =
{   -.25f,-0.6f,-0.8f,-0.8f,-0.5f,0.0f,0.5f,1.1f,1.5f,1.7f,
    1.8f,1.8f,1.7f,1.6f,1.4f,1.2f,0.8f,0.5f,0.0f,-0.5f
};

/*  Upper limits of the approximated critical bands
    in bark-scale (ZUP)                                                 */
static double ZUP[] =
{   0.9f,1.8f,2.8f,3.5f,4.4f,5.4f,6.6f,7.9f,9.2f,10.6f,12.3f,
    13.8f,15.2f,16.7f,18.1f,19.3f,20.6f,21.8f,22.7f,23.6f,24.0f
};

/*  Value-range of specific loudness, which defines the
    steepness of upper slopes in the specific
    loudness - critical-band-rate pattern (RNS)                         */
static double RNS[] =
{   21.5f,18.0f,15.1f,11.5f,9.0f,6.1f,4.4f,3.1f,2.13f,1.36f,
    0.82f,0.42f,0.30f,0.22f,0.15f,0.10f,0.035f,0.0f 
};

/*  Steepness of upper slopes in the specific loudness
    - critical-band-rate pattern for the value range
    RNS as function of the number of the critical band (USL)            */
static double USL[N_RNS_RANGES][N_CB_RANGES] =
{   { 13.00f,8.20f,6.30f,5.50f,5.50f,5.50f,5.50f,5.50f},
    { 9.00f,7.50f,6.00f,5.10f,4.50f,4.50f,4.50f,4.50f},
    { 7.80f,6.70f,5.60f,4.90f,4.40f,3.90f,3.90f,3.90f},
    { 6.20f,5.40f,4.60f,4.00f,3.50f,3.20f,3.20f,3.20f},
    { 4.50f,3.80f,3.60f,3.20f,2.90f,2.70f,2.70f,2.70f},
    { 3.70f,3.00f,2.80f,2.35f,2.20f,2.20f,2.20f,2.20f},
    { 2.90f,2.30f,2.10f,1.90f,1.80f,1.70f,1.70f,1.70f},
    { 2.40f,1.70f,1.50f,1.35f,1.30f,1.30f,1.30f,1.30f},
    { 1.95f,1.45f,1.30f,1.15f,1.10f,1.10f,1.10f,1.10f},
    { 1.50f,1.20f,0.94f,0.86f,0.82f,0.82f,0.82f,0.82f},
    { 0.72f,0.67f,0.64f,0.63f,0.62f,0.62f,0.62f,0.62f},
    { 0.59f,0.53f,0.51f,0.50f,0.42f,0.42f,0.42f,0.42f},
    { 0.40f,0.33f,0.26f,0.24f,0.22f,0.22f,0.22f,0.22f},
    { 0.27f,0.21f,0.20f,0.18f,0.17f,0.17f,0.17f,0.17f},
    { 0.16f,0.15f,0.14f,0.12f,0.11f,0.11f,0.11f,0.11f},
    { 0.12f,0.11f,0.10f,0.08f,0.08f,0.08f,0.08f,0.08f},
    { 0.09f,0.08f,0.07f,0.06f,0.06f,0.06f,0.06f,0.05f},
    { 0.06f,0.05f,0.03f,0.02f,0.02f,0.02f,0.02f,0.02f} 
};

/*  Correction of third octave band levels according to the equal
    loudness contours and calculation of the intensities for the
    third octave bands up to 320 Hz                                     */
void f_corr_third_octave_intensities(   double **ThirdOctaveLevel,
                                        double **ThirdOctaveIntens,
                                        int    IdxTime)
{
    short   IdxIntens, IdxLevelRange;
    double  CorrLevel;
    double  *pLevel, *pIntens;

    for (IdxIntens = 0; IdxIntens < N_LCB_BANDS; IdxIntens++)
    {
        pIntens = ThirdOctaveIntens[IdxIntens]+IdxTime;
        pLevel  = ThirdOctaveLevel[IdxIntens]+IdxTime;

        IdxLevelRange = 0;

        while ((*pLevel >   RAP[IdxLevelRange] - DLL[IdxLevelRange][IdxIntens]) && 
                            (IdxLevelRange < N_RAP_RANGES-1))
            IdxLevelRange++;

        CorrLevel = *pLevel + DLL[IdxLevelRange][IdxIntens];
        *pIntens = pow(10., CorrLevel / 10.);
    }
}

/*  Determination of the levels LCB(1), LCB(2) und LCB(3)
    within the first three critical bands                               */
void f_calc_lcbs(double **ThirdOctaveIntens, double **Lcb, int IdxTime)
{
    short   IdxIntens, Idx3CB;
    double  *pcbi, *pIntens;

    pcbi    = Lcb[0]+IdxTime;
    pIntens = ThirdOctaveIntens[0]+IdxTime;
    *pcbi   = *pIntens;
    for (IdxIntens = 1; IdxIntens < 6; IdxIntens++)
    {
        pIntens = ThirdOctaveIntens[IdxIntens]+IdxTime;
        *pcbi   += *pIntens;
    }
    pcbi    = Lcb[1]+IdxTime;
    pIntens = ThirdOctaveIntens[6]+IdxTime;
    *pcbi   = *pIntens;
    for (IdxIntens = 7; IdxIntens < 9; IdxIntens++)
    {
        pIntens = ThirdOctaveIntens[IdxIntens]+IdxTime;
        *pcbi   += *pIntens;
    }
    pcbi    = Lcb[2]+IdxTime;
    pIntens = ThirdOctaveIntens[9]+IdxTime;
    *pcbi   = *pIntens;
    for (IdxIntens = 10; IdxIntens < N_LCB_BANDS; IdxIntens++)
    {
        pIntens = ThirdOctaveIntens[IdxIntens]+IdxTime;
        *pcbi   += *pIntens;
    }
    for (Idx3CB = 0; Idx3CB < N_LCBS; Idx3CB++)
    {
        pcbi    = Lcb[Idx3CB]+IdxTime;
        if (*pcbi >0.) *pcbi=(double)(10. * log10(*pcbi));
    }
}

/*  Calculation of loudness related to critical band level              */
void f_calc_core_loudness(  double **ThirdOctaveLevel, double **Lcb,
                            double **CoreLoudness, int SoundField, int IdxTime)
{
    short   IdxCL;
    double  S;
    double  MP1, MP2;
    double  *pLe, *pLtq, *pCoreL;

    pLtq = LTQ;

    for (IdxCL = 0; IdxCL < N_CORE_LOUDN-1; IdxCL++)
    {
        if (IdxCL < N_LCBS)
            pLe = Lcb[IdxCL]+IdxTime;
        else
            pLe = ThirdOctaveLevel[IdxCL+8]+IdxTime;

        pCoreL = CoreLoudness[IdxCL]+IdxTime;

        *pLe -= A0[IdxCL];
        *pCoreL = 0.;
        if (SoundField == SoundFieldDiffuse)
            *pLe += DDF[IdxCL];
        if (*pLe > *pLtq)
        {
            *pLe += -DCB[IdxCL];
            S = .25;
            MP1 = .0635f * (double)pow(10., 0.025 * (*pLtq));
            MP2 = (double)pow((1. - S + S * pow(10., 0.1 * (*pLe - *pLtq))), .25) - 1.0f;
            *pCoreL = MP1 * MP2;
            if (*pCoreL <= 0.) 
                *pCoreL = 0.;
        }
        pLtq++;
    }

    /*  Set last critical band to zero                                  */
    pCoreL  = CoreLoudness[IdxCL]+IdxTime;
    *pCoreL = 0.;
}

/*  Correction of the specific loudness within the lowest critical band
    for the consideration of the run of threshold in quiet within this 
    critical band                                                       */
void f_corr_loudness(double **CoreLoudness, int IdxTime)
{
    double  CorrCL, *pCoreLoudness;
    
    pCoreLoudness = CoreLoudness[0]+IdxTime;
    CorrCL = 0.4f + 0.32f * (double)pow(*pCoreLoudness, .2);
    if (CorrCL < 1.) *pCoreLoudness *= CorrCL;
}

/*  Calculation of specific loudness pattern and integration of overall 
    loudness by attaching slopes towards higher frequencies             */
void f_calc_slopes( double **CoreLoudness, double *Loudness,
                    double *SpecLoudness[N_BARK_BANDS], int IdxTime)
{
    short   IdxCL, IdxNS, IdxCBN, IdxRNS;
    int     NextCriticalBand;
    double  N1, N2, Z, Z1, Z2, ZK, DZ;
    double  _USL, _ZUP, CoreL;
    double  *pLoudness = Loudness+IdxTime, *pCoreL; 
    double  NS[N_BARK_BANDS];

    N1          = 0.;
    Z           = 0.1f;
    Z1          = 0.;
    IdxRNS      = 0;
    IdxNS       = 0;
    *pLoudness  = 0;

    for (IdxCL = 0; IdxCL < N_CORE_LOUDN; IdxCL++)  /*  Do for all core
                                                        loudness values */
    {
        pCoreL  = CoreLoudness[IdxCL]+IdxTime;
        CoreL   = *pCoreL;
        _ZUP    = ZUP[IdxCL];
        _ZUP    += .0001f;
        IdxCBN  = IdxCL - 1;
        if (IdxCBN > N_CB_RANGES-1) 
            IdxCBN  = N_CB_RANGES-1;
        NextCriticalBand    = 0;
        do 
        {   if (N1 > CoreL) /*  Slope loudness > core loudness?         */
            {   _USL    = USL[IdxRNS][IdxCBN];
                /*  Contribution of the value N2 of the specific 
                    loudness at the cut-off frequency of the 
                    corresponding critical band                         */
                N2      = RNS[IdxRNS];
                if (N2 < CoreL)
                    N2  = CoreL;
                DZ  = (N1 - N2) / _USL;
                Z2  = Z1 + DZ;
                if (Z2 > _ZUP)
                {
                    NextCriticalBand=1;
                    Z2  = _ZUP;
                    DZ  = Z2 - Z1;
                    N2  = N1 - DZ * _USL;
                }
                /*  Contribution of the loudness of slope excitation 
                    to the overall loudness and calculation of the 
                    associated interpolated values NS(IdxNS) in the 
                    interval Z=IdxNS*O.1 BARK                           */
                *pLoudness += DZ * (N1 + N2) / 2.;
                for (ZK = Z; ZK <= Z2; ZK = ZK + 0.1f)
                {
                    NS[IdxNS] = N1 - (ZK - Z1) * _USL;
                    SpecLoudness[IdxNS][IdxTime] = NS[IdxNS];
                    IdxNS++;
                }
                Z   = ZK;
            }
            else    /*  Slope loudness <= core loudness                 */
            {   /*  Determination of the number J of the sector of 
                    specific loudness                                   */
                if (N1 < CoreL)
                {
                    IdxRNS  = 0;
                    while ((IdxRNS < N_RNS_RANGES) && RNS[IdxRNS] >=CoreL)
                        IdxRNS++;
                }
                /*  Contribution of the non-masked loudness related to 
                    critical band level to the total loudness and 
                    calculation of the interpolated values NS(IdxCL) 
                    in the interval Z=IdxNS*O.1 BARK                    */
                NextCriticalBand = 1;
                Z2  = _ZUP;
                N2  = CoreL;
                *pLoudness += N2 * (Z2 - Z1);
                ZK  = Z;
                while (ZK <= Z2)
                {
                    NS[IdxNS] = N2;
                    SpecLoudness[IdxNS][IdxTime] = NS[IdxNS];
                    IdxNS++;
                    ZK  = ZK + 0.1f;
                }
                Z   = ZK;
            }
        /*  Step to next segment    */
        while  ((N2 <= RNS[IdxRNS]) && (IdxRNS < N_RNS_RANGES-1))
            IdxRNS++;
        if (IdxRNS > N_RNS_RANGES-1) 
            IdxRNS  = N_RNS_RANGES-1;
        Z1  = Z2;
        N1  = N2;
        } while (!NextCriticalBand);

        if (*pLoudness < 0.) 
            *pLoudness  = 0;
    }
}

int f_loudness_from_levels(
    double  **ThirdOctaveLevel,
    int     NumSamplesLevel,
    int     SoundField,
    int     Method,
    double  *OutLoudness,
    double  *OutSpecLoudness[N_BARK_BANDS]
    )
{
    double  *CoreLoudness[N_CORE_LOUDN];
    double  *ThirdOctaveIntens[N_LCB_BANDS], *Lcb[N_LCBS];

    int     IdxTime;
    int     SampleRateLevel;
    int     retval;

    SampleRateLevel = SR_LEVEL;

    /*  Memory allocation                                               */
    retval = callocRaggedArray(CoreLoudness, N_CORE_LOUDN, NumSamplesLevel);
    if (retval < 0)
    {
        return retval;
    }

    retval = callocRaggedArray(Lcb, N_LCBS, NumSamplesLevel);
    if (retval < 0)
    {
        freeRaggedArray(CoreLoudness, N_CORE_LOUDN);
        return retval;
    }

    retval = callocRaggedArray(ThirdOctaveIntens, N_LCB_BANDS, NumSamplesLevel);
    if (retval < 0)
    {
        freeRaggedArray(CoreLoudness, N_CORE_LOUDN);
        freeRaggedArray(Lcb, N_LCBS);
        return retval;
    }

    /*  Calculate core loudness                                         */
    for (IdxTime = 0; IdxTime < NumSamplesLevel; IdxTime++)
    {
        f_corr_third_octave_intensities(ThirdOctaveLevel,
            ThirdOctaveIntens, IdxTime);
        f_calc_lcbs(ThirdOctaveIntens, Lcb, IdxTime);
        f_calc_core_loudness(ThirdOctaveLevel, Lcb, CoreLoudness,
            SoundField, IdxTime);
    }

    /*  Memory deallocation                                             */
    freeRaggedArray(Lcb, N_LCBS);
    freeRaggedArray(ThirdOctaveIntens, N_LCB_BANDS);

    /*  Correction of specific loudness within lowest critical band     */
    for (IdxTime = 0; IdxTime < NumSamplesLevel; IdxTime++)
        f_corr_loudness(CoreLoudness, IdxTime);

    /*  Time-varying loudness: nonlinearity                             */
    if (Method == LoudnessMethodTimeVarying)
        f_nl(CoreLoudness, SampleRateLevel, NumSamplesLevel);

    /*  Calculation of specific loudness                                */
    for (IdxTime = 0; IdxTime < NumSamplesLevel; IdxTime++)
    {
        f_calc_slopes(CoreLoudness, OutLoudness, OutSpecLoudness,
            IdxTime);
    }

    /*  Memory deallocation                                             */
    freeRaggedArray(CoreLoudness, N_CORE_LOUDN);

    /*  Time-varying loudness: temporal weighting                       */
    if (Method == LoudnessMethodTimeVarying)
    {
        retval = f_temporal_weight_loudness(OutLoudness, SampleRateLevel,
                    NumSamplesLevel);
        if (retval < 0)
        {
            return retval;
        }
    }

    return NumSamplesLevel;
}

int f_loudness_from_signal(
    struct  InputData *pSignal,
    int     SoundField,
    int     Method,
    double  TimeSkip,
    double  *OutLoudness,
    double  *OutSpecLoudness[N_BARK_BANDS],
    int     SizeOutput
    )
{
    double  *ThirdOctaveLevel[N_LEVEL_BANDS];
    int     SampleRateLevel = 1;
    int     DecFactorLevel = 1;
    int     NumSamplesTime = 1;
    int     NumSamplesLevel = 1;

    int retval;

    if (Method == LoudnessMethodStationary)
    {
        DecFactorLevel = (int)(pSignal->NumSamples);
    }
    else if (Method == LoudnessMethodTimeVarying)
    {
        SampleRateLevel = SR_LEVEL;

        DecFactorLevel = (int)(pSignal->SampleRate / SampleRateLevel);

        NumSamplesTime = pSignal->NumSamples;
        NumSamplesLevel = NumSamplesTime / DecFactorLevel;
    }
    else
    {
        return LoudnessErrorUnsupportedMethod;
    }

    /*  Check size of output vectors                                    */
    if (SizeOutput < NumSamplesLevel)
    {
        return LoudnessErrorOutputVectorTooSmall;
    }

    /*  Memory allocation                                               */
    retval = callocRaggedArray(ThirdOctaveLevel, N_LEVEL_BANDS, NumSamplesLevel);
    if (retval < 0)
    {
        return retval;
    }

    /*  Calculate third octave levels                                   */
    retval = f_calc_third_octave_levels(pSignal, ThirdOctaveLevel,
                DecFactorLevel, Method, TimeSkip);
    if (retval < 0)
    {
        freeRaggedArray(ThirdOctaveLevel, N_LEVEL_BANDS);
        return retval;
    }

    /*  Loudness calculation                                            */
    retval = f_loudness_from_levels(ThirdOctaveLevel, NumSamplesLevel,
                SoundField, Method, OutLoudness,
                OutSpecLoudness);

    /*  Memory deallocation                                             */
    freeRaggedArray(ThirdOctaveLevel, N_LEVEL_BANDS);

    return retval;
}

