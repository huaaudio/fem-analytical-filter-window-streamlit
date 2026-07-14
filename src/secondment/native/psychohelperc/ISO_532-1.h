#ifndef HEADER_ISO_532_1
#define HEADER_ISO_532_1

/************************************************************************/
/*  Loudness calculation according to ISO 532-1,                        */
/*  methods for stationary and time varying signals                     */
/************************************************************************/

#ifdef __cplusplus 
extern "C"
{
#endif

#if defined(_WIN32) && !defined(ISO532_STATIC)
#  if defined(ISO532_BUILD)
#    define ISO532_API __declspec(dllexport)
#  else
#    define ISO532_API
#  endif
#else
#  define ISO532_API
#endif

/*  Loudness calculation methods                                        */
enum _LoudnessMethod
{
    LoudnessMethodStationary  = 0,
    LoudnessMethodTimeVarying = 1,
};

/* Sound field types                                                    */
enum _SoundField
{
    SoundFieldFree    = 0,
    SoundFieldDiffuse = 1
};

/* Error codes                                                          */
enum _LoudnessErrorCodes
{
    LoudnessErrorOutputVectorTooSmall = -1,
    LoudnessErrorMemoryAllocFailed    = -2,
    LoudnessErrorUnsupportedMethod    = -3,
    LoudnessErrorSignalTooShort       = -4
};

/*  Inputfile data structure                                            */
struct InputData
{
    int     NumSamples;
    double  SampleRate;
    double  *pData;
};

/*  Reference value for intensity calculation                           */
#define I_REF           4e-10

/*  Sampling rate to which third-octave-levels are downsampled          */
#define SR_LEVEL        2000

/*  Number of third octave frequency bands                              */
#define N_LEVEL_BANDS   28

/*  Number of bark bands (resolution 0.1 Bark)                          */
#define N_BARK_BANDS    240

/*  Loudness calculation from provided third octave levels
    Input parameters:
    ThirdOctaveLevel:   Array of 28 double arrays with one array per
                        frequency band. All 28 double arrays must
                        have the same length which is the number of
                        time samples.
    NumSamplesLevel:    Number of time samples in ThirdOctaveLevel
    SoundField:         One of SoundFieldDiffuse or SoundFieldFree
    Method:             One of LoudnessMethodStationary or
                        LoudnessMethodTimeVarying
    OutLoudness:        Pointer to a double array of length
                        NumSamplesLevel to which the loudness result
                        is written.
    OutSpecLoudness:    Pointer to a double array of length
                        NumSamplesLevel to which the specific loudness
                        is written.
    Output value:       Number of values actually written into
                        output parameters or negative error code.       */
ISO532_API int f_loudness_from_levels(
    double  **ThirdOctaveLevel,
    int     NumSamplesLevel,
    int     SoundField,
    int     Method,
    double  *OutLoudness,
    double  *OutSpecLoudness[N_BARK_BANDS]
    );

/*  Loudness calculation from provided time signal
    Input parameters:
    pSignal:            Data structure with input signal at 48 kHz.
    SoundField:         One of SoundFieldDiffuse or SoundFieldFree
    Method:             One of LoudnessMethodStationary or
                        LoudnessMethodTimeVarying
    TimeSkip:           Value specifying number of seconds to
                        skip for level calculation. (only relevant
                        for Method = LoudnessMethodStationary)
    OutLoudness:        Pointer to a double array of length
                        NumSamplesLevel to which the loudness result
                        is written.
    OutSpecLoudness:    Pointer to a double array of length
                        NumSamplesLevel to which the specific loudness
                        is written.
    SizeOutput:         Size of output variables OutLoudness and 
                        OutSpecLoudness
    Output value:       Number of values actually written into
                        output parameters or negative error code.       */
ISO532_API int f_loudness_from_signal(
    struct  InputData *pSignal, 
    int     SoundField,
    int     Method, 
    double  TimeSkip,
    double  *OutLoudness,
    double  *OutSpecLoudness[N_BARK_BANDS],
    int     SizeOutput
    );

#ifdef __cplusplus 
}
#endif

#endif /* HEADER_ISO_532_1 */
