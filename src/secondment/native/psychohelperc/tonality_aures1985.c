/************************************************************************/
/*  Tonality calculation according to Aures (1985)                      */
/*                                                                      */
/*  Ported from archive/PA/Tonality_Aures1985/Tonality_Aures1985.m.     */
/************************************************************************/

#define TONALITY_AURES1985_BUILD
#include "tonality_aures1985.h"

#include "ISO_532-1.h"
#include "pocketfft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define AURES_TIME_RESOLUTION 0.160
#define AURES_MIN_FREQUENCY   20.0
#define AURES_MAX_FREQUENCY   5000.0
#define AURES_TONE_THRESHOLD  7.0
#define AURES_TINY_VALUE      1e-99
#define AURES_I_REF           4e-10
#define AURES_PREF            2e-5

static int round_to_int(double x)
{
    return (int)floor(x + 0.5);
}

static double safe_log10(double x)
{
    return (x <= 0.0) ? -400.0 : log10(x);
}

static double complex_abs_at(const double *fftData, int idx)
{
    double re = fftData[2 * idx];
    double im = fftData[2 * idx + 1];
    return sqrt(re * re + im * im);
}

static void hann_window(double *window, int n)
{
    if (n <= 1) {
        if (n == 1) window[0] = 1.0;
        return;
    }

    for (int i = 0; i < n; i++) {
        window[i] = 0.5 - 0.5 * cos(2.0 * M_PI * (double)i / (double)(n - 1));
    }
}

static double mean_array(const double *x, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i];
    return (n > 0) ? s / (double)n : 0.0;
}

static double fq_to_bark(double f)
{
    double fk = f / 1000.0;
    return 13.0 * atan(0.76 * fk) + 3.5 * atan((fk / 7.5) * (fk / 7.5));
}

static double threshold_hearing(double f)
{
    double fk = f / 1000.0;
    if (fk <= 0.0) return 130.0;
    return 3.64 * pow(fk, -0.8)
         - 6.5 * exp(-0.6 * (fk - 3.3) * (fk - 3.3))
         + 1e-3 * fk * fk * fk * fk;
}

static double interp_xy(double x0, double y0, double x1, double y1, double x)
{
    if (fabs(x1 - x0) < 1e-30) return 0.5 * (y0 + y1);
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

static void do_fft(double *data, int n, cfft_plan plan)
{
    (void)n;
    cfft_forward(plan, data, 1.0);
}

static void do_ifft(double *data, int n, cfft_plan plan)
{
    cfft_backward(plan, data, 1.0 / (double)n);
}

static double *resample_linear(const double *in, int inLen, double inRate,
                               double outRate, int *outLen)
{
    if (inRate <= 0.0 || outRate <= 0.0 || inLen <= 0) return NULL;

    double scale = outRate / inRate;
    int n = (int)floor((double)inLen * scale + 0.5);
    if (n < 1) n = 1;

    double *out = (double*)calloc((size_t)n, sizeof(double));
    if (!out) return NULL;

    for (int i = 0; i < n; i++) {
        double s = (double)i / scale;
        int idx = (int)floor(s);
        double frac = s - (double)idx;

        if (idx + 1 < inLen) {
            out[i] = in[idx] * (1.0 - frac) + in[idx + 1] * frac;
        } else if (idx < inLen) {
            out[i] = in[idx];
        }
    }

    *outLen = n;
    return out;
}

static int compute_stationary_loudness(const double *signal, int n, double fs,
                                       int soundField, double timeSkip,
                                       double *outLoudness)
{
    struct InputData in;
    double loudness[1] = {0.0};
    double specStore[N_BARK_BANDS];
    double *specPtrs[N_BARK_BANDS];

    for (int i = 0; i < N_BARK_BANDS; i++) {
        specStore[i] = 0.0;
        specPtrs[i] = &specStore[i];
    }

    in.NumSamples = n;
    in.SampleRate = fs;
    in.pData = (double*)signal;

    int ret = f_loudness_from_signal(&in, soundField, LoudnessMethodStationary,
                                     timeSkip, loudness, specPtrs, 1);
    if (ret < 0) return ret;

    *outLoudness = loudness[0];
    return ret;
}

static int compare_double_ascending(const void *a, const void *b)
{
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static double percentile_sorted(const double *sorted, int n, double q)
{
    if (n <= 0) return 0.0;
    if (n == 1) return sorted[0];
    if (q <= 0.0) return sorted[0];
    if (q >= 1.0) return sorted[n - 1];

    double pos = q * (double)(n - 1);
    int lo = (int)floor(pos);
    int hi = lo + 1;
    double frac = pos - (double)lo;
    if (hi >= n) return sorted[n - 1];
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static int fill_stats(const double *values, const double *time, int n,
                      double timeSkip, double *outStats)
{
    int start = 0;
    double best = fabs(time[0] - timeSkip);

    for (int i = 1; i < n; i++) {
        double d = fabs(time[i] - timeSkip);
        if (d < best) {
            best = d;
            start = i;
        }
    }

    int m = n - start;
    if (m <= 0) return TonalityAures1985ErrorSignalTooShort;

    double mean = 0.0;
    double maxv = values[start];
    double minv = values[start];
    for (int i = start; i < n; i++) {
        double v = values[i];
        mean += v;
        if (v > maxv) maxv = v;
        if (v < minv) minv = v;
    }
    mean /= (double)m;

    double ss = 0.0;
    for (int i = start; i < n; i++) {
        double d = values[i] - mean;
        ss += d * d;
    }
    double stdv = (m > 1) ? sqrt(ss / (double)(m - 1)) : 0.0;

    double *tmp = (double*)calloc((size_t)m, sizeof(double));
    if (!tmp) return TonalityAures1985ErrorMemoryAlloc;
    memcpy(tmp, values + start, (size_t)m * sizeof(double));
    qsort(tmp, (size_t)m, sizeof(double), compare_double_ascending);
    double p5 = percentile_sorted(tmp, m, 0.95);
    free(tmp);

    outStats[0] = mean;
    outStats[1] = stdv;
    outStats[2] = maxv;
    outStats[3] = minv;
    outStats[4] = p5;

    return 0;
}

static void estimate_bandwidths(const double *spl, int n, double fs,
                                const int *toneBins, const double *toneLevels,
                                int nTones, double *outBw)
{
    for (int i = 0; i < nTones; i++) {
        int idx = toneBins[i];
        double target = toneLevels[i] * 0.707;
        int lowIdx = -1;
        int highIdx = -1;

        for (int j = 0; j <= idx; j++) {
            if (spl[j] < target) lowIdx = j;
        }
        if (lowIdx < 3) lowIdx = 3;
        if (lowIdx + 1 >= n) lowIdx = n - 2;

        for (int j = idx + 1; j < n; j++) {
            if (spl[j] < target) {
                highIdx = j;
                break;
            }
        }
        if (highIdx < 1) highIdx = idx + 1;
        if (highIdx >= n) highIdx = n - 1;

        double fLow = interp_xy(spl[lowIdx], (double)lowIdx * fs / (double)n,
                                spl[lowIdx + 1], (double)(lowIdx + 1) * fs / (double)n,
                                target);
        double fHigh = interp_xy(spl[highIdx - 1], (double)(highIdx - 1) * fs / (double)n,
                                 spl[highIdx], (double)highIdx * fs / (double)n,
                                 target);
        double bw = fHigh - fLow;
        if (!isfinite(bw) || bw == 0.0) bw = 1.0;
        if (bw < 0.0) bw = -bw;
        outBw[i] = bw;
    }
}

static void remove_tones_preserve_phase(double *spectrum, int n, double fs,
                                        const double *toneFreq,
                                        const double *toneBw,
                                        int nTones)
{
    int singleLen = n / 2 + 1;

    for (int i = 0; i < nTones; i++) {
        double fLow = toneFreq[i] - 0.5 * toneBw[i];
        double fHigh = toneFreq[i] + 0.5 * toneBw[i];
        int low = 0;
        int up = singleLen - 1;

        for (int k = 0; k < singleLen; k++) {
            if ((double)k * fs / (double)n >= fLow) {
                low = k;
                break;
            }
        }
        for (int k = 0; k < singleLen; k++) {
            if ((double)k * fs / (double)n >= fHigh) {
                up = k;
                break;
            }
        }

        if (low < 0) low = 0;
        if (up < low) up = low;
        if (up >= singleLen) up = singleLen - 1;

        int left = (low == 0) ? low : low - 1;
        int right = (up + 1 < singleLen) ? up + 1 : up;
        double mag = 0.5 * (complex_abs_at(spectrum, left) +
                            complex_abs_at(spectrum, right));

        for (int k = low; k <= up; k++) {
            double phase = atan2(spectrum[2 * k + 1], spectrum[2 * k]);
            spectrum[2 * k] = mag * cos(phase);
            spectrum[2 * k + 1] = mag * sin(phase);
        }
    }

    for (int k = 1; k < singleLen - 1; k++) {
        int dst = n - k;
        spectrum[2 * dst] = spectrum[2 * k];
        spectrum[2 * dst + 1] = -spectrum[2 * k + 1];
    }
    spectrum[1] = 0.0;
    if ((n % 2) == 0) spectrum[2 * (n / 2) + 1] = 0.0;
}

static void spl_excess(const double *splCrop, int cropLen, int minIdx,
                       double fs, int nFft, const double *toneFreq,
                       const double *toneLevel, const int *toneCropIdx,
                       int nTones, double *outLx)
{
    for (int i = 0; i < nTones; i++) {
        double toneBark = fq_to_bark(toneFreq[i]);
        double lowBark = round(toneBark - 0.5);
        double highBark = round(toneBark + 0.5);
        double egr = 0.0;

        for (int k = 0; k < cropLen; k++) {
            double freq = (double)(minIdx + k) * fs / (double)nFft;
            double bark = fq_to_bark(freq);
            int skipTone = (k >= toneCropIdx[i] - 2 && k <= toneCropIdx[i] + 2);
            if (!skipTone && bark >= lowBark && bark <= highBark) {
                egr += AURES_PREF * pow(10.0, splCrop[k] / 10.0);
            }
        }

        double sumlo = AURES_TINY_VALUE;
        double sumhi = AURES_TINY_VALUE;
        for (int j = 0; j < nTones; j++) {
            if (j == i) continue;

            double barkJ = fq_to_bark(toneFreq[j]);
            double lji;
            if (j < i) {
                double s = -24.0 - (230.0 / toneFreq[j]) + (0.2 * toneLevel[j]);
                lji = toneLevel[j] - s * (barkJ - toneBark);
                sumlo += pow(10.0, lji / 20.0);
            } else {
                lji = toneLevel[j] - 27.0 * (barkJ - toneBark);
                sumhi += pow(10.0, lji / 20.0);
            }
        }

        double aek = sumlo + sumhi;
        double ehs = pow(10.0, threshold_hearing(toneFreq[i]) / 10.0);
        double den = (nTones == 1) ? (egr + ehs) : (aek * aek + egr + ehs);
        double lxi = toneLevel[i] - 10.0 * safe_log10(den);
        outLx[i] = (lxi > 0.0 && isfinite(lxi)) ? lxi : 0.0;
    }
}

static double tonal_weighting(const double *toneFreq, const double *toneBw,
                              const double *toneLx, int nTones, double df)
{
    double sumSq = 0.0;

    for (int i = 0; i < nTones; i++) {
        double fc = toneFreq[i];
        double bw = toneBw[i];
        double deltaL = toneLx[i];
        if (fc <= 0.0 || bw <= 0.0 || deltaL <= 0.0) continue;

        double zup = fq_to_bark(fc + 0.5 * bw);
        double zlow = fq_to_bark(fc - 0.5 * bw);
        double dz = (zup - zlow) / (df * df);
        double w1 = 0.13 / (dz + 0.13);
        double x = fc / 700.0 + 700.0 / fc;
        double w2 = pow(1.0 / sqrt(1.0 + 0.2 * x * x), 0.29);
        double w3 = pow(1.0 - exp(-deltaL / 15.0), 0.29);

        if (w1 < 0.0 || !isfinite(w1)) w1 = 0.0;
        if (w2 < 0.0 || !isfinite(w2)) w2 = 0.0;
        if (w3 < 0.0 || !isfinite(w3)) w3 = 0.0;

        double ww1 = pow(w1, 1.0 / 0.29);
        double ww2 = pow(w2, 1.0 / 0.29);
        double ww3 = pow(w3, 1.0 / 0.29);
        double prod = ww1 * ww2 * ww3;
        sumSq += prod * prod;
    }

    return sqrt(sumSq);
}

int tonality_aures1985(
    const double *signal,
    int           numSamples,
    double        sampleRate,
    int           soundField,
    double        timeSkip,
    double       *outTonality,
    double       *outTonalWeighting,
    double       *outLoudnessWeighting,
    double       *outTime,
    int          *pNumFrames,
    double       *outStats)
{
    const double *audio = signal;
    int audioLen = numSamples;
    double fs = sampleRate;
    double *resampled = NULL;

    if (!signal || !outTonality || !pNumFrames || numSamples <= 0) {
        return TonalityAures1985ErrorInvalidArgument;
    }
    if (sampleRate <= 0.0 || !isfinite(sampleRate)) {
        return TonalityAures1985ErrorInvalidSampleRate;
    }
    if (soundField != SoundFieldFree && soundField != SoundFieldDiffuse) {
        return TonalityAures1985ErrorInvalidArgument;
    }

    if (fs != 44100.0 && fs != 48000.0) {
        int newLen = 0;
        resampled = resample_linear(signal, numSamples, fs, 44100.0, &newLen);
        if (!resampled) return TonalityAures1985ErrorMemoryAlloc;
        audio = resampled;
        audioLen = newLen;
        fs = 44100.0;
    }

    int n = round_to_int(fs * AURES_TIME_RESOLUTION);
    int hop = round_to_int(0.5 * (double)n);
    int numFrames = (audioLen - n) / hop;
    if (n <= 0 || hop <= 0 || numFrames <= 0) {
        free(resampled);
        return TonalityAures1985ErrorSignalTooShort;
    }
    if (*pNumFrames < numFrames) {
        *pNumFrames = numFrames;
        free(resampled);
        return TonalityAures1985ErrorOutputTooSmall;
    }
    *pNumFrames = numFrames;

    cfft_plan fftPlan = make_cfft_plan((size_t)n);
    if (!fftPlan) {
        free(resampled);
        return TonalityAures1985ErrorFFTPlan;
    }

    double *window = (double*)calloc((size_t)n, sizeof(double));
    double *fftBuf = (double*)calloc((size_t)2 * n, sizeof(double));
    double *filtBuf = (double*)calloc((size_t)2 * n, sizeof(double));
    double *spl = (double*)calloc((size_t)n, sizeof(double));
    double *frame = (double*)calloc((size_t)n, sizeof(double));
    double *filtered = (double*)calloc((size_t)n, sizeof(double));

    int minIdx = (int)ceil(1.0 + AURES_MIN_FREQUENCY * ((double)n / fs)) - 1;
    int maxIdx = (int)ceil(1.0 + AURES_MAX_FREQUENCY * ((double)n / fs)) - 1;
    if (minIdx < 0) minIdx = 0;
    if (maxIdx >= n / 2) maxIdx = n / 2;
    int cropLen = maxIdx - minIdx + 1;

    int *toneBin = (int*)calloc((size_t)cropLen, sizeof(int));
    int *toneCropIdx = (int*)calloc((size_t)cropLen, sizeof(int));
    double *toneFreq = (double*)calloc((size_t)cropLen, sizeof(double));
    double *toneLevel = (double*)calloc((size_t)cropLen, sizeof(double));
    double *toneBw = (double*)calloc((size_t)cropLen, sizeof(double));
    double *toneLx = (double*)calloc((size_t)cropLen, sizeof(double));

    if (!window || !fftBuf || !filtBuf || !spl || !frame || !filtered ||
        !toneBin || !toneCropIdx || !toneFreq || !toneLevel || !toneBw || !toneLx ||
        cropLen < 8) {
        free(window); free(fftBuf); free(filtBuf); free(spl); free(frame); free(filtered);
        free(toneBin); free(toneCropIdx); free(toneFreq); free(toneLevel); free(toneBw); free(toneLx);
        destroy_cfft_plan(fftPlan);
        free(resampled);
        return TonalityAures1985ErrorMemoryAlloc;
    }

    hann_window(window, n);
    double winMean = mean_array(window, n);
    double fftGain = sqrt(2.0) / ((double)n * winMean);
    double df = fs / (double)n;
    double loudnessTimeSkip = AURES_TIME_RESOLUTION * 0.05;

    for (int fr = 0; fr < numFrames; fr++) {
        int start = fr * hop;
        int nTones = 0;

        for (int i = 0; i < n; i++) {
            frame[i] = audio[start + i];
            fftBuf[2 * i] = frame[i] * window[i] * fftGain;
            fftBuf[2 * i + 1] = 0.0;
        }

        do_fft(fftBuf, n, fftPlan);
        for (int i = 0; i < n; i++) {
            double re = fftBuf[2 * i];
            double im = fftBuf[2 * i + 1];
            double energy = re * re + im * im;
            spl[i] = 10.0 * safe_log10((energy + AURES_TINY_VALUE) / AURES_I_REF);
        }

        for (int ci = 3; ci <= cropLen - 4; ci++) {
            int bi = minIdx + ci;
            double p = spl[bi];
            if (p > spl[bi - 1] &&
                p >= spl[bi + 1] &&
                p - spl[bi - 3] >= AURES_TONE_THRESHOLD &&
                p - spl[bi - 2] >= AURES_TONE_THRESHOLD &&
                p - spl[bi + 2] >= AURES_TONE_THRESHOLD &&
                p - spl[bi + 3] >= AURES_TONE_THRESHOLD) {
                toneBin[nTones] = bi;
                toneCropIdx[nTones] = ci;
                toneLevel[nTones] = p;
                toneFreq[nTones] = (double)bi * fs / (double)n;
                nTones++;
            }
        }

        if (nTones > 0) {
            estimate_bandwidths(spl, n, fs, toneBin, toneLevel, nTones, toneBw);

            int keep = 0;
            for (int i = 0; i < nTones; i++) {
                if (toneLevel[i] > 0.0) {
                    toneBin[keep] = toneBin[i];
                    toneCropIdx[keep] = toneCropIdx[i];
                    toneFreq[keep] = toneFreq[i];
                    toneLevel[keep] = toneLevel[i];
                    toneBw[keep] = toneBw[i];
                    keep++;
                }
            }
            nTones = keep;
        }

        double wTonal = 0.0;
        double wLoudness = 0.0;
        double tonality = 0.0;

        if (nTones > 0) {
            for (int i = 0; i < n; i++) {
                filtBuf[2 * i] = frame[i];
                filtBuf[2 * i + 1] = 0.0;
            }
            do_fft(filtBuf, n, fftPlan);
            remove_tones_preserve_phase(filtBuf, n, fs, toneFreq, toneBw, nTones);
            do_ifft(filtBuf, n, fftPlan);
            for (int i = 0; i < n; i++) filtered[i] = filtBuf[2 * i];

            double lTotal = 0.0;
            double lFiltered = 0.0;
            int retTotal = compute_stationary_loudness(frame, n, fs, soundField,
                                                       loudnessTimeSkip, &lTotal);
            int retFiltered = compute_stationary_loudness(filtered, n, fs, soundField,
                                                          loudnessTimeSkip, &lFiltered);
            if (retTotal < 0 || retFiltered < 0) {
                free(window); free(fftBuf); free(filtBuf); free(spl); free(frame); free(filtered);
                free(toneBin); free(toneCropIdx); free(toneFreq); free(toneLevel); free(toneBw); free(toneLx);
                destroy_cfft_plan(fftPlan);
                free(resampled);
                return TonalityAures1985ErrorLoudnessFailed;
            }

            if (lTotal > AURES_TINY_VALUE) {
                wLoudness = 1.0 - (lFiltered / lTotal);
                if (wLoudness < 0.0 || !isfinite(wLoudness)) wLoudness = 0.0;
            }

            spl_excess(spl + minIdx, cropLen, minIdx, fs, n,
                       toneFreq, toneLevel, toneCropIdx, nTones, toneLx);
            wTonal = tonal_weighting(toneFreq, toneBw, toneLx, nTones, df);
            tonality = fabs(1.125 * pow(wTonal, 0.29) * pow(wLoudness, 0.79));
            if (!isfinite(tonality)) tonality = 0.0;
        }

        outTonality[fr] = tonality;
        if (outTonalWeighting) outTonalWeighting[fr] = wTonal;
        if (outLoudnessWeighting) outLoudnessWeighting[fr] = wLoudness;
        if (outTime) outTime[fr] = ((double)start + 1.0) / fs;
    }

    int statsRet = 0;
    if (outStats) {
        if (outTime) {
            statsRet = fill_stats(outTonality, outTime, numFrames, timeSkip, outStats);
        } else {
            double *tmpTime = (double*)calloc((size_t)numFrames, sizeof(double));
            if (!tmpTime) {
                statsRet = TonalityAures1985ErrorMemoryAlloc;
            } else {
                for (int fr = 0; fr < numFrames; fr++) {
                    tmpTime[fr] = ((double)(fr * hop) + 1.0) / fs;
                }
                statsRet = fill_stats(outTonality, tmpTime, numFrames, timeSkip, outStats);
                free(tmpTime);
            }
        }
    }

    free(window);
    free(fftBuf);
    free(filtBuf);
    free(spl);
    free(frame);
    free(filtered);
    free(toneBin);
    free(toneCropIdx);
    free(toneFreq);
    free(toneLevel);
    free(toneBw);
    free(toneLx);
    destroy_cfft_plan(fftPlan);
    free(resampled);

    if (statsRet < 0) return statsRet;
    return numFrames;
}


