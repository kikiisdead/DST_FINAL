#pragma once
#ifndef DEAD_FFT_H
#define DEAD_FFT_H

#include "daisy_seed.h"
#include "daisysp.h"
#include <math.h>
#include "complex.h"
#include <complex>

using C = std::complex<float>;

class DeadFFT{
private:
    /**
     * @brief Hermite Interpolation
     * 
     * @param array to interpolate from
     * @param pos within array needing access
     * @param size size of the array
     * @return float 
     */
    static float interpolate(float* array, float pos, int size) {
        
        int integral   = static_cast<int>(pos);
        float fractional = pos - integral;

        float xm1 = array[(integral - 1 + size) % size];
        float x0  = array[(integral     + size) % size];
        float x1  = array[(integral + 1 + size) % size];
        float x2  = array[(integral + 2 + size) % size];

        float c = (x1 - xm1) * 0.5;
        float v = x0 - x1;
        float w = c + v;
        float a = w + v + (x2 - x0) * 0.5;
        float b_neg = w + a;
        float f = fractional;

        return (((a * f) - b_neg) * f + c) * f + x0; // the interpolated sample
    }

    /**
     * @brief FFT recursive function calls (the thing doing the DFS)
     * 
     * @param x complex input
     * @param x_start input starting index
     * @param N FFT size
     * @param out complex output
     * @param out_start output starting index
     * @param step step size through input and output
     */
    static void FFT_recursion(C* x, int x_start, int N, C* out, int out_start, int step) {
        // size 1 DFT base case
        if (N == 1) {
            out[out_start] = x[x_start];
        } else {
            // recursive call splitting into odd and even
            FFT_recursion(x, x_start, N/2, out, out_start, step*2);
            FFT_recursion(x, x_start + step, N/2, out, out_start + N/2, step*2);
            // recombining even and odd DFTs with some phase shifting twiddle factor
            for (int k = 0; k < N/2; k++) {
                C twiddle = C(cosf(TWOPI_F * (float) k / (float) N), sinf(TWOPI_F * (float) k / (float) N));
                C even = out[k + out_start];
                C odd = out[k + out_start + N/2] * twiddle;

                out[k + out_start] = even + odd;
                out[k + out_start + N/2] = even - odd;
            }
        }
    }

    static float* autocorrelate(float* in_real, int N) {
        C in[N*2];
        for (int n = 0; n < N * 2; n++) {
            if (n < N)
                in[n] = C(in_real[n]);
            else 
                in[n] = C(0);
        }
        C fft[N*2];
        FFT_recursion(in, 0, N * 2, fft, 0, 1);
        C abs_square[N*2];

        for (int n = 0; n < N * 2; n++) {
            abs_square[n] = fft[n] * conj(fft[n]);
        }

        return IFFT(abs_square, N*2);
    }

public:
    
    /**
     * @brief Implementation of the Cooley-Turkey divide and conquer FFT
     * 
     * @param in_real incoming sequence of data
     * @param N length of data / size of FFT window
     * @return complex<double>* 
     */
    static C* FFT(float* in_real, int N) {
        C in[N];
        for (int i = 0; i < N; i ++) {
            in[i] = C(in_real[i]);
        }
        C* out = new C[N]{};
        FFT_recursion(in, 0, N, out, 0, 1);
        return out;
    }

    /**
     * @brief Inverse fourier transform
     * 
     * @param in complex input to transform
     * @param N window size
     * @return float* 
     */
    static float* IFFT(C* in, int N) {
        C out[N]{0};
        for (int n = 0; n < N; n++) {
            in[n] = conj(in[n]);
        }
        FFT_recursion(in, 0, N, out, 0, 1);

        float* out_real = new float[N]{};
        for (int n = 0; n < N; n ++) {
            out_real[n] = out[n].real() / static_cast<float>(N);
        }
        return out_real;
    }

    /**
     * @brief Cumulative Mean-Normalized Difference Function for computation of Yin pitch detector
     * 
     * @param in_real real input
     * @param N 
     * @return float* 
     */
    static float* CMNDF(float* in_real, int N) {
        float* autocorr = autocorrelate(in_real, N);

        float r0 = 0;
        for (int n = 0; n < N; n++) {
            r0 += in_real[n] * in_real[n];
        }

        float sum = 0;
        float avg;

        float* cmndf = new float[N];
        cmndf[0] = 1.;

        for (int tau = 1; tau < N; tau++) {
            float diff = 2. * r0 - 2. * autocorr[tau];
            sum += diff;
            avg = sum / static_cast<float>(tau);
            cmndf[tau] = diff / avg;
        }

        delete[] autocorr;

        return cmndf;
    }

    /**
     * @brief Finds the period of a window given the CMNDF
     * 
     * @param CMNDF float array of Cumulative Mean-Normalized Difference Function
     * @param size size of the array
     * @param thresh optional: threshold which peaks must be below. 
     * @return the period of the window or 0
     */
    static int period(float* CMNDF, int size, float thresh = 0.4) {
        int T = 0;
        float min = thresh;

        for (int tau = 0; tau < size; tau++) {
            // finding nearest lowest point below thresh
            if (CMNDF[tau] < thresh) {
                if (CMNDF[tau] < min){
                    min = CMNDF[tau];
                    T = tau;
                }
            }
        }

        return T;
    }

    /**
     * @brief 
     * 
     * @param in_real array to be phase vocoded
     * @param pitch_shift pitch shift to apply
     * @param N size of array
     * @param hop hop size between windows
     * @param phs_accu phs accumulator for correlation
     * @return float* 
     */
    static float* phase_vocode(float in_real[], float pitch_shift, int N, int hop, double phs_accu[]) {
        C* fft = FFT(in_real, N);
        C fft_shift[N];
        double del_omega[N];

        float real[N];
        float imag[N];

        for (int n = 0; n < N; n++) {
            real[n] = fft[n].real();
            imag[n] = fft[n].imag();
        }

        for (int n = 2; n < N/2; n++) {
            double nf = static_cast<double>(n);

            del_omega[n] = (((nf / pitch_shift) - nf) * TWOPI_F) / static_cast<double>(N);
            phs_accu[n] = fmod(phs_accu[n] + (del_omega[n] * static_cast<double>(hop)), TWOPI_F);

            float new_real = interpolate(real, nf / pitch_shift, (N/2));
            float new_imag = interpolate(imag, nf / pitch_shift, (N/2));

            double sine = sin(phs_accu[n]);
            double cosi = cos(phs_accu[n]);

            // applying delta omega as a rotation matrix (thank you linear algebra)
            float x = new_real * cosi - new_imag * sine;
            float y = new_real * sine + new_imag * cosi;

            fft_shift[n] = C(x, y);

            fft_shift[N - n] = conj(fft_shift[n]);
            
        }
        float* y = IFFT(fft_shift, N);
        delete[] fft;
        return y;
    }
};

#endif