#include "daisy_seed.h"
#include "daisysp.h"
#include "circular_buffer.h"
#include <deque>
#include <math.h>
#include "complex.h"
#include "dead_fft.h"

using namespace daisy;
using namespace daisysp;

#define N 			512
#define OVERLAP 	3
#define HOP 		N / OVERLAP
#define NUM_SCALES 	5

static int Fs;

static CircularBuffer<float> input(N);
static std::deque<float*> output;

// Scales (first number is number of notes in the scale)
static int major[] 		= {7, 0, 2, 4, 5, 7, 9, 11};
static int minor[] 		= {7, 0, 2, 3, 5, 7, 8, 10};	
static int majPent[]	= {5, 0, 2, 4, 7, 9};
static int minPent[]	= {5, 0, 3, 5, 7, 10};
static int chromatic[]	= {12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};		

// all possible scales
static int *scales[] = {
	major, minor, majPent, minPent, chromatic
};

static double win[N] = {0};
static double phs_accu[N];
static float NOTES[128];
static int *scale;
static int offset = 0;

DaisySeed hw;

float nearest_note(float target) {
	float min_diff = NOTES[127];
	int idx = 0;
	int rep = 128 / 12;
	for (int r = 0; r < rep; r++) {
		for (int i = 1; i <= scale[0]; i++) {
			// if the note exists in our scale
			int note = (scale[i] + offset) % 12;
			float diff = abs(target - NOTES[note + (r * 12)]);
			if (diff < min_diff) {
				min_diff = diff;
				idx = note + (r * 12);
			}
		}
	}
	return NOTES[idx];
}

void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size) {
	// read pot 1 to set offset
	offset = static_cast<int>(hw.adc.GetFloat(0) * 12.) % 12;

	// read pot 2 to set scale
	int pos = static_cast<int>(hw.adc.GetFloat(1) * NUM_SCALES) % NUM_SCALES;
	scale = scales[pos];
	
	// reading audio and setting up windowing
	for(int i = 0; i < N; i ++) {
		if (win[i] == 0) {
			win[i] = 0.5f * (1.0f - cosf(TWOPI_F * (float) i / (float)(N)));
		}
	}

	for (size_t i = 0; i < size; i++) {
		input.write((in[0][i] + in[1][i])/2.0);
		out[0][i] = 0;
		out[1][i] = 0;
	}

	delete[] output.back();
	output.pop_back();

	float x[N];
	float x_win[N];
	for (int n = 0; n < N; n++) {
		x[n] = input.read(n);
		x_win[n] = input.read(n) * win[n];
	}

	// detecting pitch
	float* cmndf = DeadFFT::CMNDF(x, N);
	int period = DeadFFT::period(cmndf, N, 0.6);
	delete[] cmndf;

	// pitch shifting if tonal, doing nothing otherwise
	float* y;
	if (period > 16) {
		float pitch = static_cast<float>(Fs) / static_cast<float>(period);
		float target = nearest_note(pitch);
		y = DeadFFT::phase_vocode(x_win, target / pitch, N, size, phs_accu);
	} else {
		// still need to phase vocode to maintain phase accumulator
		y = DeadFFT::phase_vocode(x_win, 1.0, N, size, phs_accu); 
	}

	output.push_front(y);

	for (int w = 0; w < OVERLAP; w++) {
		for (size_t i = 0; i < size; i++) {
			out[0][i] += output[w][i + w * size];
			out[1][i] += output[w][i + w * size];
		}
	}
}

int main(void)
{
	hw.Init();
	hw.SetAudioBlockSize(HOP); // number of samples handled per callback
	hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_16KHZ);

	// initializing 2 adc pins for pots
    AdcChannelConfig adcConfig[2];
	adcConfig[0].InitSingle(hw.GetPin(15));
	adcConfig[1].InitSingle(hw.GetPin(16));

	hw.adc.Init(adcConfig, 2);

	Fs = hw.AudioSampleRate();
	
	for (int w = 0; w < OVERLAP; w++) 
		output.push_front(new float[N]{0});

	for (int n = 0; n < N; n++) 
		phs_accu[n] = 0.0;

	for (int n = 0; n < 128; n++) 
		NOTES[n] = 440.0 * pow(2., (n - 69.0) / 12);

	scale = scales[0];

	hw.adc.Start();
	hw.StartAudio(AudioCallback);

	while(1) {}
}
 