# AUTOTUNE

A phase vocoding autotune on a microcontroller

The motivation for this project was seeing the overlap between the optimized Yin Algorithm pitch tracking (based around an FFT autocorrelation) and phase vocoding pitch shifting (also based around an FFT). I wanted to see how far I could take real-time FFT on a limited processor like STM32H750 based Daisy Seed. I implemented a recursive FFT which is definitely not the most efficient form of FFT. A mixed-radix approach might work nicer to smooth phase-accumulator discontinuities. 