#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Copyright by Nicola Nicolici
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
import math

# use "custom" fmDemodArctan
from fmSupportLib import fmDemodArctan
from fmSupportLib import fmDemodFriendly


# the radio-frequency (RF) sampling rate
# this sampling rate is either configured on RF hardware
# or documented when a raw file with IQ samples is provided
rf_Fs = 2.4e6

# the cutoff frequency to extract the FM channel from raw IQ data
rf_Fc = 100e3

# the number of taps for the low-pass filter to extract the FM channel
# this default value for the width of the impulse response should be changed
# depending on some target objectives, like the width of the transition band
# and/or the minimum expected attenuation from the pass to the stop band
rf_taps = 151

# the decimation rate when reducing the front end sampling rate (i.e., RF)
# to a smaller samping rate at the intermediate frequency (IF) where
# the demodulated data will be split into the mono/stereo/radio data channels
rf_decim = 10

# audio sampling rate (we assume audio will be at 48 KSamples/sec)
audio_Fs = 48e3

# complete your own settings for the mono channel
# (cutoff freq, audio taps, decimation rate, ...)
audio_Fc = 1.6e4
audio_taps = 151
audio_decim = 5

def convolution(x, h):
	M = len(x)
	N = len(h)
	y = np.zeros(x.shape[0])
	for n in range(M+N-1): #finite sequence, thus bound ensures all values are covered
		for k in range(N+1):
			if((n-k)>=0 and k<=(N-1) and (n-k)<=(M-1)): #conditions which result in invalid indexing (convolution value of 0)
				y[n] += x[n - k]*h[k] #convolution summation
		if n==len(x)-1: #if block size reached, break from loop
			break
	return y


def lowPass(fc, fs, taps):
	nc = fc/(fs/2)
	h = [0]*taps
	for i in range(taps):
		if(i ==(taps-1)/2):
			h[i] = nc
		else:
			h[i] = nc*( (np.sin(np.pi*nc*(i - (taps-1)/2)))/(np.pi*nc*(i - (taps-1)/2)))
		h[i] = h[i] * (np.sin((i*np.pi)/taps)**2)
	return h



if __name__ == "__main__":

	# read the raw IQ data from the recorded file
	# IQ data is normalized between -1 and +1 and interleaved
	in_fname = "../data/iq_samples.raw"
	iq_data = np.fromfile(in_fname, dtype='float32')
	print("Read raw RF data from \"" + in_fname + "\" in float32 format")


	#SIGNAL FLOW GRAPH

	# coefficients for the front-end low-pass filter
	# rf_coeff = signal.firwin(rf_taps, rf_Fc/(rf_Fs/2), window=('hann'))


	#************************TAKEHOME EXERCISE #1****************************************
	rf_coeff = lowPass(rf_Fc, rf_Fs, rf_taps)


	#IN PHASE/QUADURATURE PASSED TO LOW PASS FILTER.
	# filter to extract the FM channel (I samples are even, Q samples are odd)
	i_filt = signal.lfilter(rf_coeff, 1.0, iq_data[0::2])
	q_filt = signal.lfilter(rf_coeff, 1.0, iq_data[1::2])
	# i_filt = convolution(iq_data[0::2], rf_coeff)
	# q_filt = convolution(iq_data[1::2], rf_coeff)

	# DOWNSAMPLE the FM channel (Reduces sample rate by 10)
	i_ds = i_filt[::rf_decim]
	q_ds = q_filt[::rf_decim]

	# FM demodulator (check the library)
	# fm_demod, dummy = fmDemodFriendly(i_ds, q_ds) 
	fm_demod, dummy = fmDemodArctan(i_ds, q_ds) 

	# we use a dummy because there is no state for this single-pass model

	# set up drawing
	fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
	fig.subplots_adjust(hspace = 1.0)

	# PSD after FM demodulation
	ax0.psd(fm_demod, NFFT=512, Fs=(rf_Fs/rf_decim)/1e3)
	ax0.set_ylabel('PSD (db/Hz)')
	ax0.set_title('Demodulated FM')


	# coefficients for the filter to extract mono audio
	# audio_coeff = signal.firwin(audio_taps, audio_Fc/(audio_Fs/2), window=('hann'))


	#************************TAKEHOME EXERCISE #1****************************************
	audio_coeff = lowPass(audio_Fc, audio_Fs, audio_taps)
	
	# extract the mono audtio data through filtering
	
	#To make this program more efficent, instead of filtering all data, then down
	#sampling just the 5th element, we might as well just pass every 5th element to the filter
	#to have an improved run-time
	audio_filt = signal.lfilter(audio_coeff, 1.0, fm_demod[::audio_decim])
	# audio_filt = convolution(fm_demod[::audio_decim], audio_coeff)

	# you should uncomment the plots below once you have processed the data

	# PSD after extracting mono audio
	ax1.psd(audio_filt, NFFT=512, Fs=(audio_Fs/audio_decim)/1e3)
	ax1.set_ylabel('PSD (db/Hz)')
	ax1.set_title('Extracted Mono')

	# downsample audio data
	audio_data = audio_filt

	# PSD after decimating mono audio
	ax2.psd(audio_data, NFFT=512, Fs=audio_Fs/1e3)
	ax2.set_ylabel('PSD (db/Hz)')
	ax2.set_title('Mono Audio')

	# write audio data to file (assumes audio_data samples are -1 to +1)
	wavfile.write("../data/fmMonoBasic.wav", int(audio_Fs), np.int16((audio_data/2)*32767))
	# during FM transmission audio samples in the mono channel will contain
	# the sum of the left and right audio channels; hence, we first
	# divide by two the audio sample value and then we rescale to fit
	# in the range offered by 16-bit signed int representation

	print('Finished processing the raw I/Q samples')

	# save PSD plots
	fig.savefig("../data/fmMonoBasic.png")
	plt.show()


	
