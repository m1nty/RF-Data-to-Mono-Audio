#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Copyright by Nicola Nicolici
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import math

# use "custom" fmDemodArctan
from fmSupportLib import fmDemodArctan
from fmSupportLib import fmDemodFriendly

rf_Fs = 2.4e6
rf_Fc = 100e3
rf_taps = 151
rf_decim = 10

audio_Fs = 48e3
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

	# # coefficients for the front-end low-pass filter
	# rf_coeff = signal.firwin(rf_taps, \
	# 						rf_Fc/(rf_Fs/2), \
	# 						window=('hann'))

	# # coefficients for the filter to extract mono audio
	# audio_coeff = signal.firwin(audio_taps, \
	# 						audio_Fc/(audio_Fs/2), \
	# 						window=('hann'))

	#************************TAKEHOME EXERCISE #2****************************************

	rf_coeff = lowPass(rf_Fc, rf_Fs, rf_taps)
	audio_coeff = lowPass(audio_Fc, audio_Fs, audio_taps)

	# set up drawing
	fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
	fig.subplots_adjust(hspace = 1.0)

	# select a block_size that is in KB and
	# a multiple of decimation factors
	block_size = 1024 * rf_decim * audio_decim * 2
	block_count = 0

	# states needed for continuity in block processing
	state_i_lpf_100k = np.zeros(rf_taps-1)
	state_q_lpf_100k = np.zeros(rf_taps-1)
	state_phase = 0
	# add state as needed for the mono channel filter
	state_audio_lpf_100k = np.zeros(audio_taps-1)
	# audio buffer that stores all the audio blocks
	audio_data = np.array([]) # to be updated by you during in-lab

	# if the number of samples in the last block is less than the block size
	# it is fine to ignore the last few samples from the raw IQ file
	while (block_count+1)*block_size < len(iq_data):
		# if you wish to have shorter runtimes while troubleshooting
		# you can control the above loop exit condition as you see fit

		print('Processing block ' + str(block_count))

		# filter to extract the FM channel (I samples are even, Q samples are odd)

		#************************TAKEHOME EXERCISE #2****************************************

		i_filt, state_i_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
				iq_data[(block_count)*block_size:(block_count+1)*block_size:2],
				zi=state_i_lpf_100k)
		q_filt, state_q_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
				iq_data[(block_count)*block_size+1:(block_count+1)*block_size:2],
				zi=state_q_lpf_100k)
		# print("here")
		# i_filt = convolution(iq_data[(block_count)*block_size:(block_count+1)*block_size:2], rf_coeff)
		# print("here1")
		# q_filt = convolution(iq_data[(block_count)*block_size+1:(block_count+1)*block_size:2], rf_coeff)
		# print("here2")

		# downsample the FM channel
		i_ds = i_filt[::rf_decim]
		q_ds = q_filt[::rf_decim]

		# FM demodulator
		# fm_demod, state_phase = fmDemodArctan(i_ds, q_ds, state_phase)
		fm_demod, state_phase = fmDemodFriendly(i_ds, q_ds, state_phase)


		#************************TAKEHOME EXERCISE #2****************************************

		#To make this program more efficent, instead of filtering all data, then down
		#sampling just the 5th element, we might as well just pass every 5th element to the filter
		#to have an improved run-time
		# extract the mono audtio data through filtering
		audio_filt = signal.lfilter(audio_coeff, 1.0, fm_demod[::audio_decim])
		# audio_filt = convolution(fm_demod[::audio_decim], audio_coeff)

		# downsample audio data
		audio_block = audio_filt

		# concatanete most recently processed audio_block
		# to the previous blocks stored in audio_data
		audio_data = np.concatenate((audio_data, audio_block))

		# to save runtime select the range of blocks to log iq_data
		# this includes both saving binary files as well plotting PSD
		# below we assume we want to plot for graphs for blocks 10 and 11
		if block_count >= 10 and block_count < 12:
			# PSD after FM demodulation
			ax0.clear()
			ax0.psd(fm_demod, NFFT=512, Fs=(rf_Fs/rf_decim)/1e3)
			ax0.set_ylabel('PSD (dB/Hz)')
			ax0.set_xlabel('Freq (kHz)')
			ax0.set_title('Demodulated FM (block ' + str(block_count) + ')')
			# output binary file name (where samples are written from Python)
			fm_demod_fname = "../data/fm_demod_" + str(block_count) + ".bin"
			# create binary file where each sample is a 32-bit float
			fm_demod.astype('float32').tofile(fm_demod_fname)

			# PSD after extracting mono audio
			ax1.clear()
			ax1.psd(audio_filt, NFFT=512, Fs=(rf_Fs/rf_decim)/1e3)
			ax1.set_ylabel('PSD (dB/Hz)')
			ax1.set_xlabel('Freq (kHz)')
			ax1.set_title('Extracted Mono')

			# PSD after decimating mono audio
			ax2.clear()
			ax2.psd(audio_block, NFFT=512, Fs=audio_Fs/1e3)
			ax2.set_ylabel('PSD (dB/Hz)')
			ax2.set_xlabel('Freq (kHz)')
			ax2.set_title('Mono Audio')

			# save figure to file
			fig.savefig("../data/fmMonoBlock" + str(block_count) + ".png")

		block_count += 1

	print('Finished processing the raw I/Q samples')

	# write audio data to a .wav file (assumes audio_data samples are -1 to +1)
	wavfile.write("../data/fmMonoBlock.wav", int(audio_Fs), np.int16((audio_data/2)*32767))

	# uncomment assuming you wish to show some plots
	plt.show()
