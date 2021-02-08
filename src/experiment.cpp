/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Copyright by Nicola Nicolici
Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "filter.h"
#include "fourier.h"
#include "genfunc.h"
#include "iofunc.h"
#include "logfunc.h"
#include <cmath>
#include <complex.h>

// Slice Function for sub-vectors
std::vector<float> slice(std::vector<float>arr,int low, int high)
{
    auto start = arr.begin() + low;
    auto end = arr.begin() + high + 1;
    std::vector<float>  result(high - low + 1);
    copy(start, end, result.begin());
    return result;
}

// Slice Function for sub-vectors
std::vector<std::complex<float>> complex_slice(std::vector<std::complex<float>> arr,int low, int high)
{
    auto start = arr.begin() + low;
    auto end = arr.begin() + high + 1;
    std::vector<std::complex<float>>  result(high - low + 1);
    copy(start, end, result.begin());
    return result;
}

void estimatePSD(const std::vector<float> &samples, float Fs, std::vector<float> &freq, std::vector<float> &psd_est) {


	int freq_bins = NFFT;
	float df = Fs/freq_bins;

	float currFs = 0;
	while (currFs < Fs/2){
		freq.push_back(currFs);
		currFs += df;
	}

	std::vector<float> hann(freq_bins);
	for (auto i =0 ; i<hann.size() ; i++){
		hann[i] = pow(sin(i*PI/freq_bins),2);
	}

	int no_segments = floor(samples.size()/freq_bins);

	std::vector<float> windowed_samples(freq_bins,0.0);
	std::vector<float> sample_window;
	std::vector<std::complex<float>> Xf;
	std::vector<float> psd_list;
	std::vector<float> arr(freq_bins, 0.0);
	psd_est = arr;

	for (auto k = 0 ; k<no_segments ; k++){

		sample_window = slice(samples, k*freq_bins, (k+1)*freq_bins);
		for(auto i = 0 ; i< freq_bins ; i++){
			windowed_samples[i] = hann[i] * sample_window[i];
		}

		DFT(windowed_samples, Xf);
		Xf = complex_slice(Xf, 0, freq_bins/2);

		for(auto i = 0 ; i<Xf.size() ; i++){
			psd_list.push_back(10*log10( 2/(Fs*freq_bins/2) * pow(abs(Xf[i]),2) ));
		}
	}

	for(auto k = 0 ; k<freq_bins/2 ; k++){
		for(auto l = 0 ; l<no_segments ; l++){
			psd_est[k] += psd_list[k + l*(freq_bins/2)];
		}
		psd_est[k] = psd_est[k] / no_segments;
	}
}

int main()
{
	// binary files can be generated through the
	// Python models from the "../model/" sub-folder
	const std::string in_fname = "../data/fm_demod_10.bin";
	std::vector<float> bin_data;
	readBinData(in_fname, bin_data);

	// generate an index vector to be used by logVector on the X axis
	std::vector<float> vector_index;
	genIndexVector(vector_index, bin_data.size());
	// log time data in the "../data/" subfolder in a file with the following name
	// note: .dat suffix will be added to the log file in the logVector function
	logVector("demod_time", vector_index, bin_data);

	// take a slice of data with a limited number of samples for the Fourier transform
	// note: NFFT constant is actually just the number of points for the
	// Fourier transform - there is no FFT implementation ... yet
	// unless you wish to wait for a very long time, keep NFFT at 1024 or below
	std::vector<float> slice_data = \
	std::vector<float>(bin_data.begin(), bin_data.begin() + NFFT);
	// note: make sure that binary data vector is big enough to take the slice

	// declare a vector of complex values for DFT
  	std::vector<std::complex<float>> Xf;
	// compute the Fourier transform
	// the function is already provided in fourier.cpp
	DFT(slice_data, Xf);
	// compute the magnitude of each frequency bin
	// note: we are concerned only with the magnitude of the frequency bin
	// (there is NO logging of the phase response, at least not at this time)
	std::vector<float> Xmag;
	// compute the magnitude of each frequency bin
	// the function is already provided in fourier.cpp
	computeVectorMagnitude(Xf, Xmag);

	// log the frequency magnitude vector
	vector_index.clear();
	genIndexVector(vector_index, Xmag.size());
	logVector("demod_freq", vector_index, Xmag); // log only positive freq

	std::vector<float> freq;
	std::vector<float> psd_est;
 	float Fs=(2.4e6/10)/1e3;
	estimatePSD(bin_data,Fs,freq,psd_est);

	vector_index.clear();
	genIndexVector(vector_index, psd_est.size());
	logVector("demod_psd", vector_index, psd_est);

	// if you wish to write some binary files, see below example
	// const std::string out_fname = "../data/outdata.bin";
	// writeBinData(out_fname, bin_data);

	// naturally, you can comment the line below once you are comfortable to run gnuplot
	std::cout << "Run: gnuplot -e 'set terminal png size 1024,768' example.gnuplot > ../data/example.png\n";

	return 0;
}
