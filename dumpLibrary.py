import numpy as np
import math
from numpy import fft

def twiddle_factor(k,N):
	w = np.exp(-1j*2*np.pi*k/N)
	return w

def radix2_arrange(x):
	k = int(math.log(len(x),2))
	x_new = (x,)
	for i in range (0,k-1):
		x_temp = x_new
		x_new = []
		for j in range (0,2**i):
			x_new.append(x_temp[j][0::2])
			x_new.append(x_temp[j][1::2])
	return x_new

def fft(x):
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    T= [np.exp(-2j*np.pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]

def batch_fft(X):
	fft = []
	for point in X:
		fft_t = np.abs(fft(point))/8192
		fft.append(fft_t[:int(len(fft_t))+1])
	return fft

#features

def crest_factor(x,k):
	s1 = k*np.max(x)/np.sum(np.sqrt(x**2))
	return s1

def shape_factor(x,k):
	s2 = np.sqrt((k)*(np.sum(x**2))*np.sum(np.abs(x)))
	return s2

def absolute_mean_amplitude(x,k):
	s3 = (1/k)*(np.sum(np.abs(x)))
	return s3

def square_root_amplitude(x,k):
	s4 = ((1/k)*np.sum(np.sqrt(np.abs(x))))**2
	return s4

def kurtosis(x,k):
	s5 = (1/k)*np.sum((x)**4)
	return s5

def variance_values(x,k):
	s6 = (1/k)*np.sum(x**2)
	return s6

def clearance_factor(x,k):
	s7 = (k**2)*np.max(x)/(np.sum(np.sqrt(x**2)))**2
	return s7

def impulse_indicator(x,k):
	s8 = k*np.max(np.abs(x))/np.sum(np.abs(x))
	return s8

def skewness_factor(x,k):
	s9 = (1/k)*np.sum((x)**3)
	return s9

def generate_features(X):
	features = []

	if len(X)>2000:
		point = X
		k = len(point)
		ftrs = [crest_factor(point,k), shape_factor(point,k),
		absolute_mean_amplitude(point,k), square_root_amplitude(point,k),
		kurtosis(point,k), variance_values(point,k), clearance_factor(point,k),
		impulse_indicator(point,k), skewness_factor(point,k)]
		features.append(ftrs)

	else:
		for point in X:
			k = len(point)
			ftrs = [crest_factor(point,k), shape_factor(point,k),
			absolute_mean_amplitude(point,k), square_root_amplitude(point,k),
			kurtosis(point,k), variance_values(point,k), clearance_factor(point,k),
			impulse_indicator(point,k), skewness_factor(point,k)]
			features.append(ftrs)

	return features

def hilbert_transform(X,detail):
	from numpy import fft
	if detail == 'freq':
		h = fft.ifft((X*-1j))
	elif detail == 'time':
		h = fft.ifft((fft.fft(X))*-1j)
	return h
