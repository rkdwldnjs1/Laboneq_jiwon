# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:44:12 2024

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from numpy import pi, cos, sin, exp, real, imag
from scipy.optimize import curve_fit, minimize
import msvcrt
from os.path import isfile
from time import time, sleep
import scipy.stats
from scipy.interpolate import interp1d

# In[]

class sFit():
    
    def __init__(self, fit_type, xdata, ydata, init_guess=None):
        
        self.fit_type = fit_type
        self.xdata = xdata
        self.ydata = ydata
        self.t0 = xdata[0]
        self.delta_t = np.diff(xdata)[0]
        if (self.fit_type != 'Cos') and (self.fit_type != 'Exp') and (self.fit_type != 'ExpCos') and (self.fit_type != 'ExpExp') and (self.fit_type != 'Gaussian') and (self.fit_type != 'GaussianCos') and (self.fit_type != 'Storage_Characterization'):
            raise Exception("No such a type!")
        
        func = self.fit_function()
        self.func = func

        if init_guess is None:
            self.initial_guess = self.smart_guess()
        else:
            self.initial_guess = init_guess

        # return curve_fit(func, self.xdata, self.ydata)
    
    def fit_function(self):
        
        if self.fit_type == 'Cos':
            return self.Cos
        elif self.fit_type == 'Exp':
            return self.Exp
        elif self.fit_type == 'ExpExp':
            return self.ExpExp
        elif self.fit_type == 'ExpCos':
            return self.ExpCos
        elif self.fit_type == 'Gaussian':
            return self.Gaussian
        elif self.fit_type == 'GaussianCos':
            return self.GaussianCos
        elif self.fit_type == 'Storage_Characterization':
            return self.Storage_Characterization
        
    def _curve_fit(self):
        return curve_fit(self.func, self.xdata, self.ydata, p0 = self.initial_guess, maxfev=10000)
    
    def smart_guess(self):
        
        
        if self.fit_type == 'Cos':
            
            Amp = (max(self.ydata) - min(self.ydata))/2
            Offset = np.mean(self.ydata)
            
            [freq, delta] = self._fft_results()
            
            print("initial_guess : amp={0}, freq={1}, delta={2}, offset={3}".format(Amp, freq, delta, Offset))
            
            return [Amp, freq, delta, Offset]
        
        elif self.fit_type == 'Exp':
            
            Amp = max(self.ydata) - min(self.ydata)
            Gamma = 1e5
            Offset = (max(self.ydata) - min(self.ydata))/2
            
            return [Amp, Gamma, Offset]
        
        elif self.fit_type == 'ExpExp':
            
            Amp = max(self.ydata) - min(self.ydata)
            alphaSize = 1
            kappa = 1e5
            Offset = (max(self.ydata) - min(self.ydata))/2

            return [Amp, alphaSize, kappa, Offset]
        
        
        elif self.fit_type == 'ExpCos':
            
            Amp = max(self.ydata) - min(self.ydata)
            [freq, delta] = self._fft_results()
            Gamma = 0
            Offset = (max(self.ydata) - min(self.ydata))/2
            
            return [np.abs(Amp), freq, Gamma, delta, Offset]
        
        elif self.fit_type == 'Gaussian':

            Amp = max(self.ydata) - min(self.ydata) if self.ydata[len(self.ydata)//2] > self.ydata[0] else min(self.ydata) - max(self.ydata)
            sigma = self.estimate_sigma_from_fwhm(self.xdata, self.ydata)
            Offset = self.ydata[-1]
            t0 = 0

            print("initial_guess : amp={0}, sigma={1}, offset={2}, t0={3}".format(Amp, sigma, Offset, t0))
            
            return [Amp, sigma, Offset, t0]
        
        elif self.fit_type == 'GaussianCos':

            Amp = (max(self.ydata) - min(self.ydata))/2
            [freq, delta] = self._fft_results()
            sigma = 1
            Offset = np.mean(self.ydata)
            t0 = np.mean(self.xdata)

            print("initial_guess : amp={0}, freq={1}, delta={2}, offset={3}, t0={4}".format(Amp, freq, delta, Offset, t0))
            
            
            return [Amp, freq, sigma, delta, Offset, t0]


    
    def _fft_results(self): 
        
        """return FT results, (freq, phase)"""
        
        fft_y = fftshift(fft(self.ydata))
        freqs = self.freq_axis( self.delta_t, length = len(self.ydata))
        
        freq = self.peak_freq(fft_y, freqs)
        delta = np.angle(fft_y[freqs == freq])[0]
        
        return [freq, delta]
            
    def freq_axis(self, delta_t, length):
        """returns an array of frequencies"""
        
        return np.linspace(-1/(2*delta_t), 1/(2*delta_t), length)
        # return np.linspace(-1/(2*dt), 1/(2*dt) - 1/(length*dt), length)
    
    def peak_freq(self, fft_y, freqs):
        
        """Find the peak frequency in an fft trace. Assumes fftshift has been applied, and ignores the DC componnent
        common usage: peak_freq(*fft_trace(trace, dt)) """
        
        n = len(fft_y)
        startInd = int(n/2+1)
        if n > 4:
            
            amps = abs(fft_y[startInd:n-1])
            maxval = max(amps)
            max_index = np.where(amps==maxval)[0][0]
            
            return freqs[startInd + max_index]
        else:
            return freqs[0]*1e9 # if fft_y is too short, just return some frequency
    
    def estimate_sigma_from_fwhm(self, x_data, y_data):
        y_data = np.array(y_data)
        x_data = np.array(x_data)

        # Step 1: Find half
        half = (np.max(y_data) + np.min(y_data))/2.0

        # Step 2: Find indices where curve crosses half max
        above = y_data > half
        crossing_indices = np.where(np.diff(above.astype(int)) != 0)[0]

        if len(crossing_indices) < 2:
            raise ValueError("Cannot determine FWHM: need two crossing points.")

        # Step 3: Interpolate to find exact crossing points
        x_crossings = []
        for idx in crossing_indices:
            f = interp1d(y_data[idx:idx+2], x_data[idx:idx+2])
            x_crossings.append(float(f(half)))

        # Step 4: Compute FWHM and sigma
        fwhm = abs(x_crossings[1] - x_crossings[0])
        sigma = fwhm / 2.3548

        return sigma
   
    
    # [traceFFT, traceFreqs] = self.fft_trace( trace, dt)
    # Freq = self.peak_freq(traceFFT, traceFreqs)
    # Amp = max(trace)-min(trace)
    # if np.where(trace == max(trace))[0][0] > np.where(trace == min(trace))[0][0]: Amp = -Amp
    # Offset = (max(trace)-min(trace))/2
    # if trace.mean() < 0: Offset = -Offset 
    # Gamma = 0
    # Delta = np.angle(traceFFT[traceFreqs == Freq])[0]
    # return Amp, Freq, Gamma, Delta, Offset
        
    def Line(self, x, a, b):
        return (a*x + b).astype(np.float64)
    def Poly2(x, a, b, c):
        return (a*x**2 + b*x + c).astype(np.float64)
    def Exp(self, t, A, gamma, offset): # 3 args
        return (A*exp(-t*gamma) + offset - A/2).astype(np.float64)
    def Cos(self, t, A, f, delta, offset): # 4 args
        return (A*cos((t-self.t0)*2*pi*f+delta)+offset).astype(np.float64)
    def ExpCos(self,  t, A, f, gamma, delta, offset): # 5 args
        return (A*exp(-t*gamma)*cos((t-self.t0)*2*pi*f+delta) + offset).astype(np.float64)
    def ExpExp(self,  t, A, alphaSize , kappa, offset ):
        return (A*(exp( - abs( alphaSize ) * exp( -kappa*t ) )) + offset).astype(np.float64)
    def Gaussian(self, t, A, sigma, offset, t0): # 3 args
        return (A * exp(-((t-t0)**2)/(2*sigma**2)) + offset).astype(np.float64)
    def GaussianCos(self, t, A, f, sigma, delta, offset, t0): # 5 args
        return (A * exp(-((t-t0)**2)/(2*sigma**2)) * cos((t-t0)*2*pi*f + delta) + offset).astype(np.float64)
    def Storage_Characterization(self, t, A, omega, T1, delta_f, offset): # 5 args
        return (A * cos(omega*exp(-t/(2*T1)*cos(2*pi*delta_f*t)) + offset)).astype(np.float64)