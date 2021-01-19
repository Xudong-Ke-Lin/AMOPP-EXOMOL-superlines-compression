#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:01:38 2020

@author: xudongke
"""

import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def atoi(text):
    '''used in natural_keys()'''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_ref_index(data):
    '''get the starting index where there are not zeros everywhere'''
    N_0 = 0
    for i,df in enumerate(data):
        N = len(df[df['intensity']==0])
        if N_0 == N:
            return i-1
        else:
            N_0 = N

def intensity_fitting(wavenumber,pf_array):
    '''Function used to fit intensities using Einstein coeff. & effective energies
    Inputs: wavenumber: specific wavenumber
            pf_array: partition function for given temperatures
    Outputs: intensity, same length as pf_array'''
    # speed of light
    c = 2.99792458e+10 # cm/s
    pi = np.pi
    cmcoef = 1/(8*pi*c)
    # constant
    c2 = 1.4387770 # cm K
    # upper state degeneracy
    g_f = 1
    def intensity(temp_array,A1,A2,E1,E2):
        '''Compute intensities for fixed wavenumber and partition function using eq. 1
        S(f<-i)=cmcoef*A*g_f*exp(-c2*E_i/T)*(1-exp(-c2*nu/T))/(nu^2*Q(t))
        cmcoef=1/(8*Pi*c)
        S1 =  cmcoef*A1*exp(-c2*E1/T)*(1-exp(-c2*nu/T))/(nu^2*pf(T))
        S2 =  cmcoef*A2*exp(-c2*E2/T)*(1-exp(-c2*nu/T))/(nu^2*pf(T))
        S(total) = S1+S2
        Inputs: temp_array: temperature array, same length as pf_array
                A1, E1: Einstein coeff. 1 & effective energy 1
                A2, E2: Einstein coeff. 2 & effective energy 2
        Outputs: S(total)'''
        # r = 1 - exp(-c2 * wavenumber/T)
        r_array = 1-np.exp(-c2*wavenumber/temp_array)
        S1 = cmcoef*A1*g_f*np.exp(-c2*E1/temp_array)*r_array/(wavenumber**2 * pf_array)
        S2 = cmcoef*A2*g_f*np.exp(-c2*E2/temp_array)*r_array/(wavenumber**2 * pf_array)
        return S1 + S2
    return intensity

def intensity_fitting_single(wavenumber,pf_array):
    '''Function used to fit intensities using one Einstein coeff. & effective energy
    Inputs: wavenumber: specific wavenumber
            pf_array: partition function for given temperatures
    Outputs: intensity, same length as pf_array'''
    # speed of light
    c = 2.99792458e+10 # cm/s
    pi = np.pi
    cmcoef = 1/(8*pi*c)
    # constant
    c2 = 1.4387770 # cm K
    # upper state degeneracy
    g_f = 1
    def intensity(temp_array,A,E):
        '''Compute intensities for fixed wavenumber and partition function using eq. 1
        S(f<-i)=cmcoef*A*g_f*exp(-c2*E_i/T)*(1-exp(-c2*nu/T))/(nu^2*Q(t))
        cmcoef=1/(8*Pi*c)
        S =  cmcoef*A*exp(-c2*E/T)*(1-exp(-c2*nu/T))/(nu^2*pf(T))
        Inputs: temp_array: temperature array, same length as pf_array
                A, E: Einstein coeff. 1 & effective energy
        Outputs: S'''
        # r = 1 - exp(-c2 * wavenumber/T)
        r_array = 1-np.exp(-c2*wavenumber/temp_array)
        S = cmcoef*A*g_f*np.exp(-c2*E/temp_array)*r_array/(wavenumber**2 * pf_array)
        return S
    return intensity

def get_initial_guesses(data, wavenumber_array, pf, temp, pw0):
    '''Calculate the initial guesses
    Inputs: data: all the intensity data 
            data_pf: partition function for the given temp
            temp: temperature array
            pw0: guess of Einstein coeff. & effective energies (A1,A2,E1,E2)
    Outputs: pw0s_mean: mean initial guess per batches'''
    # total wavenumber length
    total_wavenumber_legth = len(wavenumber_array)
    
    # number of batches (and guesses)
    number_batches = 100
    # number of wavenumber sample in one batch
    wavenumber_size = 10
    # step in the total wavenumber array
    wavenumber_step = int(total_wavenumber_legth/(wavenumber_size*number_batches-1))
    # wavenumber sample, size=number_batches x wavenumber_size
    wavenumber_sample = wavenumber_array[::wavenumber_step]
    # length of wavenumber array
    wavenumber_length = len(wavenumber_sample)
    # intensities corresponding to wavenumber_sample
    intensity = data[::wavenumber_step]
    
    # array of parameters for wavenumber_sample
    pw0s = np.zeros((wavenumber_length,4))
    
    for i,wavenumber in enumerate(wavenumber_sample):
        pw, cov = curve_fit(intensity_fitting(wavenumber,pf), temp, 
                            intensity[i], pw0,maxfev=int(1e8))
        # if negative Einstein coefficient, use single fitting
        if (pw[0] < 0.0) or (pw[1] < 0.0):
            pw0_single = (pw0[0],pw0[2])
            pw, cov = curve_fit(intensity_fitting_single(wavenumber,pf), 
                                temp, intensity[i], pw0_single,maxfev=int(1e8))
            # use same parameter for both
            pw0s[i] = [pw[0],pw[0],pw[1],pw[1]]
        else:
            pw0s[i] = pw
    
    # array of the mean parameter per batch
    pw0s_mean = np.zeros((number_batches,4))
    for batch in range(number_batches):
        pw0s_mean[batch] = np.mean(pw0s[batch*wavenumber_size:wavenumber_size*(batch+1)],axis=0)
        
    # round to the magnitude
    pw0s_mean = 10**np.round(np.log10(pw0s_mean)) 
    # keep the energies using original guess (good for H2O and SiO2)
    pw0s_mean[:,2:4] = [pw0[2],pw0[3]]
    
    # replace bigger pw[0] (low temp Eins. coeff) to pw[1] (high temp Eins. coeff)
    pw0s_mean[np.where(pw0s_mean[:,0]>pw0s_mean[:,1]),0]=pw0s_mean[np.where(pw0s_mean[:,0]>pw0s_mean[:,1]),1]
    
    return pw0s_mean

def get_parameters(data, data_pf, temp, wavenumber_array, pw0):
    '''Calculate the Einstein coeff. & effective energies using eq.1 for a range of wavenumbers
    Inputs: data: all the data (wavenumbers and intensities; at least one non-zero intensity)
            data_pf: partition function data
            temp: temperature array
            wavenumber_array: wavenumber array
            pw0: guess of Einstein coeff. & effective energies (A1,A2,E1,E2)
            index: reference index
    Outputs: data: dataframe containing waveumbers, einstein coeff. and energies
             wavenumber_error: unfitted wavenumbers due to RuntimeError'''
    
    # length of wavenumber array
    wavenumber_length = len(wavenumber_array)
    # length of temperature array
    temp_length = len(temp)

    # create intensity data for the wavenumber range
    intensity = np.zeros((wavenumber_length,temp_length))
    for i in range(temp_length):
        intensity[:,i] = data[i][data[i]['wavenumber'].isin(wavenumber_array)]['intensity']

    # partition function data for the given temperatures
    pf = data_pf.loc[data_pf['temperature'].isin(temp)]['partition function'].to_numpy()    

    # arrays for wavenumbers, Einstein coeff. and eff. energies
    wavenumber_list = []
    wavenumber_error = []
    einstein_coeff = []
    energy = []
    
    # get initial guesses
    pw0s_mean = get_initial_guesses(intensity, wavenumber_array, pf, temp, pw0)
    number_batches = len(pw0s_mean)
    
    # start count
    i = 0
    for batch,pw0_batch in enumerate(pw0s_mean):
        start = round(batch*wavenumber_length/number_batches)
        end = round((batch+1)*wavenumber_length/number_batches)
        print(f'batch={batch},pw0={pw0_batch},wavenumber from {wavenumber_array[start]}')
        # single energy fit initial guess
        pw0_single = (pw0_batch[0],pw0_batch[2])
        for wavenumber in wavenumber_array[start:end]:
            # try dual fit
            try:
                pw, cov = curve_fit(intensity_fitting(wavenumber,pf), temp, intensity[i], pw0_batch,maxfev=20000)
                
                # if negative Einstein coefficient
                if (pw[0] < 0.0) or (pw[1] < 0.0):
                    # try dual fit with different initial guess
                    for scale in [1e1,1e-1,1e2,1e-2]:
                        
                        try:
                            # use scale for Einstein coeff.
                            pw, cov = curve_fit(intensity_fitting(wavenumber,pf), 
                                                temp, intensity[i], 
                                                [pw0_batch[0]*scale,pw0_batch[1]*scale,pw0_batch[2],pw0_batch[3]], 
                                                maxfev=20000)
                        except RuntimeError:
                            # if RuntimeError after all the scales
                            if scale==1e-2:
                                # try single fit
                                try:
                                    pw, cov = curve_fit(intensity_fitting(wavenumber,pf), temp, intensity[i], 
                                                        pw0_batch,maxfev=20000)
                                    wavenumber_list.append(wavenumber)
                                    einstein_coeff.append([pw[0],0])
                                    energy.append([pw[1],0])
                                except RuntimeError:
                                    wavenumber_error.append(wavenumber)
                            continue                   
                        
                        # if negative Einstein coefficient
                        if (pw[0] < 0.0) or (pw[1] < 0.0):
                            # if negative Einstein coefficient after all the scales
                            if scale==1e-2:
                                # try single fit
                                try:
                                    pw, cov = curve_fit(intensity_fitting(wavenumber,pf), temp, intensity[i], 
                                                        pw0_batch,maxfev=20000)
                                    wavenumber_list.append(wavenumber)
                                    einstein_coeff.append([pw[0],0])
                                    energy.append([pw[1],0])
                                except RuntimeError:
                                    wavenumber_error.append(wavenumber)
                            continue
                        
                        # append when not negative Einstein coefficient after using scale
                        else:
                            wavenumber_list.append(wavenumber)
                            einstein_coeff.append([pw[0],pw[1]])
                            energy.append([pw[2],pw[3]])
                            break
                         
                # append when not negative Einstein coefficient
                else:
                    wavenumber_list.append(wavenumber)
                    einstein_coeff.append([pw[0],pw[1]])
                    energy.append([pw[2],pw[3]])
                    
            # fitting exceeds maxfev
            except RuntimeError:
                try:
                    pw, cov = curve_fit(intensity_fitting_single(wavenumber,pf), 
                                            temp, intensity[i], pw0_single, maxfev=20000)
                    wavenumber_list.append(wavenumber)
                    einstein_coeff.append([pw[0],0])
                    energy.append([pw[1],0])
                except RuntimeError:
                    wavenumber_error.append(wavenumber)
            i += 1
    
    wavenumber_list = np.array(wavenumber_list)
    einstein_coeff = np.array(einstein_coeff)
    energy = np.array(energy)
    wavenumber_error = np.array(wavenumber_error)
    
    # total wavenumber length
    wavenumber_list_length = len(wavenumber_list)
    data = pd.DataFrame(columns=['wavenumber','A','E'])
    data.wavenumber = np.zeros(2*wavenumber_list_length)
    data.wavenumber.iloc[::2] = wavenumber_list
    data.wavenumber.iloc[1::2] = wavenumber_list
    data.A.iloc[::2] = einstein_coeff[:,0]
    data.A.iloc[1::2] = einstein_coeff[:,1]
    data.E.iloc[::2] = energy[:,0]
    data.E.iloc[1::2] = energy[:,1]
    # drop zero Einstein coefficients
    data.drop(data[data.A==0.0].index,inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data,wavenumber_error

def get_exomol_files(data, start, g, J, quantum_number):
    '''Get states and trans file using Exomol format
    Inputs: data: energy dataframe containing wavenumbers Einstein coefficients
                  and effective energies
            start: starting ID
            g: total degeneracy
            J: rotational quantum number
            quantum_number: quantum numbers
    Outputs: df_states: states dataframe
             df_trans: trans dataframe'''
    
    # total data length 
    total_length = len(data) 
    
    # states dataframe
    df_states = pd.DataFrame(columns=['ID','E','g','J','quantum numbers'])  
    end = start + 2*total_length 
    ID = np.arange(start,end)  
    df_states.ID = ID 
    df_states.E = np.zeros(2*total_length) 
    df_states.E.iloc[::2] = data.E.to_numpy() 
    df_states.E.iloc[1::2] = data.E.to_numpy() + data.wavenumber.to_numpy() 
    df_states.g = g
    df_states.J = J
    df_states['quantum numbers'] = quantum_number
    
    # trans dataframe
    df_transitions = pd.DataFrame(columns=['ID_u','ID_l','A'])
    df_transitions.ID_u = ID[1::2] 
    df_transitions.ID_l = ID[::2]
    df_transitions.A = data.A
    
    return df_states, df_transitions
