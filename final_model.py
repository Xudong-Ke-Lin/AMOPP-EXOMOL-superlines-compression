#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:09:36 2020

@author: xudongke
"""

import re
import numpy as np
import pandas as pd
from glob import glob
from utils import *


def main():
    # get filenames
    filenames = glob('data/28Si-16O2/28Si-16O2__OYT3__0-6000__*K__grid-6-w-4000-SL.super.bz2')
    filenames.sort(key=natural_keys)
    
    # temperature list
    temp_length = len(filenames)
    temp = np.zeros(temp_length)
    for i,file in enumerate(filenames):
        res = re.findall("(\d+)K", file)
        temp[i] = int(res[0])
    
    
    # load partition function data
    data_pf = pd.read_csv("data/28Si-16O2/28Si-16O2__OYT3.pf.txt", delim_whitespace=True
                          , names=['temperature', 'partition function'], header=None)
    
    # load intensity data
    df_all = [pd.read_csv(f,names=['wavenumber','intensity'], 
                          delim_whitespace=True, header=None) for f in filenames]
    
    # get index of the first temperature with common zero intensities
    index = get_ref_index(df_all)
    
    # index of common zero intensities
    zero_intensities = df_all[index][df_all[index]['intensity']==0].index
    # drop common zero intensities from data
    for i in range(temp_length):
        df_all[i].drop(zero_intensities,inplace=True)
        
    # non-zero-intentities wavenumbers
    wavenumber_non_zero = df_all[0]['wavenumber'].to_numpy()
    # avoid zero wavenumber
    wavenumber_non_zero = np.delete(wavenumber_non_zero,np.where(wavenumber_non_zero==0.0))
     
    # guess for Einstein coefficients & effecrtive energies
    pw0 = (1e2,1e2,10000,15000)
    

    # get Einstein coeff. & effective energies
    data,wavenumber_error = get_parameters(df_all, data_pf, temp, 
                                           wavenumber_non_zero, pw0)
    
    # save unfitted wavenumbers
    np.savetxt('wavenumber_error_SiO2_final.txt',wavenumber_error,delimiter=" ",fmt='%.6f')
    
    # starting ID
    start = 5688943
    # total degeneracy
    g = 1
    # rotational quantum number
    J = 0
    # quantum number
    quantum_number = 's'
    
    # get states and trans files
    df_states, df_transitions = get_exomol_files(data, start, g, J, quantum_number)
    # change format and save
    df_states.E = df_states.E.map('{:.6f}'.format)
    df_states.to_csv('SiO2_final.states',header=None, index=None, sep="\t") 
    df_transitions.A = df_transitions.A.map('{:.4E}'.format)
    df_transitions.to_csv('SiO2_final.trans',header=None, index=None, sep="\t") 

if __name__ == '__main__':
    main()
