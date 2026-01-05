# In[]
import matplotlib
# matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt

import numpy as np
import qutip as qt
import sys
# import os
# sys.path.append("D:/Software/SHFQC/")
from pathlib import Path

# Helpers:
from laboneq.contrib.example_helpers.generate_device_setup import (
    generate_device_setup_qubits,
)

# LabOne Q:
from laboneq.simple import *

# from Basic_qubit_characterization import Basic_qubit_characterization_experiments

from Bosonic_experiments import Bosonic_experiments

from laboneq.contrib.bloch_simulator_pulse_plotter.inspector.update_inspect import (
    pulse_update,
)

from laboneq.dsl.experiment import pulse_library
from laboneq.contrib.bloch_simulator_pulse_plotter.inspector.update_inspect import (
    pulse_inspector,
)


# In[] initial parameters for experiment (6 components available)

# 같은 physical port를 공유하면, 초기 logical signal line calibration할 때
# LO oscillator와 physical port와 관련한 변수들이 나중에 선언한 것으로 덮어씌워짐.
# (IF oscillator만 별도로 선언 가능)
# drive line과 drive_ef line이 서로 공유하므로, 나중에 선언한 drive_ef line의 변수들로 덮어씌워짐.
# "q0"의 measure, acquire line과 "q1"의 measure, acquire line이 서로 공유하므로, 
# 나중에 선언한 "q1"의 measure, acquire line의 변수들로 덮어씌워짐.
# In[] physical_ports
physical_ports = {
    "measure":{
        "freq_LO": 7.4e9,   # f = f_LO + f_IF (positive sideband)
        "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
        "port_mode": None,
        "delay_signal": None, # Global delay for the signal line, implemented by adjusting the waveform.
        "threshold": None,
        "added_outputs": None, # Added outputs to the signal line's physical channel port. (RTR option enabled)
        "automute": True, # Mute output channel when no waveform is played on it i.e for the duration of delays.
        "range": 10,
        "amplitude": 1.0, # it does not work!
                          # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
    },
    "acquire":{
        "freq_LO": 7.4e9,   # f = f_LO + f_IF (positive sideband) (should be same with measure line)
        "port_delay": (200)*1e-9, # Because readout pulse shape is gussian square, for eliminating the rising edge.
        "port_mode": None,
        "delay_signal": None, # Global delay for the signal line, implemented by adjusting the waveform.
        "threshold": None,
        "added_outputs": None, # Added outputs to the signal line's physical channel port. (RTR option enabled)
        "automute": False, # Mute output channel when no waveform is played on it i.e for the duration of delays.
        "range": -5,
        "amplitude": 1.0, # it does not work!
                          # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
    },
    "drive":{
        "q0":{
            "freq_LO": 4e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
        "q1":{ # 2nd control line port
            "freq_LO": 3.8e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
        "q2":{ # 3rd control line port
            "freq_LO": 4.2e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
        "m0":{ # mapped to port4 
            "freq_LO": 5e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
        "m1":{ # mapped to port5
            "freq_LO": 6.6e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
        "m2":{ # mapped to port6 
            "freq_LO": 5e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
            "voltage_offset": 0,
        },
    }
}
# In[] # qubits_parameters, cavity_parameters
### These parameters will be utilized in defining experimental sequences.
qubits_parameters = {
    # two(cr) qubit gate : each parameter is assigned to each qubit except ef_frequency;

    "q0": {

        #port 1 (15-1)

        "ge_frequency" : 4.28575e9 + 0.05e6,
        "ef_frequency" : 4.2975e9, #4.275e9, #4.3865e9,
        "readout_frequency" : 7.6642e9, 
        ### readout parameters ###
        "readout_amp" : 0.25, #0.25, #0.25, #0.18,
        'readout_pulse_length': 1400e-9,
        "readout_integration_length": (1200)*1e-9,
        "readout_integration_amp": 1,

        "drachma_readout_pulse_length": 1400e-9,
        "drachma_readout_integration_length": (1200)*1e-9,
        "drachma_readout_amp": 0.65,

        "readout_phase": (75) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.308, 
        "drive_pulse_length": 64e-9,
        "drive_beta": 0.015,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.154,
        "pi2_beta": 0.015,

        "pi_length": 64e-9,
        "pi_amp": 0.308,
        "pi_beta": 0.015,

        "cond_pi_length": 2048e-9,
        "cond_pi_amp": 0.011,
        "cond_pi_beta": 0.015,

        "rabi_drive_amp": 0.334/2,
        "ramp_length": 120e-9,

        ### extra parameters ###
        "reset_delay_length": 300e-6,
        "readout_kappa": 2*np.pi*1e6,
        "readout_chi": 2*np.pi*0.93e6,

        "is_cr" : False,
        "cr" : {
            "control_qubit_frequency": 4.9341e9+0.039e6,
            "target_qubit_frequency_1": 4.7e9+0.12e6,
            "target_qubit_frequency_2": 4.7e9,  # temporally arbitrary value
        }

    },

    "q1": {

        #port 1 (15-1)

        "ge_frequency" : 3.9542e9 + 0.047e6,
        "ef_frequency" : 3.955e9, #4.275e9, #4.3865e9,
        "readout_frequency" : 7.6642e9, 
        ### readout parameters ###
        "readout_amp" : 0.18, #0.25, #0.25, #0.18,
        'readout_pulse_length': 1400e-9,
        "readout_integration_length": (1200)*1e-9,
        "readout_integration_amp": 1,

        "drachma_readout_pulse_length": 1400e-9,
        "drachma_readout_integration_length": (1200)*1e-9,
        "drachma_readout_amp": 0.6,

        "readout_phase": (0) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.446, 
        "drive_pulse_length": 64e-9,
        "drive_beta": 0.0,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.223,
        "pi2_beta": 0.0,

        "pi_length": 64e-9,
        "pi_amp": 0.446,
        "pi_beta": 0.0,

        "cond_pi_length": 2048e-9,
        "cond_pi_amp": 0.011,
        "cond_pi_beta": 0.0,

        "rabi_drive_amp": 0.334/2,
        "ramp_length": 120e-9,

        ### extra parameters ###
        "reset_delay_length": 300e-6,
        "readout_kappa": 2*np.pi*1e6,
        "readout_chi": 2*np.pi*0.93e6,

        "is_cr" : False,
        "cr" : {
            "control_qubit_frequency": 4.9341e9+0.039e6,
            "target_qubit_frequency_1": 4.7e9+0.12e6,
            "target_qubit_frequency_2": 4.7e9,  # temporally arbitrary value
        }

    },

    "q2": { #control out port 1 (8-2)
        ### frequency parameters ###
        "ge_frequency" : 4.4017e9, #4.275e9, #4.3865e9,
        "ef_frequency" : 4.4017e9, #4.275e9, #4.3865e9,
        "readout_frequency" : 7.6642e9,  
        ### readout parameters ###
        "readout_amp" : 0.25, #0.18,
        'readout_pulse_length': 1800e-9,
        "readout_integration_length": 1600e-9,
        "readout_integration_amp": 1,
        "readout_phase": (77) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.29, 
        "drive_pulse_length": 64e-9,
        # "drive_ef_amp" : 0.6,
        # "drive_ef_pulse_length": 64e-9,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.145,
        "pi2_beta": 0.03,

        "pi_length": 64e-9,
        "pi_amp": 0.286,
        "pi_beta": 0.03,

        "cond_pi_length": 2*512e-9,
        "cond_pi_amp": 0.06,
        "cond_pi_beta": 0.0,

        "rabi_drive_amp": 0.2,
        "ramp_length": 0e-9,
        ### extra parameters ###
        "reset_delay_length": 200e-6,

        "is_cr" : False,
        "cr" : {
            "control_qubit_frequency": 4.9341e9+0.039e6,
            "target_qubit_frequency_1": 4.7e9+0.12e6,
            "target_qubit_frequency_2": 4.7e9,  # temporally arbitrary value
        }

    },
}

cavity_parameters = {

    'm0': { # mm2 - qubit 4.394GHz
        "mode_frequency": 5.152209e9, #5.152209e9 - 0.1e6, # center of g and e freq
        "cavity_drive_length": 40e-9, # this should be as short as possible to cover wide frequency range
        "cavity_drive_amp": 0.057*6, #(this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_cavity_drive_amp": 0.057, # 1 when it is not found yet. (This is amp when photon = 1) (this should be same with scaling_factor in disp_amp_calibration_parity

        "cond_disp_pulse_length": 200e-9, #200e-9,
        "cond_disp_pulse_amp": 0.085, #0.0446j, # (this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_CNOD_amp": 0.085, # 1 when it is not found yet. (this should be same with scaling_factor in CNOD calibration)

        # (omega_d : w_LO + w_IF , omega_e or g : w_LO + w_IF + cond_disp_pulse_frequency)
        "cavity_mode_chi": -0.2e6, 
        "chi_for_cnod": -0.2e6, #-0.2e6/2, # it depends on the driving freq frame.
        "cond_disp_pulse_frequency": -0.2e6, 
        # driving only cavity mode of ground state in driving freq(mode_frequency) frame.
        "cond_disp_pulse_detuning": 0.5e6,
        "cond_disp_pulse_sigma": 2,
        "geophase_correction_coeff": 0.039/2, # after calibrating geometric phase.

        "sideband_length": 1e-6,
        "sideband_chunk_length": 1e-6,
        "sideband_rise_fall_length": 120e-9,

        "sideband_frequency_l" : 30e6, #30e6, # input unit is Hz
        "sideband_frequency_h" : 30e6, #30e6,
        "sideband_amp_l" : 0.1,
        "sideband_amp_h" : 0.1,
        "sideband_phase" : 0, # in radians
        "sideband_att_l" : 531, #448, # The larger value, the smaller attenuation
        "sideband_att_h" : 211, #239, # The larger value, the smaller attenuation
        "sideband_extra_phase": 0, # in radians

        "reset_delay_length": 500e-6,
    },

    'm1': { # mm1
        "mode_frequency": 6.795139e9 - 1.29e6, # center of g and e freq
        "cavity_drive_length": 40e-9, # this should be as short as possible to cover wide frequency range
        "cavity_drive_amp": 0.0564*5, #(this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_cavity_drive_amp": 0.0564, # 1 when it is not found yet. (This is amp when photon = 1) (this should be same with scaling_factor in disp_amp_calibration_parity

        "cond_disp_pulse_length": 280e-9, #200e-9,
        "cond_disp_pulse_amp": 0.8, #0.0446j, # (this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_CNOD_amp": 0.1045, # 1 when it is not found yet. (this should be same with scaling_factor in CNOD calibration)

        # (omega_d : w_LO + w_IF , omega_e or g : w_LO + w_IF + cond_disp_pulse_frequency)
        "cavity_mode_chi": -0.067e6, #-0.824e6,
        "cond_disp_pulse_frequency": -0.035e6, # driving only cavity mode of ground state in driving freq(mode_frequency) frame.
        "cond_disp_pulse_detuning": 0.5e6,
        "cond_disp_pulse_sigma": 2,

        "sideband_length": 1e-6,
        "sideband_chunk_length": 1e-6,
        "sideband_rise_fall_length": 120e-9,

        "sideband_frequency_l" : 10e6, # input unit is Hz
        "sideband_frequency_h" : 10e6,
        "sideband_amp_l" : 0.5,
        "sideband_amp_h" : 0.5,
        "sideband_phase" : 0, # in radians
        "sideband_att_l" : 194, # The larger value, the smaller attenuation
        "sideband_att_h" : 182, # The larger value, the smaller attenuation

        "reset_delay_length": 500e-6,
    },

    'm2': { 
        "mode_frequency": 5.151637e9 + 0.45e6, # center of g and e freq
        "cavity_drive_length": 40e-9, # this should be as short as possible to cover wide frequency range
        "cavity_drive_amp": 0.0362*3, #(this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_cavity_drive_amp": 0.0362, # 1 when it is not found yet. (This is amp when photon = 1) (this should be same with scaling_factor in disp_amp_calibration_parity

        "cond_disp_pulse_length": 160e-9, #200e-9,
        "cond_disp_pulse_amp": 0.083, #0.0446j, # (this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_CNOD_amp": 0.083, # 1 when it is not found yet. (this should be same with scaling_factor in CNOD calibration)

        # (omega_d : w_LO + w_IF , omega_e or g : w_LO + w_IF + cond_disp_pulse_frequency)
        "cavity_mode_chi": -0.22e6, #-0.824e6,
        "cond_disp_pulse_frequency": -0.11e6, # driving only cavity mode of ground state in driving freq(mode_frequency) frame.
        "cond_disp_pulse_detuning": 0.5e6,
        "cond_disp_pulse_sigma": 2,

        "sideband_length": 1e-6,
        "sideband_chunk_length": 1e-6,
        "sideband_rise_fall_length": 120e-9,

        "sideband_frequency_l" : 20e6, # input unit is Hz
        "sideband_frequency_h" : 20e6,
        "sideband_amp_l" : 0.1,
        "sideband_amp_h" : 0.1,
        "sideband_phase" : 0, # in radians
        "sideband_att_l" : 1105, # The larger value, the smaller attenuation
        "sideband_att_h" : 576, # The larger value, the smaller attenuation

        "reset_delay_length": 500e-6,
    },

}

# In[]

my_run = Bosonic_experiments(physical_ports, qubits_parameters, cavity_parameters, 
                 number_of_qubits=3,
                 number_of_memory_modes=3,
                 is_memory_mode = True, 
                 which_qubit=0,
                 which_mode =0,
                 which_data= "I", 
                 cr_drive_lines=False, 
                 multiplex_drive_lines = False,
                 is_cavity_trajectory_weighting_function = True,
                 use_emulation = False)

# %% propagation delay calibration for readout and drive line

my_run.prop_delay_calibration(line = "readout", average_exponent=12, readout_pulse_type = "const")

# %%
my_run.nopi_pi(average_exponent=12, phase = 75, readout_pulse_type="gaussian_square",
                            readout_weighting_type ="cavity_trajectory", is_plot_simulation=False)
## readout weighting is not important for above experiment ; result is obtained through RAW data
my_run.single_shot_nopi_pi(npts_exponent = 12, phase = 75, readout_pulse_type="gaussian_square",
                            readout_weighting_type ="cavity_trajectory", is_plot_simulation=False)
my_run.plot_nopi_pi(npts = 100)

# %%
my_run.nopi_pi(average_exponent=12, phase = 75, readout_pulse_type="drachma",
                            readout_weighting_type ="drachma", is_plot_simulation=False) 
## readout weighting is not important for above experiment ; result is obtained through RAW data
my_run.single_shot_nopi_pi(npts_exponent = 12, phase = 75, readout_pulse_type="drachma",
                            readout_weighting_type ="drachma", is_plot_simulation=False)
my_run.plot_nopi_pi(npts = 100)


#In[]

my_run.T1(average_exponent=11, duration = 60e-6, npts = 31, is_plot_simulation=False)
my_run.plot_T1()

# In[]

my_run.Pi2_cal(average_exponent=11, start = 0, npts = 12, is_plot_simulation=False)
my_run.plot_Pi2_cal()

# In[]

my_run.Pi_cal(average_exponent=11, npts = 12, is_plot_simulation=False, is_cond_pulse= False)
my_run.plot_Pi_cal()

# In[]
my_run.Ramsey(is_echo = True, n_pi_pulse=1, # for CP or CPMG
              qubit_phase = 0, # 0 CP, pi/2 CPMG
              detuning = 0.2e6, 
              average_exponent=10, duration = 30e-6, npts = 60,
              is_zz_interaction= False,
              control_qubit= 2,
              is_plot_simulation= False)
my_run.plot_Ramsey(is_fourier_transform=True)


# In[]
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.15
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.15

my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2
my_run.cavity_parameters['m0']['sideband_extra_phase'] = 0 * np.pi/180

my_run.Ramsey_with_photon(is_echo = True,
              detuning = 0.0e6,
              cavity_freq_detuning= 0e6, # memory mode weak driving (when sideband driving, it should be zero)
              average_exponent=9, duration = 5e-6, npts = 60, #duration/npts = integer
              steady_time = 5e-6,
              delay_after_disp_drive = 0e-6,
              qubit_extra_phase= 0,
              is_constant_drive= False,
              is_sideband_drive= True,
              is_displacement_drive = True, ##### cautious
              is_plot_simulation= False)

my_run.plot_Ramsey(is_ramsey_with_photon = True, is_fourier_transform=True)

# In[]
my_run.cavity_parameters['m0']['sideband_phase'] = 0

my_run.Ramsey_amp_sweep(is_echo = False, detuning = 0.2e6, 
                        cavity_freq_detuning= 30e6, # memory mode weak driving (when sideband driving, it should be zero)
              average_exponent=9, duration = 10e-6, npts = 50, #duration/npts = integer
              sweep_start = 0.0, sweep_stop = 0.3, sweep_npts = 11, # sweep_start = -np.pi/16, sweep_stop = np.pi/16, sweep_npts = 11,
              fixed_sideband_amp_l = 0, # if not zero, sideband_amp_l is fixed during amp sweep. only sideband_amp_h is swept.
              # in case is_sideband_phase_sweep = True, this parameter is used as fixed amplitude for amp_l and amp_h 
              kappa = 0.0025, chi = 0.2, # in MHz unit # chi = chi_ge/2
              steady_time = 5e-6,
              is_constant_drive= True,
              is_sideband_drive= False, # both sidebands (do not forget making cavity_freq_detuning zero)
              is_sideband_phase_sweep = False, # when it wants to be swept, is_sideband_drive should be True either
              is_fourier_transform = True,
              is_plot_figure=False,
              is_plot_simulation= False)


# In[]
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.15
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.15
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2 # sweep variable if is_amp_sweep = False
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.1 # sweep variable if is_amp_sweep = True

my_run.Ramsey_echo_sweep_with_SB(detuning = 0.2e6,
               average_exponent = 9, duration = 30e-6, npts = 60,
               sweep_start = 0, sweep_stop = np.pi, sweep_npts = 11, # sweep_start = -0.1, sweep_stop = 0.2, sweep_npts = 11,
               is_disp_amp_real = True, # if True, disp_amp is real number sweep. if False, imaginary number sweep.
               steady_time = 5e-6,
               delay_after_disp_drive = 0e-9,
               is_amp_sweep = False, # else sideband_phase sweep with const disp_amp
               is_plot_figure = False,
               is_plot_simulation = False)


# In[]
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.25
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.25
my_run.cavity_parameters['m0']['sideband_phase'] = np.pi/2

my_run.Ramsey_echo_disp_amp_2Dsweep_with_SB(detuning = 0.2e6,
               average_exponent = 9, duration = 30e-6, npts = 60,
               amp_x_start = 0, amp_x_stop = 0.2, amp_x_npts = 7,
               amp_y_start = -0.1, amp_y_stop = 0.1, amp_y_npts = 7,
               steady_time = 5e-6,
               delay_after_disp_drive = 0,
               is_plot_figure = False,
               is_plot_simulation = False)

# In[] long_time ramsey
my_run.long_time_Ramsey(time_end= 30, time_gap=0.5, average_exponent=10, detuning=0.2e6, duration=20e-6, npts=41)

# In[]
my_run.RamseyEcho_to_chi(detuning = 0.2e6,
               average_exponent = 8, duration = 20e-6, npts = 40,
               steady_time = 5e-6,
               wait_time = 0.5e-6,
               cavity_freq_start = -2e6, cavity_freq_stop = 2e6, cavity_freq_npts = 81,
               is_plot_simulation = False)
my_run.plot_RamseyEcho_to_chi()

# In[]
my_run.readout_chi_spectroscopy(average_exponent = 10, npts = 61, 
                                freq_start = -2e6,
                                freq_stop = 4e6,
                                is_plot_simulation = False)
my_run.plot_readout_chi_spectroscopy()


# %%
my_run.error_amplification(average_exponent= 10, pulse_npts = 12, amp_npts = 21, 
                           start_amp = -0.1, end_amp = 0.1,
                           is_drag_beta_calibration = True, #if True, beta will be swept
                           is_plot_simulation=False)
my_run.plot_error_amplification(is_drag_beta_calibration = True)
 # In[]
# my_run.Rabi_amplitude(average_exponent=12, npts = 101, is_plot_simulation=True)

# In[]
my_run.Rabi_length(average_exponent=10, duration = 0.5e-6, start_time = 1e-6, npts = 100,
                   detuning_freq = 0e6, # detuned freq for qubit drive
                   is_single_shot = False, is_plot_simulation=False)
my_run.plot_Rabi_length()

# In[]
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.1
my_run.cavity_parameters['m0']['sideband_phase'] = 0

my_run.Rabi_length_spin_locking(average_exponent = 9, duration = 0.5e-6, start_time = 1e-6,
                                detuning_freq = -0.156e6, # detuned freq for qubit drive
                                npts = 100, rabi_phase = np.pi/2, is_init_pi2=True, init_pi2_phase=0,
                                is_sideband_drive= True, steady_time = 5e-6, 
                                is_plot_simulation = False)
my_run.plot_Rabi_length_spin_locking()

# In[]
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.1
my_run.cavity_parameters['m0']['sideband_amp_l'] = 0.2
my_run.cavity_parameters['m0']['sideband_amp_h'] = 0.2
my_run.cavity_parameters['m0']['sideband_phase'] = 0

my_run.Rabi_length_with_photon(average_exponent=10, duration = 40e-6, start_time = 1e-6, npts = 800,
                   detuning_freq = -0.605e6, # detuned freq for qubit drive
                   steady_time = 5e-6, is_sideband_drive= True,
                   is_single_shot = False, is_plot_simulation=False)
my_run.plot_Rabi_length()

# In[] cr_calibration_amp
my_run.cr_calibration_amp(average_exponent=12, npts = 101, control_qubit = 0, target_qubit = 1, is_plot_simulation=False)
my_run.plot_cr_calibration_amp()

# In[] cr_calibration_length

my_run.cr_calibration_length(average_exponent=12, npts = 101, control_qubit = 0, target_qubit = 1,
                             duration = 1e-6, target_qubit_amp = 0.5,
                             is_plot_simulation=False)
my_run.plot_cr_calibration_length()


# In[]
my_run.All_XY(average_exponent=12, is_plot_simulation=False)
my_run.plot_All_XY()


# In[] drag_calibration
my_run.drag_calibration(average_exponent=11, beta_start = -0.2, beta_stop = 0.2, beta_count = 81, is_plot_simulation=False)
my_run.plot_drag_calibration()

# In[] cavity_T1 (crosskerr effect : it requires large photon number => it could have side-effect)

my_run.cavity_T1(average_exponent=10, start = 0e-6, duration = 300e-6, npts = 101, is_plot_simulation=False)
my_run.plot_cavity_T1(init_guess = [1.5, 10, 20000, 0.8])

# In[]

# In[] cavity_mode_spectroscopy

my_run.cavity_mode_spectroscopy(average_exponent=11, freq_start = 100e6,
                                freq_stop = 200e6, npts = 101, is_plot_simulation=False)

my_run.plot_cavity_mode_spectroscopy()

# In[] finding chi between cavity mode and qubit
my_run.cavity_parameters['m0']['cavity_drive_length'] = 40e-9
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.15

my_run.qubit_state_revival(average_exponent=11, wait_time= 12e-6, wait_npts = 121, is_plot_simulation=False)
my_run.plot_qubit_state_revival()


# In[]
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.05
my_run.cavity_parameters['m0']["cond_disp_pulse_length"] = 480e-9 #320e-9
my_run.cavity_parameters['m0']['cond_disp_pulse_detuning'] = 0.5e6
my_run.cavity_parameters['m0']['cond_disp_pulse_sigma'] = 2

my_run.cavity_pi_nopi(average_exponent=9, freq_start = -0.2e6, freq_stop = 0.3e6, 
                      freq_npts = 50, 
                      is_qubit2 = False, 
                      qubit2 = 1,
                      auto_chunking=True,
                      is_plot_simulation=False)
# not accurate for finding chi with cavity_pi_nopi experiment

my_run.plot_cavity_pi_nopi()


# In[]
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.3 # sweep variable
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.058

my_run.CNOD_calibration(average_exponent=10, amp_range=1j,  npts= 61, qubit_phase = 0, is_calibrated_geo_phase = False,
                        is_displaced_state= False, is_plot_simulation=False)
my_run.plot_CNOD_calibration(scaling_factor= 0.085)

# In[]
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.085*5

# CNOD 방향 dependance 잇는지 체크

my_run.CNOD_geophase_calibration(average_exponent=11, amp_start= 0.0, amp_stop = 1, npts = 21, 
                                 is_calibrated_geo_phase = False,
                                 is_plot_simulation=False)
my_run.plot_CNOD_geophase_calibration(is_normalize=True)

# In[]

# my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.2 # sweep variable
# my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.046*2

# my_run.Characteristic_function_2D(average_exponent=10, npts=61, qubit_phase = 90, is_plot_simulation=False)
# my_run.plot_Characteristic_function_2D()

# In[] before doing it, "alpha_1_CNOD_amp" needs to be calibrated. 
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.15 # sweep variable
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 1 # 이게 1이어야 함.. 코드 내부에서 amp 자리에 CNOD_alpha_1_amp가 들어가도록 되어 있음.
# 이제까지 중복해서 곱해지는 현상 떄문에 원했던 amp보다 덜 들어가는 상황이었음.
my_run.cavity_parameters['m0']['alpha_1_CNOD_amp'] = 0.085

my_run.disp_pulse_calibration_geophase(average_exponent=9, amp_sweep = 1, amp_npts=51, is_plot_simulation=False)
my_run.plot_disp_pulse_calibration_geophase()

# In[] Not proper in small chi regime
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.2 # sweep variable

my_run.disp_pulse_calibration_parity(average_exponent=10, amp_start = -1, amp_stop=1, amp_npts = 101, is_plot_simulation=False)
my_run.plot_disp_pulse_calibration_parity(is_fit=True, scaling_factor=0.0458) # scaling factor -> alpha_1_cavity_drive_amp 
                                                        # sigma should be 0.5 by adjusting scaling factor

# %%
my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.9 # sweep variable
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.0458 # assigned constant from above experiment

my_run.out_and_back_measurement(average_exponent=8, init_state = "e",
                                 phase_start= 180, phase_stop=250, # deg
                                 cavity_drive_amp_start = 0, cavity_drive_amp_stop = 1, # final amp will be multiplying it by "cavity_drive_amp"
                                 phase_npts=30, amp_npts=51,
                                 wait_time=1e-6,
                                 is_plot_simulation=False)

my_run.plot_out_and_back_measurement(fitting=False, 
                                     x_threshold=30, 
                                     y_threshold=False, 
                                     wait_time=1e-6, init_state="e")


 # In[]
my_run.cavity_parameters['m0']['cavity_drive_amp'] = my_run.cavity_parameters['m0']["alpha_1_cavity_drive_amp"]*2
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = my_run.cavity_parameters['m0']['alpha_1_CNOD_amp'] * 0.3j

my_run.storage_mode_characterization(average_exponent=10, 
                                    wait_time = 20e-6,
                                    wait_npts = 81,
                                    detuning = 0e6,#-0.0e6,
                                    init_state = "g",
                                    is_plot_simulation=False)
# init_guess = [amplitude,omega,T1,freq=2*detuning,offset]
my_run.plot_storage_mode_characterization(is_fit=True,init_guess=[-0.69, 5.279, 60e-6, 0.0e6, -0.024], is_fourier_transform=True)
# In[]
my_run.plot_storage_mode_characterization(is_fit=True,init_guess=[-2, 2.5, 10e-6, 0.0e6, 0.5])


# In[] with post selection (singleshot measure를 해야해서 measure point 수 한계가 있음)

my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.057*4 # sweep variable for wigner
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.057
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.085 * 8
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.085

# 'cavity_drive_amp' * amplitude에 해당하는 displaced state 형성
# 'cavity_drive_amp'/(2*'alpha_1_cavity_drive_amp') 사이즈에 해당하는 Re-Im plane plot in case of wigner

my_run.wigner_characteristic_function_2D(average_exponent=10, npts_x = 21, npts_y = 21, amplitude = 0.5, #amplitude for coherent state
                          is_wigner_function= False,
                          is_coherent_state = False, is_schrodinger_cat_state= False, 
                          is_schrodinger_cat_state_2=False,
                          is_cat_state=False,
                          is_cat_state_2=True,
                          is_cat_and_back=False,
                          acquire_delay= 40e-9,
                          alpha= 0.5, beta= -0.05,
                          chunk_count = 21,
                          is_plot_simulation=False)
x_grid, y_grid, G_data, E_data = my_run.plot_wigner_characteristic_function_measurement_2D(is_normalize = True, 
                                                                                           is_plot_G=True, is_plot_E=True,
                                                                                           is_wo_post_selection=False)

# In[]
psi_init = qt.coherent(N = 20, alpha = 1.5) + qt.coherent(N = 20, alpha = -1.5)
psi_init = psi_init.unit()
rho_init = qt.ket2dm(psi_init)

rho = my_run.density_matrix_reconstruction(N = 20, rho_init = rho_init, data = G_data, 
                                           x_grid = x_grid, y_grid = y_grid,
                                           maxiter=2000, maxfun=2000000)
my_run.state_fidelity(rho_ideal = rho_init, rho = rho)

my_run.plot_characteristic_function(x_grid, y_grid, rho)
my_run.plot_characteristic_function(x_grid, y_grid, rho_init)


# In[] without post selection

my_run.cavity_parameters['m0']['cavity_drive_amp'] = 0.057*4 # sweep variable for wigner
my_run.cavity_parameters['m0']['alpha_1_cavity_drive_amp'] = 0.057
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.085 * 6
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.085

# 'cavity_drive_amp' * amplitude에 해당하는 displaced state 형성
# 'cavity_drive_amp'/(2*'alpha_1_cavity_drive_amp') 사이즈에 해당하는 Re-Im plane plot in case of wigner

my_run._wigner_characteristic_function_2D(average_exponent=10, npts_x = 31, npts_y = 31, amplitude = 0.5, #amplitude for coherent state
                          is_wigner_function=False,
                          is_coherent_state = True, is_schrodinger_cat_state= False, 
                          is_schrodinger_cat_state_2=False,
                          is_cat_state=False,
                          alpha=0.5, beta= -0.05,
                          is_autochunking=True,
                          chunk_count=1,
                          delay_after_displacement= 0e-6,
                          is_plot_simulation=False)
x_grid, y_grid, data = my_run._plot_wigner_characteristic_function_measurement_2D(is_normalize=True)

# In[]
psi_init = qt.coherent(N = 20, alpha = 2)
rho_init = qt.ket2dm(psi_init)

rho = my_run.density_matrix_reconstruction(N = 20, rho_init = rho_init, data = data, 
                                           x_grid = x_grid, y_grid = y_grid,
                                           maxiter=2000, maxfun=2000000)
my_run.state_fidelity(rho_ideal = rho_init, rho = rho)

my_run.plot_characteristic_function(x_grid, y_grid, rho)
my_run.plot_characteristic_function(x_grid, y_grid, rho_init)
# In[]
my_run.cavity_parameters['m0']["alpha_1_CNOD_amp"] = 0.077
my_run.cavity_parameters['m0']['cond_disp_pulse_amp'] = 0.077 * 4

my_run.disentangling_power_sweep(average_exponent = 10,
                                  alpha = 1,
                                  beta_sweep_start = -0.5,
                                  beta_sweep_stop = 0.5,
                                  beta_sweep_count = 21,
                                  is_xyz=True)
my_run.plot_disentangling_power_sweep(is_normalize=True)



# In[]

my_run.cavity_parameters['m0']["sideband_chunk_length"] = 1e-6
my_run.cavity_parameters['m0']["sideband_rise_fall_length"] = 1e-6
my_run.cavity_parameters['m0']["sideband_frequency_l"] = 10e6
my_run.cavity_parameters['m0']["sideband_frequency_h"] = 10e6
my_run.cavity_parameters['m0']["sideband_amp_l"] = 0.7
my_run.cavity_parameters['m0']["sideband_amp_h"] = 0.7
my_run.cavity_parameters['m0']["sideband_phase"] = 0
my_run.cavity_parameters['m0']["sideband_att_l"] = 194
my_run.cavity_parameters['m0']["sideband_att_h"] = 182

my_run.qubits_parameters['q0']["rabi_drive_amp"] = 0.2192


# rabi_pulse_length:200ns 배수
# sidebands_pulse_length:1us 배수로 설정

# m = np.linspace(0.21, 0.22, 11)
# for i in m :
#     my_run.qubits_parameters['q0']["rabi_drive_amp"] = i

my_run.calibrate_sideband_pulse_phase(average_exponent=10, sidebands_pulse_length= 10e-6, rabi_pulse_length= 3e-6,
                                    qubit_drive_detuning_freq= -1.044e6,
                                    npts_phase_sweep= 81, is_sideband_phase_sweep = True, # else rabi phase sweep
                                    phase_sweep_stop = 2*np.pi,
                                    rabi_phase = 0,
                                    auto_chunking = True,
                                    is_init_qubit_pi2=False,
                                    is_plot_simulation = True)

my_run.plot_calibrate_sideband_pulse_phase(is_normalize=True, is_fig_plot=True)


















# In[]

my_run.continuous_wave(average_exponent=19, freq_l=5e6, freq_h=10e6, amp_l=0.5, amp_h=0.1, phase = np.pi/2,
                       amp_cont = 0.1,
                       is_sideband_pulse = True, is_plot_simulation=False)






# In[]
## define pulse
my_pulse = pulse_library.cond_disp_pulse(uid="my_pulse", length=100e-9, 
                                         amplitude=1.0, sigma = 1,
                                         chi = 1e6, detuning = 0.001e6,
                                         zero_boundaries=False)


_pulse = pulse_library.gaussian(
                uid="cavity_drive_pulse",
                length= 2048e-9, #cavity_parameters["m1"]["cavity_drive_length"],
                amplitude=cavity_parameters["m1"]["cavity_drive_amp"],
                zero_boundaries=False,
                sigma = 1/3,
            )

_pulse = pulse_library.drachma_readout_pulse(
                uid="drachma_readout_pulse",
                length = qubits_parameters["q0"]["readout_pulse_length"],
                amplitude=qubits_parameters["q0"]["readout_amp"],
                kappa=qubits_parameters["q0"]["readout_kappa"],
                chi_list=[-qubits_parameters["q0"]["readout_chi"]/2, qubits_parameters["q0"]["readout_chi"]/2],
                zeta_list = [-2*np.pi*1000,-2*np.pi*1000],
            )

_pulse = pulse_library.sidebands_pulse(
                uid="sidebands_drive_pulse",
                length=cavity_parameters["m0"]["sideband_chunk_length"],
                frequency_l=cavity_parameters["m0"]["sideband_frequency_l"],
                frequency_h=cavity_parameters["m0"]["sideband_frequency_h"],
                amp_l=0.1,
                amp_h=0.2,
                phase=np.pi/2,
            )


pulse = pulse_update(
            _pulse,
            spectral_window = None,
            flip_angle = None,
            pulse_parameters=_pulse.pulse_parameters,
        )

my_run.analyze_pulse(_pulse, high_res = False)




# %%
