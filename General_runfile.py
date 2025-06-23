# In[]

import matplotlib.pyplot as plt
%matplotlib qt5
import numpy as np
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

from General_functions import *

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

physical_ports = {
    "measure":{
        "freq_LO": 7.2e9,   # f = f_LO + f_IF (positive sideband)
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
        "freq_LO": 7.2e9,   # f = f_LO + f_IF (positive sideband) (should be same with measure line)
        "port_delay": 76e-9, # 72e-9, (8-2) 
        "port_mode": None,
        "delay_signal": None, # Global delay for the signal line, implemented by adjusting the waveform.
        "threshold": None,
        "added_outputs": None, # Added outputs to the signal line's physical channel port. (RTR option enabled)
        "automute": False, # Mute output channel when no waveform is played on it i.e for the duration of delays.
        "range": -10,
        "amplitude": 1.0, # it does not work!
                          # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
    },
    "drive":{
        "q0":{ # 1st control line port # 현재 안 쓰는 포트
            "freq_LO": 4.2e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
        },
        "q1":{ # 2nd control line port
            "freq_LO": 6e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
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
        },
        "m0":{ # mapped to port4 # 현재 안 쓰는 포트
            "freq_LO": 5e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
        },
        "m1":{ # mapped to port5
            "freq_LO": 5e9,
            "port_delay": 0e-9, # Not currently supported on SHFSG output channels.
            "port_mode": None,
            "delay_signal": None, 
            "threshold": None,
            "added_outputs": None, 
            "automute": True, 
            "range": 10,
            "amplitude": 1, # Only supported by the SHFQA. Amplitude multiplying all waveforms played on the signal line.
        }
    }
}

### These parameters will be utilized in defining experimental sequences.
qubits_parameters = {
    # two(cr) qubit gate : each parameter is assigned to each qubit except ef_frequency;

    "q0": { #control out port 1 # 현재 안 쓰는 포트
        ### frequency parameters ###
        "ge_frequency" : 4.3997e9-0.049e6,
        "ef_frequency" : 4.3997e9-0.049e6,
        "readout_frequency" : 7.4097e9, 
        ### readout parameters ###
        "readout_amp" : 0.17, #0.18,
        'readout_pulse_length': 1500e-9,
        "readout_integration_length": 1400e-9,
        "readout_integration_amp": 1,
        "readout_phase": (135) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.584, 
        "drive_pulse_length": 64e-9,
        # "drive_ef_amp" : 0.6,
        # "drive_ef_pulse_length": 64e-9,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.29,
        "pi2_beta": 0.025,

        "pi_length": 64e-9,
        "pi_amp": 0.582,
        "pi_beta": 0.025,

        "cond_pi_length": 64e-9*6,# 64e-9*8, (8-2)
        "cond_pi_amp": 0.071, #0.0715, (8-2)
        "cond_pi_beta": 0.0,

        "rabi_drive_amp": 0.01,
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
    "q1": { #control out port 2 (7-1)
        ### frequency parameters ###
        "ge_frequency" : 6.284358e9, 
        "ef_frequency" : 6.284358e9,
        "readout_frequency" : 7.50824e9,
        ### readout parameters ###
        "readout_amp" : 0.17, #0.18,
        'readout_pulse_length': 1500e-9,
        "readout_integration_length": 1400e-9,
        "readout_integration_amp": 1,
        "readout_phase": (235) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.426, 
        "drive_pulse_length": 64e-9,
        # "drive_ef_amp" : 0.6,
        # "drive_ef_pulse_length": 64e-9,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.198, 
        "pi2_beta": 0.025,

        "pi_length": 64e-9,
        "pi_amp": 0.396, 
        "pi_beta": 0.025,

        "cond_pi_length": 64e-9*6,
        "cond_pi_amp": 0.0675, 
        "cond_pi_beta": 0.0,

        "rabi_drive_amp": 0.01,
        "ramp_length": 0e-9,
        ### extra parameters ###
        "reset_delay_length": 200e-6,

        "is_cr" : False,
        "cr" : {
            "control_qubit_frequency": 4.9341e9,
            "target_qubit_frequency_1": 4.7e9+0.12e6,
            "target_qubit_frequency_2": 4.7e9,
        }
    },

    "q2": { #control out port 1 (8-2)
        ### frequency parameters ###
        "ge_frequency" : 4.3997e9-0.049e6-0.134e6,
        "ef_frequency" : 4.3997e9-0.049e6,
        "readout_frequency" : 7.4097e9, 
        ### readout parameters ###
        "readout_amp" : 0.17, #0.18,
        'readout_pulse_length': 1500e-9,
        "readout_integration_length": 1400e-9,
        "readout_integration_amp": 1,
        "readout_phase": (195) * np.pi/180,
        ### drive parameters ###
        "drive_amp" : 0.584, 
        "drive_pulse_length": 64e-9,
        # "drive_ef_amp" : 0.6,
        # "drive_ef_pulse_length": 64e-9,
        
        "pi2_length": 64e-9,
        "pi2_amp": 0.299,
        "pi2_beta": 0.029,

        "pi_length": 64e-9,
        "pi_amp": 0.6,
        "pi_beta": 0.029,

        "cond_pi_length": 768e-9,
        "cond_pi_amp": 0.049,
        "cond_pi_beta": 0.0,

        "rabi_drive_amp": 0.01,
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
    'm0': { # port 4 # 현재 안 쓰는 포트
        "mode_frequency": 5.128112e9, # omega_d
        "cavity_drive_length": 120e-9,
        "cavity_drive_amp": 0.1, #0.02,
        "reset_delay_length": 700e-6,

        "cond_disp_pulse_length": 200e-9, #200e-9,
        "cond_disp_pulse_amp": 0.9, #0.5,
        "alpha_1_CNOD_amp": 0.1104, # 1 when it is not found yet.

        "cavity_mode_chi": -0.81e6, # driving cavity mode of ground state in ground state frame.
        "cond_disp_pulse_frequency": 0.05e6, #0.08e6,
        "cond_disp_pulse_detuning": 0.05e6,
        "cond_disp_pulse_sigma": 2,
    },
    'm1': { # port 5
        # 5.128217e9 : ground state freq
        "mode_frequency": 5.128217e9, # omega_d
        "cavity_drive_length": 40e-9,
        "cavity_drive_amp": 0.0166*2*2,#0.03, (this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_cavity_drive_amp": 0.0166*2, # 1 when it is not found yet. (This is amp when photon = 1) (this should be same with scaling_factor in disp_amp_calibration_parity
        "reset_delay_length": 640e-6,

        "cond_disp_pulse_length": 200e-9, #200e-9,
        "cond_disp_pulse_amp": 0.8, #1j*0.1193/2, #1j*0.1147/3, (this is the maximum amplitude range, and this goes to amp of experiment)
        "alpha_1_CNOD_amp": 0.1151, # 1 when it is not found yet. (this should be same with scaling_factor in CNOD calibration)

        # (omega_d : w_LO + w_IF , omega_e or g : w_LO + w_IF + cond_disp_pulse_frequency)
        "cavity_mode_chi": -0.947e6, #-0.824e6,
        "cond_disp_pulse_frequency": -0.824e6, # driving only cavity mode of ground state in driving freq(mode_frequency) frame.
        "cond_disp_pulse_detuning": 0.05e6, 
        "cond_disp_pulse_sigma": 2, #4,
    }

}

# In[]
my_run = ZI_QCCS(physical_ports, qubits_parameters, cavity_parameters, 
                 number_of_qubits=3,
                 number_of_memory_modes=2,
                 is_memory_mode = True, 
                 which_qubit=2,
                 which_mode =1,
                 which_data= "I", 
                 cr_drive_lines=False, 
                 multiplex_drive_lines = False, 
                 use_emulation = False)

# %% propagation delay calibration for readout and drive line

my_run.prop_delay_calibration(line = "readout", average_exponent=12)


# %%
my_run.nopi_pi(average_exponent=12, phase = 195, is_plot_simulation=False) # used for 2^n averages, n=average_exponent, maximum: n = 19
my_run.single_shot_nopi_pi(npts_exponent = 12, phase = 195)
my_run.plot_nopi_pi(npts = 100)

#In[]

my_run.T1(average_exponent=11, duration = 30e-6, npts = 61, is_plot_simulation=False)
my_run.plot_T1()

# In[]

my_run.Pi2_cal(average_exponent=12, start = 0, npts = 12, is_plot_simulation=False)
my_run.plot_Pi2_cal()

# In[]

my_run.Pi_cal(average_exponent=12, npts = 18, is_plot_simulation=False, is_cond_pulse=False)
my_run.plot_Pi_cal()

# In[]
my_run.Ramsey(is_echo = False, detuning = 0.5e6, average_exponent=11, duration = 5e-6, npts = 41,
              is_zz_interaction= False,
              control_qubit= 1,
              is_plot_simulation= False)
my_run.plot_Ramsey()


# %%
my_run.error_amplification(average_exponent= 10, pulse_npts = 18, amp_npts = 41, 
                           start_amp = -0.05, end_amp = 0.1,
                           is_drag_beta_calibration = True, #if True, beta will be swept
                           is_plot_simulation=False)
my_run.plot_error_amplification(is_drag_beta_calibration = True)
 # In[]
# my_run.Rabi_amplitude(average_exponent=12, npts = 101, is_plot_simulation=True)

# In[]
my_run.Rabi_length(average_exponent=11, duration = 20e-6, npts = 101, is_plot_simulation=False)
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
my_run.drag_calibration(average_exponent=15, beta_start = -0.2, beta_stop = 0.2, beta_count = 81, is_plot_simulation=False)
my_run.plot_drag_calibration()

# In[] cavity_T1

my_run.cavity_T1(average_exponent=10, duration = 300e-6, npts = 51, is_plot_simulation=False)
my_run.plot_cavity_T1()

# In[] cavity_mode_spectroscopy

my_run.cavity_mode_spectroscopy(average_exponent=12, freq_start = 110e6,
                                freq_stop = 150e6, npts = 121, is_plot_simulation=False)

my_run.plot_cavity_mode_spectroscopy()

# In[]

my_run.cavity_pi_nopi(average_exponent=9, freq_start = -0.25e6, freq_stop = 1.25e6, 
                      freq_npts = 81, 
                      is_qubit2 = False, 
                      qubit2 = 1,
                      is_plot_simulation=False)
# not accurate for finding chi with cavity_pi_nopi experiment

my_run.plot_cavity_pi_nopi()


# In[]

my_run.CNOD_calibration(average_exponent=11, amp_range=0.5j, npts=201, qubit_phase = 0, is_displaced_state= True, is_plot_simulation=False)
my_run.plot_CNOD_calibration(scaling_factor=0.1151)

# In[]

my_run.Char_func_displaced(average_exponent=3, npts=21, qubit_phase =0, is_plot_simulation=False)
my_run.plot_Char_func_displaced()

# In[]

my_run.disp_pulse_calibration_geophase(average_exponent=11, amp_sweep = 1, amp_npts=51, is_plot_simulation=False)
my_run.plot_disp_pulse_calibration_geophase()

# In[]

my_run.disp_pulse_calibration_parity(average_exponent=11, amp_start = -1, amp_stop=1, amp_npts = 101, is_plot_simulation=False)
my_run.plot_disp_pulse_calibration_parity(is_fit=True, scaling_factor=0.0166*2) # scaling factor -> alpha_1_cavity_drive_amp 
                                                                        # sigma should be 0.5 by adjusting scaling factor

# %%

my_run.out_and_back_measurement(average_exponent=9, init_state = "g",
                                 phase_start= -30, phase_stop=30, # deg
                                 cavity_drive_amp_start = 0, cavity_drive_amp_stop = 1, # final amp will be multiplying it by "cavity_drive_amp"
                                 phase_npts=21, amp_npts=41,
                                 wait_time=1e-6,
                                 is_plot_simulation=False)

my_run.plot_out_and_back_measurement(Disp_amp_exp=0.033, Photon_number_exp= 1, 
                                    fitting=True, x_threshold=10, wait_time=1e-6, init_state="g")

# In[] finding chi between cavity mode and qubit

my_run.qubit_state_revival(average_exponent=11, wait_time=2.2e-6, wait_npts = 101, is_plot_simulation=False)
my_run.plot_qubit_state_revival()

# In[]

my_run.storage_mode_characterization(average_exponent=10, 
                                    wait_time = 4e-6,
                                    wait_npts = 101,
                                    detuning = 0.0e6,
                                    init_state = "e",
                                    is_plot_simulation=False)
# init_guess = [amplitude,omega,T1,freq=2*detuning,offset]
my_run.plot_storage_mode_characterization(is_fit=True,init_guess=[0.1, 2, 25e-6, 0.0e6, 0.0])





# In[]
## define pulse
my_pulse = pulse_library.cond_disp_pulse(uid="my_pulse", length=100e-9, 
                                         amplitude=1.0, sigma = 1,
                                         chi = 1e6, detuning = 0.001e6,
                                         zero_boundaries=False)

cavity_drive_pulse = pulse_library.gaussian(
                uid="cavity_drive_pulse",
                length=cavity_parameters["m1"]["cavity_drive_length"],
                amplitude=cavity_parameters["m1"]["cavity_drive_amp"]
            )

pulse = pulse_update(
            cavity_drive_pulse,
            spectral_window = None,
            flip_angle = None,
            pulse_parameters=cavity_drive_pulse.pulse_parameters,
        )

my_run.analyze_pulse(cavity_drive_pulse)




# %%
