# In[]

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time as time_module
import scipy.optimize

import sys
sys.path.append("D:/Software/SHFQC/")
from smart_fit import sFit

from General_functions import ZI_QCCS

from Basic_qubit_characterization import Basic_qubit_characterization_experiments

# Helpers:
from laboneq.contrib.example_helpers.generate_device_setup import (
    generate_device_setup_qubits,
)

# LabOne Q:
from laboneq.simple import *
from laboneq.dsl.experiment.builtins import *
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_results
from laboneq.contrib.example_helpers.randomized_benchmarking_helper import (
    clifford_parametrized,
    generate_play_rb_pulses,
    make_pauli_gate_map,
)
from laboneq_applications.analysis.fitting_helpers import exponential_decay_fit
from laboneq.contrib.bloch_simulator_pulse_plotter.inspector.update_inspect import (
    pulse_update,
)

# In[]

class Bosonic_experiments(Basic_qubit_characterization_experiments):

# In[] Bosonic experiments helpers

    def CNOD(self, exp, cond_disp_pulse, pi_pulse, amp, prev_uid, uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid = "pi_pulse_1"):
        
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        fix_angle = cavity_parameters[cavity_component]["cavity_mode_chi"] * (qubits_parameters[qubits_component]["pi_length"]/2 + cavity_parameters[cavity_component]["cond_disp_pulse_length"]) # in radians

        with exp.section(uid=uid1, play_after=prev_uid):
            exp.play(signal="cavity_drive", pulse=cond_disp_pulse, amplitude=amp)

        with exp.section(uid=pi_pulse_uid, play_after=uid1):
            exp.play(signal="drive", pulse=pi_pulse)

        with exp.section(uid=uid2, play_after=pi_pulse_uid):
            exp.play(signal="cavity_drive", pulse=cond_disp_pulse, 
                     amplitude=amp, phase = np.pi, increment_oscillator_phase = 2*np.pi*fix_angle)

    def wigner_function(self, exp, cavity_drive_pulse, pi2_pulse, amplitude_sweep, sweep_case, prev_uid=None):

        def correction(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)
    
            return decorator

        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        with exp.section(uid="alpha_sweep", play_after=prev_uid):
            # D^+(alpha)
            exp.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude_sweep, phase=np.pi)

        with exp.section(uid="qubit_excitation_1", play_after="alpha_sweep"):
            exp.play(signal="drive", pulse=pi2_pulse)
            exp.delay(signal="drive", time=np.abs(1/(2*cavity_parameters[cavity_component]["cavity_mode_chi"]))) # delay for cross Kerr effect

        with exp.section(uid="qubit_excitation_2", play_after="qubit_excitation_1"):
            @correction(sweep_case, exp=exp)
            def play_correction(v):
                if v == 0:
                    exp.play(signal="drive", pulse=pi2_pulse, phase = 0)
                elif v == 1:
                    exp.play(signal="drive", pulse=pi2_pulse, phase = np.pi)

    def characteristic_function(self, exp, pi2_pulse, pi_pulse, cond_disp_pulse, amplitude_sweep, qubit_phase=0, prev_uid=None):

        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        with exp.section(uid="qubit_excitation_1", play_after = prev_uid):
            exp.play(signal="drive", pulse=pi2_pulse)

        self.CNOD(exp=exp, cond_disp_pulse=cond_disp_pulse, 
                pi_pulse = pi_pulse, amp = amplitude_sweep, prev_uid="qubit_excitation_1", 
                uid1 = "char_cond_disp_pulse_1", uid2 = "char_cond_disp_pulse_2", pi_pulse_uid = "char_pi_pulse_1")
        
        with exp.section(uid="qubit_excitation_2", play_after="char_cond_disp_pulse_2"):
            exp.play(signal="drive", pulse=pi2_pulse, phase=qubit_phase)


# %% cavity crosskerr effect
    def cavity_T1(self, average_exponent=12, start = 1e-6, duration=100e-6, npts=101, is_plot_simulation=False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters

        component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        cond_disp_pulse, cavity_drive_pulse, _, _ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[component]["readout_phase"]

        # Sweep parameters
        delay_sweep = LinearSweepParameter(uid="delay", start=start, stop=start+duration, count=npts)

        sweep_case_2 = LinearSweepParameter(uid="crosskerr_check", start=0, stop=1, count=2)

        def on_off_cond_pi_pulse(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)

            return decorator

        # Define the experiment
        exp_cavity_T1 = Experiment(
            uid="cavity_T1",
            signals=[
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_cavity_T1.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_cavity_T1.sweep(uid="crosskerr_check", parameter=sweep_case_2):
                with exp_cavity_T1.sweep(uid="sweep", parameter=delay_sweep, alignment=SectionAlignment.RIGHT):
                    with exp_cavity_T1.section(uid="cavity_excitation", alignment=SectionAlignment.RIGHT):
                        exp_cavity_T1.play(signal="cavity_drive", pulse=cavity_drive_pulse)
                        exp_cavity_T1.delay(signal="cavity_drive", time=delay_sweep)

                    with exp_cavity_T1.section(uid="cond_pi_pulse", play_after="cavity_excitation"):
                        @on_off_cond_pi_pulse(sweep_case_2, exp=exp_cavity_T1)
                        def play_crosskerr_check(v):
                            if v == 0:
                                pass
                            elif v == 1:
                                exp_cavity_T1.play(signal="drive", pulse=cond_pi_pulse)

                    with exp_cavity_T1.section(uid="measure", play_after="cond_pi_pulse"):
                        exp_cavity_T1.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_cavity_T1.acquire(
                            signal="acquire",
                            handle="cavity_T1",
                            kernel=readout_weighting_function,
                        )
                    with exp_cavity_T1.section(uid="relax", 
                                            length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_cavity_T1.reserve(signal="measure")
        
        signal_map = {
            "measure": device_setup.logical_signal_groups[component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_cavity_T1.set_signal_map(signal_map)

        compiled_experiment_cavity_T1 = self.session.compile(exp_cavity_T1)

        self.cavity_T1_results = self.session.run(compiled_experiment_cavity_T1)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_cavity_T1, start_time=0, length=20e-6)

    def plot_cavity_T1(self, init_guess = None, is_fit = True):

        ### data processing ###############################################################

        averaged_nums = len(self.cavity_T1_results.acquired_results['cavity_T1'].axis[0])
        # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

        self.cavity_T1_data = self.cavity_T1_results.get_data("cavity_T1") # (2^N, npts) array
        # time = self.cavity_T1_results.acquired_results['cavity_T1'].axis[1]
        time = self.cavity_T1_results.acquired_results['cavity_T1'].axis[2]

        if self.which_data == "I":
            
            data_0 = np.real(np.mean(self.cavity_T1_data, axis = 0)[0]) # v ==0
            data_1 = np.real(np.mean(self.cavity_T1_data, axis = 0)[1]) # v ==1
            std_data_0 = np.real(np.std(self.cavity_T1_data, axis = 0)[0]/np.sqrt(averaged_nums))
            std_data_1 = np.real(np.std(self.cavity_T1_data, axis = 0)[1]/np.sqrt(averaged_nums))

            # data = np.real(np.mean(self.cavity_T1_data, axis = 0))
            # std_data = np.real(np.std(self.cavity_T1_data, axis = 0)/np.sqrt(averaged_nums))
           
        else:
            data_0 = np.imag(np.mean(self.cavity_T1_data, axis = 0)[0]) # v ==0
            data_1 = np.imag(np.mean(self.cavity_T1_data, axis = 0)[1]) # v ==1
            std_data_0 = np.imag(np.std(self.cavity_T1_data, axis = 0)[0]/np.sqrt(averaged_nums))
            std_data_1 = np.imag(np.std(self.cavity_T1_data, axis = 0)[1]/np.sqrt(averaged_nums))

            # data = np.imag(np.mean(self.cavity_T1_data, axis = 0))
            # std_data = np.imag(np.std(self.cavity_T1_data, axis = 0)/np.sqrt(averaged_nums))


        ### data plot ######################################################################

        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 20))

        ax1[0].errorbar(time*1e6, data_0, yerr = std_data_0, fmt = '--or', capsize = 5, markersize = 3, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
        ax1[1].errorbar(time*1e6, data_1, yerr = std_data_1, fmt = '--ob', capsize = 5, markersize = 3, ecolor = 'k', mfc=(0,0,1,0.5), mec = (0,0,0,1))

        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
        # ax2.errorbar(time*1e6, data, yerr = std_data, fmt = '--ok', capsize = 5, markersize = 3, ecolor = 'k', mfc=(0,0,0,0.5), mec = (0,0,0,1))
        ax2.errorbar(time*1e6, data_1-data_0, yerr = std_data_0+std_data_1, fmt = '--ok', capsize = 5, markersize = 3, ecolor = 'k', mfc=(0,0,0,0.5), mec = (0,0,0,1))
        
        if is_fit :
            sfit1 = sFit('ExpExp', time, data_1-data_0, init_guess=init_guess)
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            _,alpha_size, decay_rate,_ = popt
            _,alpha_size_err, decay_rate_err,_ = np.sqrt(np.diag(pcov))

            print(f"popt: {popt}")

            ax2.plot(time*1e6, sfit1.func(time, *popt))
            an = ax2.annotate((f'T1 = {(1/decay_rate*1e6):.2f}±{(1/(decay_rate)**2*decay_rate_err*1e6):.2f}[us]\n'
                               r'$\frac{\kappa}{2\pi}$' + f' = {decay_rate/2/np.pi*1e-3:.2f}±{decay_rate_err/2/np.pi*1e-3:.3f}[kHz]\n'
                               f'alpha = {alpha_size:.2f}±{alpha_size_err:.2f}'), 
                             xy = (np.average(time*1e6), np.average((data_1-data_0)[0:10]) ),
                             size = 16)
            an.draggable()

        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='minor', labelsize=15)

        ax2.set_title("cavity T1 measurement", fontsize=20)
        ax2.set_xlabel("Time (us)", fontsize=20)
        ax2.set_ylabel(f"{self.which_data} (a.u.)", fontsize=20)

        plt.show()

# In[]
    def cavity_mode_spectroscopy(self,
                                freq_start,
                                freq_stop,
                                npts,
                                average_exponent=12, 
                                is_plot_simulation=False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters

        component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse",
            length=qubits_parameters[component]["readout_pulse_length"],
            amplitude=qubits_parameters[component]["readout_amp"]
        )

        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function",
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"]
        )

        cavity_drive_pulse = pulse_library.gaussian_square(
            uid="cavity_drive_pulse",
            length=cavity_parameters[cavity_component]["cavity_drive_length"],
            amplitude=cavity_parameters[cavity_component]["cavity_drive_amp"]
        )

        phase = qubits_parameters[component]["readout_phase"]

        # Sweep parameters
        freq_sweep = LinearSweepParameter(uid="mode_freq", start=freq_start, stop=freq_stop, count=npts)

        # Define the experiment
        exp_cavity_mode_freq = Experiment(
            uid="cavity_mode_freq",
            signals=[
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_cavity_mode_freq.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_cavity_mode_freq.sweep(uid="sweep", parameter=freq_sweep, alignment=SectionAlignment.RIGHT):
                with exp_cavity_mode_freq.section(uid="cavity_excitation", alignment=SectionAlignment.RIGHT):
                    exp_cavity_mode_freq.play(signal="cavity_drive", pulse=cavity_drive_pulse)

                with exp_cavity_mode_freq.section(uid="measure", play_after="cavity_excitation"):
                    exp_cavity_mode_freq.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_cavity_mode_freq.acquire(
                        signal="acquire",
                        handle="cavity_mode_freq",
                        kernel=readout_weighting_function,
                    )
                with exp_cavity_mode_freq.section(uid="relax", 
                                           length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_cavity_mode_freq.reserve(signal="measure")
        
        exp_calibration = Calibration()
        # sets the oscillator of the experimental measure signal
        # for spectroscopy, set the sweep parameter as frequency
        cavity_mode_oscillator = Oscillator(
            "cavity_drive_if_osc",
            frequency=freq_sweep,
        )
        exp_calibration["cavity_drive"] = SignalCalibration( # experimental signal line 이름으로 signal calibration : 해당 실험 일시적 적용
            oscillator=cavity_mode_oscillator
        )

        exp_cavity_mode_freq.set_calibration(exp_calibration)

        signal_map = {
            "measure": device_setup.logical_signal_groups[component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }
        
        exp_cavity_mode_freq.set_signal_map(signal_map)

        compiled_experiment_cavity_mode_freq = self.session.compile(exp_cavity_mode_freq)

        self.cavity_mode_freq_results = self.session.run(compiled_experiment_cavity_mode_freq)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_cavity_mode_freq, start_time=0, length=20e-6)

    def plot_cavity_mode_spectroscopy(self):

        ### data processing ###############################################################

        averaged_nums = len(self.cavity_mode_freq_results.acquired_results['cavity_mode_freq'].axis[0])
        # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

        self.cavity_mode_freq_data = self.cavity_mode_freq_results.get_data("cavity_mode_freq") # (2^N, npts) array
        freqs = self.cavity_mode_freq_results.acquired_results['cavity_mode_freq'].axis[1]

        if self.which_data == "I":
            data = np.real(np.mean(self.cavity_mode_freq_data, axis = 0))
            std_data = np.real(np.std(self.cavity_mode_freq_data, axis = 0)/np.sqrt(averaged_nums))

        else:
            data = np.imag(np.mean(self.cavity_mode_freq_data, axis = 0))
            std_data = np.imag(np.std(self.cavity_mode_freq_data, axis = 0)/np.sqrt(averaged_nums))

        ### data plot ######################################################################

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        ax.errorbar(freqs, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))

        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=15)

        ax.set_title("cavity freq spectroscopy", fontsize=20)
        ax.set_xlabel("Freq", fontsize=20)
        ax.set_ylabel(f"{self.which_data} (a.u.)", fontsize=20)

        plt.show()

# In[] cavity_pi_nopi

    def cavity_pi_nopi(self, average_exponent=12, 
                       freq_start = 0, # freq in Hz
                       freq_stop = 0,
                       freq_npts = 0,
                       chunk_count = 2,
                       is_qubit2 = False,
                       qubit2=None,
                       auto_chunking=True,
                       is_plot_simulation=False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters

        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        if is_qubit2:
            self.is_qubit2 = True
            qubit2_component = list(qubits_parameters.keys())[qubit2]
            q2_pi2_pulse, q2_pi_pulse, q2_cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        cond_disp_pulse, cavity_drive_pulse, _, _ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        freq = cavity_parameters[cavity_component]["cond_disp_pulse_frequency"]

        sweep_case_1 = LinearSweepParameter(uid="on_off_case", start=0, stop=1, count=2)

        sweep_case_2 = LinearSweepParameter(uid="crosskerr_check", start=0, stop=1, count=2)

        sweep_freq_cases = LinearSweepParameter(uid="sweep_freq", start = freq + freq_start, stop = freq + freq_stop, count = freq_npts)

        def sweep_exp_cases(sweep_case: int | SweepParameter | LinearSweepParameter,
                            sweep_freq_cases: int | SweepParameter | LinearSweepParameter,
                            exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for exp_case_num in sweep_case.values:
                            with exp.case(exp_case_num):
                                with exp.match(sweep_parameter=sweep_freq_cases):
                                    for sweep_value in sweep_freq_cases.values:
                                        with exp.case(sweep_value):
                                            f(exp_case_num, sweep_value)
            return decorator
        
        def on_off_cond_pi_pulse(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)

            return decorator
        
        if is_qubit2:
            exp_cavity_pi_nopi = Experiment(
                uid="cavity_pi_nopi",
                signals=[
                    ExperimentSignal("cavity_drive"),
                    ExperimentSignal("drive"),
                    ExperimentSignal("drive2"),
                    ExperimentSignal("measure"),
                    ExperimentSignal("acquire"),
                ],
            )

        else :
            exp_cavity_pi_nopi = Experiment(
                uid="cavity_pi_nopi",
                signals=[
                    ExperimentSignal("cavity_drive"),
                    ExperimentSignal("drive"),
                    ExperimentSignal("measure"),
                    ExperimentSignal("acquire"),
                ],
            )


        with exp_cavity_pi_nopi.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_cavity_pi_nopi.sweep(uid="crosskerr_check", parameter=sweep_case_2):
                with exp_cavity_pi_nopi.sweep(uid="on_off_pi", parameter=sweep_case_1):
                    with exp_cavity_pi_nopi.sweep(uid="sweep_freq", parameter=sweep_freq_cases, chunk_count=chunk_count, auto_chunking=auto_chunking):

                        with exp_cavity_pi_nopi.section(uid="pi_nopi_cavity"):

                            @sweep_exp_cases(sweep_case_1, sweep_freq_cases, exp=exp_cavity_pi_nopi)
                            def play_cavity_pi_nopi(exp_case_num, sweep_value):

                                if exp_case_num == 0:  # nopi case

                                    with exp_cavity_pi_nopi.section():
                                        exp_cavity_pi_nopi.play(signal="cavity_drive", 
                                                                pulse=cond_disp_pulse, 
                                                                pulse_parameters={"frequency": sweep_value})
                                        exp_cavity_pi_nopi.reserve(signal="drive")
                                    
                                elif exp_case_num == 1: # pi case

                                    if is_qubit2:
                                        with exp_cavity_pi_nopi.section():
                                            exp_cavity_pi_nopi.play(signal="drive", pulse=pi_pulse)
                                            exp_cavity_pi_nopi.play(signal="drive2", pulse=q2_pi_pulse)
                                            exp_cavity_pi_nopi.reserve(signal="cavity_drive")
                                    else:
                                        with exp_cavity_pi_nopi.section():
                                            exp_cavity_pi_nopi.play(signal="drive", pulse=pi_pulse)
                                            exp_cavity_pi_nopi.reserve(signal="cavity_drive")

                                    with exp_cavity_pi_nopi.section():
                                        exp_cavity_pi_nopi.play(signal="cavity_drive", 
                                                                pulse=cond_disp_pulse, 
                                                                pulse_parameters={"frequency": sweep_value})
                                        if is_qubit2:
                                            exp_cavity_pi_nopi.reserve(signal="drive")
                                            exp_cavity_pi_nopi.reserve(signal="drive2")
                                        else:
                                            exp_cavity_pi_nopi.reserve(signal="drive")

                                    if is_qubit2:
                                        with exp_cavity_pi_nopi.section():
                                            exp_cavity_pi_nopi.play(signal="drive", pulse=pi_pulse)
                                            exp_cavity_pi_nopi.play(signal="drive2", pulse=q2_pi_pulse)
                                    else:
                                        with exp_cavity_pi_nopi.section():
                                            exp_cavity_pi_nopi.play(signal="drive", pulse=pi_pulse)

                            @on_off_cond_pi_pulse(sweep_case_2, exp=exp_cavity_pi_nopi)
                            def play_crosskerr_check(v):
                                if v == 0:
                                    self.handle_2 = "x"
                                    pass
                                elif v == 1:
                                    self.handle_2 = "o"
                                    exp_cavity_pi_nopi.play(signal="drive", pulse=cond_pi_pulse)
                    
                        with exp_cavity_pi_nopi.section(uid="measure", play_after="pi_nopi_cavity"):
                            exp_cavity_pi_nopi.play(signal="measure", pulse=readout_pulse, phase=phase)
                            exp_cavity_pi_nopi.acquire(
                                signal="acquire",
                                handle=f"cavity_pi_nopi",
                                kernel=readout_weighting_function,
                            )

                        with exp_cavity_pi_nopi.section(uid="relax", 
                                                        length=cavity_parameters[cavity_component]["reset_delay_length"]):
                            exp_cavity_pi_nopi.reserve(signal="measure")

        if is_qubit2:
            signal_map = {
                "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
                "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
                "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
                "drive2": device_setup.logical_signal_groups[qubit2_component].logical_signals["drive"],
                "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
            }
        else:
            signal_map = {
                "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
                "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
                "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
                "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
            }

        exp_cavity_pi_nopi.set_signal_map(signal_map)

        self.compiled_experiment_cavity_pi_nopi = self.session.compile(exp_cavity_pi_nopi)

        self.cavity_pi_nopi_results = self.session.run(self.compiled_experiment_cavity_pi_nopi)

        if is_plot_simulation:
            self.simulation_plot(self.compiled_experiment_cavity_pi_nopi, start_time=0, length=20e-6)
            show_pulse_sheet("cavity_pinopi", self.compiled_experiment_cavity_pi_nopi, max_events_to_publish=100000)
                    
    def plot_cavity_pi_nopi(self):

        self.cavity_pi_nopi_data = self.cavity_pi_nopi_results.get_data("cavity_pi_nopi")

        if self.which_data == "I":
            data = np.real(self.cavity_pi_nopi_data)
        else:
            data = np.imag(self.cavity_pi_nopi_data)

        if self.which_data == "I":
            data = np.real(self.cavity_pi_nopi_data)
        else:
            data = np.imag(self.cavity_pi_nopi_data)

        freq_sweep_list = self.cavity_pi_nopi_results.acquired_results['cavity_pi_nopi'].axis[2]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        freq = cavity_parameters[cavity_component]["cond_disp_pulse_frequency"]

        length = cavity_parameters[cavity_component]["cond_disp_pulse_length"]
        amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        detuning = cavity_parameters[cavity_component]["cond_disp_pulse_detuning"]
        sigma = cavity_parameters[cavity_component]["cond_disp_pulse_sigma"]

        ############# Plotting the data #####################

        fig, ax = plt.subplots(2, 1, figsize=(20, 16))

        fig.suptitle(f"length:{length}, amp:{amp}, detuning:{detuning/1e6}MHz, sigma:{sigma}", fontsize=18)

        ax[0].plot((freq_sweep_list-freq)/1e6, data[0][0], label='nopi', marker='o', linestyle=':', color='b')
        ax[0].plot((freq_sweep_list-freq)/1e6, data[0][1], label='pi', marker='o', linestyle=':', color='r')
        ax[0].set_title('Cavity pi-nopi, only cross Kerr effect')
        ax[0].set_xlabel('Frequency (MHz)')
        ax[0].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[0].legend()

        ax[1].plot((freq_sweep_list-freq)/1e6, data[1][0], label='nopi', marker='o', linestyle=':', color='b')
        ax[1].plot((freq_sweep_list-freq)/1e6, data[1][1], label='pi', marker='o', linestyle=':', color='r')
        ax[1].set_title('Cavity pi-nopi')
        ax[1].set_xlabel('Frequency (MHz)')
        ax[1].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[1].legend()

        fig2, ax2 = plt.subplots(1, 1, figsize=(20, 8))

        fig2.suptitle(f"length:{length}, amp:{amp}, detuning:{detuning/1e6}MHz, sigma:{sigma}", fontsize=18)

        ax2.plot((freq_sweep_list-freq)/1e6, data[1][0]-data[0][0], label='nopi', marker='o', linestyle=':', color='b')
        ax2.plot((freq_sweep_list-freq)/1e6, data[1][1]-data[0][1], label='pi', marker='o', linestyle=':', color='r')
        ax2.set_title('Cavity pi-nopi without cross Kerr effect')
        ax2.set_xlabel('Frequency (MHz)')
        ax2.set_ylabel(f'[{self.which_data}] (a.u.)')
        ax2.legend()

        plt.show()


# In[] CNOD_calibration

    def CNOD_calibration(self, average_exponent=12, amp_range=1, npts=11, qubit_phase = 0, is_displaced_state = False, is_plot_simulation=False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        self.qubit_phase = qubit_phase # real or imag of characteristic function
        self.is_displaced_state = is_displaced_state

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        cond_disp_pulse, cavity_drive_pulse, _, _ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        self.CNOD_calibration_amp_range = amp_range
        amplitude_sweep = LinearSweepParameter(uid="delay", start= -amp_range, stop=amp_range, count=npts)

        sweep_case = LinearSweepParameter(uid="crosskerr_check", start=0, stop=1, count=2)

        def crosskerr_check(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)

            return decorator

        # Define the experiment
        exp_CNOD_calibration = Experiment(
            uid="CNOD_calibration",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_CNOD_calibration.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True,
        ):
            # with exp_CNOD_calibration.sweep(uid="crosskerr_check", parameter=sweep_case):
            with exp_CNOD_calibration.sweep(uid="sweep", parameter=amplitude_sweep, reset_oscillator_phase=True):
                
                if is_displaced_state:
                    with exp_CNOD_calibration.section(uid="disp_pulse"):
                        exp_CNOD_calibration.play(signal="cavity_drive", pulse=cavity_drive_pulse)

                    with exp_CNOD_calibration.section(uid="qubit_excitation_1", play_after="disp_pulse"):
                        exp_CNOD_calibration.play(signal="drive", pulse=pi2_pulse)

                    self.CNOD(exp = exp_CNOD_calibration, cond_disp_pulse = cond_disp_pulse,
                            pi_pulse = pi_pulse, amp = amplitude_sweep, 
                            prev_uid="qubit_excitation_1")

                else:
                    with exp_CNOD_calibration.section(uid="qubit_excitation_1"):
                        exp_CNOD_calibration.play(signal="drive", pulse=pi2_pulse)

                    self.CNOD(exp = exp_CNOD_calibration, cond_disp_pulse = cond_disp_pulse,
                            pi_pulse = pi_pulse, amp = amplitude_sweep, 
                            prev_uid="qubit_excitation_1")
                     
                # with exp_CNOD_calibration.section(uid="qubit_excitation_2", play_after="cond_disp_pulse_2"):
                #     @crosskerr_check(sweep_case, exp=exp_CNOD_calibration)
                #     def play_crosskerr_check(v):
                #                 if v == 0:
                #                     pass
                #                 elif v == 1:
                #                     exp_CNOD_calibration.play(signal="drive", pulse=pi2_pulse, phase = qubit_phase)
                with exp_CNOD_calibration.section(uid="qubit_excitation_2", play_after="cond_disp_pulse_2"):
                    exp_CNOD_calibration.play(signal="drive", pulse=pi2_pulse, phase = qubit_phase)
                    
                with exp_CNOD_calibration.section(uid="measure", play_after="qubit_excitation_2"):
                    exp_CNOD_calibration.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_CNOD_calibration.acquire(signal="acquire",
                                                handle="CNOD_calibration",
                                                kernel=readout_weighting_function)
                with exp_CNOD_calibration.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_CNOD_calibration.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_CNOD_calibration.set_signal_map(signal_map)

        compiled_exp_CNOD_calibration = self.session.compile(exp_CNOD_calibration)

        self.CNOD_calibration_results = self.session.run(compiled_exp_CNOD_calibration)

        if is_plot_simulation:
            self.simulation_plot(compiled_exp_CNOD_calibration, start_time=0, length=20e-6)
            show_pulse_sheet("CNOD_calibration", compiled_exp_CNOD_calibration)

    def plot_CNOD_calibration(self, scaling_factor=1):

        self.CNOD_calibration_data = self.CNOD_calibration_results.get_data("CNOD_calibration")

        if self.which_data == "I":
            data = np.real(self.CNOD_calibration_data)
        else:
            data = np.imag(self.CNOD_calibration_data)

        if np.isreal(self.CNOD_calibration_amp_range):
            # real axis sweep
            amplitude_sweep_list = np.real(self.CNOD_calibration_results.acquired_results['CNOD_calibration'].axis[0])
        else:
            # imaginary axis sweep
            amplitude_sweep_list = np.imag(self.CNOD_calibration_results.acquired_results['CNOD_calibration'].axis[0])

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        length = cavity_parameters[cavity_component]["cond_disp_pulse_length"]
        amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        detuning = cavity_parameters[cavity_component]["cond_disp_pulse_detuning"]
        sigma = cavity_parameters[cavity_component]["cond_disp_pulse_sigma"]
        # alpha_1_CNOD_amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"]

        if self.is_displaced_state:
            disp_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
            disp_length = cavity_parameters[cavity_component]["cavity_drive_length"]
            

        ############# Plotting the data #####################

        fig, ax = plt.subplots(1, 1, figsize=(20, 12))

        fig.suptitle(f"CNOD amp calibration", fontsize=18)
        
        fig.text(0.5, 0.91, f"length:{length}, amp:{amp}, detuning:{detuning/1e6}MHz, sigma:{sigma}", ha="center", fontsize=16, color="black")
        
        ax.plot(amp*amplitude_sweep_list/scaling_factor, data, marker='o', label = 'original', linestyle=':', color='k')
        

        if self.is_displaced_state:
            
            sfit1 = sFit('GaussianCos', amp*amplitude_sweep_list/scaling_factor, data)
            # Scale the amplitude into alpha (photon number) by using alpha_1_CNOD_amp
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            ax.plot(amp*amplitude_sweep_list/scaling_factor, 
                    sfit1.func(amp*amplitude_sweep_list/scaling_factor, *popt), label='fit', color='g')
            
            an1 = ax.annotate(rf'frequency = {popt[1]:.4f}±{np.sqrt(np.diag(pcov))[1]:.4f}'+'\n'
                              + rf'$\sigma$ = {popt[2]:.4f}±{np.sqrt(np.diag(pcov))[2]:.4f},'+ '\n'
                              + rf'$\beta$ = {popt[1]*np.pi:.4f}',
                xy=(np.average(amplitude_sweep_list), np.average(data)*0.95),
                size=12)
            an1.draggable()
            
            an2 = ax.annotate(f'Disp_amp = {disp_amp:.4f}, Disp_length = {np.round(disp_length*1e9)}ns',
                xy=(np.average(amplitude_sweep_list), np.average(data)*0.9),
                size=12)
            an2.draggable()
        else:
            sfit1 = sFit('Gaussian', amp*amplitude_sweep_list/scaling_factor, data)
            # Scale the amplitude into alpha (photon number) by using alpha_1_CNOD_amp
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            ax.plot(amp*amplitude_sweep_list/scaling_factor, 
                    sfit1.func(amp*amplitude_sweep_list/scaling_factor, *popt), label='fit', color='g')
            
            an1 = ax.annotate(rf'$\sigma$ = {popt[1]:.4f}±{np.sqrt(np.diag(pcov))[1]:.4f},',
                xy=(np.average(amplitude_sweep_list), np.average(data)*0.95),
                size=12)
            an1.draggable()
            
        
        if self.qubit_phase == 0:
            ax.set_title('real part')
        else:
            ax.set_title('imaginary part')
        ax.set_xlabel('Amplitude (a.u.)')
        ax.set_ylabel(f'[{self.which_data}] (a.u.)')
        ax.legend()
 
        plt.show()

# In[] acquired CNOD_geometric_phase

    # def CNOD_geophase_calibration(self, average_exponent=12, amp_sweep = 1, amp_npts = 11, is_plot_simulation=False):
        
    #     device_setup = self.device_setup
    #     qubits_parameters = self.qubits_parameters
    #     cavity_parameters = self.cavity_parameters
    #     qubits_component = list(qubits_parameters.keys())[self.which_qubit]
    #     cavity_component = list(cavity_parameters.keys())[self.which_mode]

    #     # Define pulses
    #     readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
    #                     qubits_component, cavity_component)

    #     pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
    #                     qubits_component, cavity_component)
        
    #     cond_disp_pulse, cavity_drive_pulse = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
    #                                                                qubits_component, cavity_component)

    #     phase = qubits_parameters[qubits_component]["readout_phase"]

    #     # Sweep parameters
    #     disp_amp_sweep = LinearSweepParameter(uid="disp_amp_sweep", start=-amp_sweep, stop=amp_sweep, count=amp_npts)

    #     # Define the experiment
    #     exp_disp_calibration = Experiment(
    #         uid="disp_calibration",
    #         signals=[
    #             ExperimentSignal("drive"),
    #             ExperimentSignal("cavity_drive"),
    #             ExperimentSignal("measure"),
    #             ExperimentSignal("acquire"),
    #         ],
    #     )
    #     with exp_disp_calibration.acquire_loop_rt(
    #         uid="shots",
    #         count=2**average_exponent,
    #         averaging_mode=AveragingMode.CYCLIC,
    #         acquisition_type=AcquisitionType.INTEGRATION,
    #         reset_oscillator_phase=True
    #     ):
    #         with exp_disp_calibration.sweep(uid="disp_amp_sweep", parameter=disp_amp_sweep, reset_oscillator_phase=True):
    #             with exp_disp_calibration.section(uid="qubit_excitation_1"):
    #                 exp_disp_calibration.play(signal="drive", pulse=pi2_pulse)
                
    #             self.CNOD(exp = exp_disp_calibration, cond_disp_pulse = cond_disp_pulse,
    #                       pi_pulse = pi_pulse, amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"], 
    #                       prev_uid="qubit_excitation_1", uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid= "pi_pulse_1")
                
    #             self.CNOD(exp = exp_disp_calibration, cond_disp_pulse = cond_disp_pulse,
    #                       pi_pulse = pi_pulse, amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"], 
    #                       prev_uid="cond_disp_pulse_2", uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid= "pi_pulse_2")
   
    #             with exp_disp_calibration.section(uid="qubit_excitation_2", play_after="disp_pulse_2"):
    #                 exp_disp_calibration.play(signal="drive", pulse=pi2_pulse, phase = 0)
                    
    #             with exp_disp_calibration.section(uid="measure", play_after="qubit_excitation_2"):
    #                 exp_disp_calibration.play(signal="measure", pulse=readout_pulse, phase=phase)
    #                 exp_disp_calibration.acquire(signal="acquire",
    #                                             handle="disp_calibration",
    #                                             kernel=readout_weighting_function)
    #             with exp_disp_calibration.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
    #                 exp_disp_calibration.reserve(signal="measure")
        
    #     signal_map = {
    #         "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
    #         "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
    #         "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
    #         "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
    #     }

    #     exp_disp_calibration.set_signal_map(signal_map)
    #     compiled_exp_disp_calibration = self.session.compile(exp_disp_calibration)
    #     self.disp_calibration_results = self.session.run(compiled_exp_disp_calibration)
    #     if is_plot_simulation:
    #         self.simulation_plot(compiled_exp_disp_calibration, start_time=0, length=20e-6)
    #         show_pulse_sheet("disp_calibration", compiled_exp_disp_calibration)

# In[] Char_func_displaced

    def Characteristic_function_2D(self, average_exponent=12, amp_range=1, npts=11, qubit_phase = 0, is_plot_simulation=False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        self.qubit_phase = qubit_phase # real or imag of characteristic function

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        cond_disp_pulse, cavity_drive_pulse, _, _ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Create a 2D grid of complex amplitudes from (-amp_range-1j*amp_range) to (amp_range+1j*amp_range)
        real_vals = np.linspace(-0.5, 0.5, npts) # should not be over unity
        imag_vals = np.linspace(-0.5, 0.5, npts) # should not be over unity
        amplitude_grid = real_vals[:, None] + 1j * imag_vals[None, :]
        
        self.amp_npts = npts

        # Sweep parameters
        amplitude_sweep = SweepParameter(uid="amp_sweep", values=amplitude_grid.flatten())
        # Define the experiment
        exp_Characteristic_function_2D = Experiment(
            uid="Char_func_2D",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        def displaced_coherent_state():

            with exp_Characteristic_function_2D.section(uid="preparation", alignment=SectionAlignment.RIGHT):

                exp_Characteristic_function_2D.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude)
        
        def schrodinger_cat_state():
            with exp_Characteristic_function_2D.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_Characteristic_function_2D.section(uid="qubit_preparation"):
                    exp_Characteristic_function_2D.play(signal="drive", pulse=pi2_pulse)
                with exp_Characteristic_function_2D.section(uid="CNOD", play_after="qubit_preparation"):
                    self.CNOD(exp=exp_Characteristic_function_2D, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=1,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None)

        def schrodinger_cat_state_2():
            with exp_Characteristic_function_2D.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                
                with exp_Characteristic_function_2D.section(uid="alpha"):
                    exp_Characteristic_function_2D.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude)
                with exp_Characteristic_function_2D.section(uid="qubit", play_after="alpha"):
                    exp_Characteristic_function_2D.play(signal="drive", pulse=pi2_pulse)
                    exp_Characteristic_function_2D.delay(signal="drive", time=np.abs(1/(2*cavity_parameters[cavity_component]["cavity_mode_chi"])/2)) # delay for cross Kerr effect

        def cat_state():
            with exp_Characteristic_function_2D.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_Characteristic_function_2D.section(uid="qubit_preparation"):
                    exp_Characteristic_function_2D.play(signal="drive", pulse=pi2_pulse)
                with exp_Characteristic_function_2D.section(uid="CNOD_1", play_after="qubit_preparation"):
                    self.CNOD(exp=exp_Characteristic_function_2D, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=alpha,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid = "pi_pulse_1")
                with exp_Characteristic_function_2D.section(uid="qubit_1", play_after="CNOD_1"):
                    exp_Characteristic_function_2D.play(signal="drive", pulse=pi2_pulse)
                with exp_Characteristic_function_2D.section(uid="CNOD_2", play_after="qubit_1"):
                    self.CNOD(exp=exp_Characteristic_function_2D, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=1j*beta,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid = "pi_pulse_2")
                with exp_Characteristic_function_2D.section(uid="qubit_2", play_after="CNOD_2"):
                    exp_Characteristic_function_2D.play(signal="drive", pulse=pi2_pulse, phase=np.pi/2)




        with exp_Char_func_displaced.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True
        ):
            # with exp_CNOD_calibration.sweep(uid="crosskerr_check", parameter=sweep_case):
            with exp_Char_func_displaced.sweep(uid="sweep", parameter=amplitude_sweep, auto_chunking = True, reset_oscillator_phase=True):
                with exp_Char_func_displaced.section(uid="qubit_excitation_1"):
                    exp_Char_func_displaced.play(signal="drive", pulse=pi2_pulse)

                with exp_Char_func_displaced.section(uid="disp_pulse", play_after="qubit_excitation_1"):
                    exp_Char_func_displaced.play(signal="cavity_drive", pulse=cavity_drive_pulse)
                
                self.CNOD(exp = exp_Char_func_displaced, cond_disp_pulse = cond_disp_pulse, 
                          pi_pulse = pi_pulse, amp = amplitude_sweep, prev_uid="disp_pulse")
                
                with exp_Char_func_displaced.section(uid="qubit_excitation_2", play_after="cond_disp_pulse_2"):
                    exp_Char_func_displaced.play(signal="drive", pulse=pi2_pulse, phase = qubit_phase)
                    
                with exp_Char_func_displaced.section(uid="measure", play_after="qubit_excitation_2"):
                    exp_Char_func_displaced.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_Char_func_displaced.acquire(signal="acquire",
                                                handle="Char_func_displaced",
                                                kernel=readout_weighting_function)
                with exp_Char_func_displaced.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_Char_func_displaced.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_Char_func_displaced.set_signal_map(signal_map)

        compiled_exp_Char_func_displaced = self.session.compile(exp_Char_func_displaced)

        self.Char_func_displaced_results = self.session.run(compiled_exp_Char_func_displaced)

        if is_plot_simulation:
            self.simulation_plot(compiled_exp_Char_func_displaced, start_time=0, length=20e-6)
            show_pulse_sheet("CNOD_calibration", compiled_exp_Char_func_displaced)

    def plot_Characteristic_function_2D(self):

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        
        length = cavity_parameters[cavity_component]["cavity_drive_length"]
        amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        alpha_1_CNOD_amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"]
        cond_disp_pulse_amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]

        self.Char_func_displaced_data = self.Char_func_displaced_results.get_data("Char_func_displaced")

        if self.which_data == "I":
            data = np.real(self.Char_func_displaced_data)
        else:
            data = np.imag(self.Char_func_displaced_data)

        amplitude_sweep_list = self.Char_func_displaced_results.acquired_results['Char_func_displaced'].axis[0]

        Z = amplitude_sweep_list.reshape((self.amp_npts, self.amp_npts))
        x = np.real(Z) * cond_disp_pulse_amp/alpha_1_CNOD_amp # scaling to alpha (photon number)
        y = np.imag(Z) * cond_disp_pulse_amp/alpha_1_CNOD_amp   # scaling to alpha (photon number)
        data = data.reshape((self.amp_npts, self.amp_npts))

        ############# Plotting the data #####################

        plt.figure(figsize=(10, 10))
        pcm = plt.pcolormesh(x, y, data, shading='auto', cmap='RdBu_r')
        plt.colorbar(pcm, label=f'[{self.which_data}] (a.u.)')
        plt.title(f"Char func of displaced state, length:{length}, amp:{amp}")
        plt.xlabel(r'Re($\alpha$)')
        plt.ylabel(r'Im($\alpha$)')

        plt.show()

# In[] disp_pulse_calibration
    def disp_pulse_calibration_geophase(self, average_exponent=12, amp_sweep = 1, amp_npts = 11, is_plot_simulation=False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        cond_disp_pulse, cavity_drive_pulse, _, _ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        disp_amp_sweep = LinearSweepParameter(uid="disp_amp_sweep", start=-amp_sweep, stop=amp_sweep, count=amp_npts)

        # Define the experiment
        exp_disp_calibration = Experiment(
            uid="disp_calibration",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        with exp_disp_calibration.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True
        ):
            with exp_disp_calibration.sweep(uid="disp_amp_sweep", parameter=disp_amp_sweep, reset_oscillator_phase=True):
                with exp_disp_calibration.section(uid="qubit_excitation_1"):
                    exp_disp_calibration.play(signal="drive", pulse=pi2_pulse)
                
                self.CNOD(exp = exp_disp_calibration, cond_disp_pulse = cond_disp_pulse,
                          pi_pulse = pi_pulse, amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"], 
                          prev_uid="qubit_excitation_1", uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid= "pi_pulse_1")

                with exp_disp_calibration.section(uid="disp_puls_1", play_after="cond_disp_pulse_2"):
                    exp_disp_calibration.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude = 1j*disp_amp_sweep)
                
                self.CNOD(exp = exp_disp_calibration, cond_disp_pulse = cond_disp_pulse,
                          pi_pulse = pi_pulse, amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"], 
                          prev_uid="disp_puls_1", uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid= "pi_pulse_2")

                with exp_disp_calibration.section(uid="disp_pulse_2", play_after="cond_disp_pulse_4"):
                    exp_disp_calibration.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude = 1j*disp_amp_sweep, phase = np.pi)
                
                with exp_disp_calibration.section(uid="qubit_excitation_2", play_after="disp_pulse_2"):
                    exp_disp_calibration.play(signal="drive", pulse=pi2_pulse, phase = 0)
                    
                with exp_disp_calibration.section(uid="measure", play_after="qubit_excitation_2"):
                    exp_disp_calibration.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_disp_calibration.acquire(signal="acquire",
                                                handle="disp_calibration",
                                                kernel=readout_weighting_function)
                with exp_disp_calibration.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_disp_calibration.reserve(signal="measure")
        
        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_disp_calibration.set_signal_map(signal_map)
        compiled_exp_disp_calibration = self.session.compile(exp_disp_calibration)
        self.disp_calibration_results = self.session.run(compiled_exp_disp_calibration)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_disp_calibration, start_time=0, length=20e-6)
            show_pulse_sheet("disp_calibration", compiled_exp_disp_calibration)
    
    def plot_disp_pulse_calibration_geophase(self, is_fit=True):

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        length = cavity_parameters[cavity_component]["cavity_drive_length"]
        amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        cond_pulse_length = cavity_parameters[cavity_component]["cond_disp_pulse_length"]
        cond_pulse_amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        alpha_1_CNOD_amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"]
        detuning = cavity_parameters[cavity_component]["cond_disp_pulse_detuning"]
        sigma = cavity_parameters[cavity_component]["cond_disp_pulse_sigma"]

        self.disp_calibration_data = self.disp_calibration_results.get_data("disp_calibration")

        if self.which_data == "I":
            data = np.real(self.disp_calibration_data)
        else:
            data = np.imag(self.disp_calibration_data)

        disp_amp_sweep_list = self.disp_calibration_results.acquired_results['disp_calibration'].axis[0]

        ############# Plotting the data #####################

        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        fig.suptitle("Displacement Pulse Calibration", fontsize=18, weight='bold')
        fig.text(0.5, 0.94, f"Displacement pulse length: {length:.1e}", ha="center", fontsize=14)
        
        ax.plot(amp*disp_amp_sweep_list, data, marker='o', linestyle=':', color='k')

        if is_fit :
            sfit1 = sFit('Cos', amp*disp_amp_sweep_list, data)
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            ax.plot(amp*disp_amp_sweep_list, sfit1.func(amp*disp_amp_sweep_list, *popt), label='fit', color='g')
        
        freq = popt[1]
        freq_err = np.sqrt(np.diag(pcov))[1]
        annotation_text = f"freq = {freq:.4f} ± {freq_err:.4f}\nscaling factor = π x freq"

        an = ax.annotate(annotation_text,
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=10, ha='left', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        an2 = ax.annotate(f'CNOD_amp = {alpha_1_CNOD_amp*cond_pulse_amp}\ncond_pulse_length ={cond_pulse_length}',
                            xy=(0, np.average(data)*0.9))
        
        an.draggable()
        an2.draggable()

        ax.set_xlabel('Driving Amplitude')
        ax.set_ylabel(f'[{self.which_data}] (a.u.)')

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Displacement ($\\beta$)')

        xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 6)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f"{x * np.pi * freq:.2f}" for x in xticks])

        ax.plot([1/(np.pi*freq), 1/(np.pi*freq)], [np.min(data), np.max(data)], '--k')

        print(f"beta 1 amplitude is {1/(np.pi*freq)}")

        ax.tick_params(axis='both', labelsize=10)
        ax2.tick_params(axis='x', labelsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


    def disp_pulse_calibration_parity(self, average_exponent=12, amp_start = 0, amp_stop=1, amp_npts = 11, is_plot_simulation=False):
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        cond_disp_pulse, cavity_drive_pulse, _, _ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        disp_amp_sweep = LinearSweepParameter(uid="disp_amp_sweep", start= amp_start, stop= amp_stop, count=amp_npts)

        sweep_case = LinearSweepParameter(uid="correction", start=0, stop=1, count=2)

        def correction(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)

            return decorator

        # Define the experiment
        exp_disp_calibration_parity = Experiment(
            uid="disp_calibration_parity",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        with exp_disp_calibration_parity.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_disp_calibration_parity.sweep(uid="correction", parameter=sweep_case):
                with exp_disp_calibration_parity.sweep(uid="disp_amp_sweep", parameter=disp_amp_sweep):
                    with exp_disp_calibration_parity.section(uid="disp_amp_drive"):
                        exp_disp_calibration_parity.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude = disp_amp_sweep)
                    
                    with exp_disp_calibration_parity.section(uid="qubit_excitation_1", play_after="disp_amp_drive"):
                        exp_disp_calibration_parity.play(signal="drive", pulse=pi2_pulse)
                        exp_disp_calibration_parity.delay(signal="drive", time=np.abs(1/(2*cavity_parameters[cavity_component]["cavity_mode_chi"]))) # delay for cross Kerr effect

                    with exp_disp_calibration_parity.section(uid="qubit_excitation_2", play_after="qubit_excitation_1"):
                        @correction(sweep_case, exp=exp_disp_calibration_parity)
                        def play_correction(v):
                            if v == 0:
                                exp_disp_calibration_parity.play(signal="drive", pulse=pi2_pulse, phase = 0)
                            elif v == 1:
                                exp_disp_calibration_parity.play(signal="drive", pulse=pi2_pulse, phase = np.pi)
                    
                    with exp_disp_calibration_parity.section(uid="measure", play_after="qubit_excitation_2"):
                        exp_disp_calibration_parity.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_disp_calibration_parity.acquire(signal="acquire",
                                                handle="disp_calibration_parity",
                                                kernel=readout_weighting_function)
                    with exp_disp_calibration_parity.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_disp_calibration_parity.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_disp_calibration_parity.set_signal_map(signal_map)
        compiled_exp_disp_calibration_parity = self.session.compile(exp_disp_calibration_parity)
        self.disp_calibration_parity_results = self.session.run(compiled_exp_disp_calibration_parity)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_disp_calibration_parity, start_time=0, length=20e-6)
            show_pulse_sheet("disp_calibration_parity", compiled_exp_disp_calibration_parity)
                    
    def plot_disp_pulse_calibration_parity(self, is_fit=True, scaling_factor=1):

        self.disp_calibration_parity_data = self.disp_calibration_parity_results.get_data("disp_calibration_parity")

        if self.which_data == "I":
            data = np.real(self.disp_calibration_parity_data)
        else:
            data = np.imag(self.disp_calibration_parity_data)

        disp_amp_sweep_list = self.disp_calibration_parity_results.acquired_results['disp_calibration_parity'].axis[1]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        length = cavity_parameters[cavity_component]["cavity_drive_length"]
        amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        chi = cavity_parameters[cavity_component]["cavity_mode_chi"]

        ############# Plotting the data #####################

        fig, ax = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle(f"Displacement pulse calibration parity", fontsize=18)
        fig.text(0.5, 0.91, f"drive_length:{length}, chi:{chi}", ha="center", fontsize=16, color="black")
        ax[0].plot(amp*disp_amp_sweep_list/scaling_factor, data[0], marker='o', linestyle=':', color='k', label='uncorrected')
        ax[0].plot(amp*disp_amp_sweep_list/scaling_factor, data[1], marker='o', linestyle=':', color='r', label='correction')
        ax[1].plot(amp*disp_amp_sweep_list/scaling_factor, data[0]-data[1], marker='o', linestyle=':', color='k', label='corrected')

        if is_fit :

            sfit1 = sFit('Gaussian', amp*disp_amp_sweep_list/scaling_factor, data[0]-data[1])
            # Scale the amplitude into alpha (photon number) by using alpha_1_CNOD_amp
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            ax[1].plot(amp*disp_amp_sweep_list/scaling_factor, 
                    sfit1.func(amp*disp_amp_sweep_list/scaling_factor, *popt), label='fit', color='g')
        
            an = ax[1].annotate(f'sigma = {popt[1]:.4f}±{np.sqrt(np.diag(pcov))[1]:.4f}',
                            xy=(np.average(amp*disp_amp_sweep_list/scaling_factor), np.average(data[0]-data[1])))
            an.draggable()

        ax[0].set_xlabel('Drive Amplitude (a.u.)')
        ax[0].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[0].legend()
        ax[1].set_xlabel('Drive Amplitude (a.u.)')
        ax[1].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[1].legend()








# In[] out-and-back measurement

    def out_and_back_measurement(self, average_exponent=12, init_state = "g",
                                 phase_start=0, phase_stop=20, # deg
                                 cavity_drive_amp_start = 0, cavity_drive_amp_stop = 0.1,
                                 phase_npts=11, amp_npts=11,
                                 wait_time=1e-6,
                                 is_plot_simulation=False):
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        cond_disp_pulse, cavity_drive_pulse, _, _ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        phase_sweep = LinearSweepParameter(uid="phase_sweep", start=phase_start, stop=phase_stop, count=phase_npts)

        drive_amp_sweep = LinearSweepParameter(uid="drive_amp_sweep", start=cavity_drive_amp_start, stop=cavity_drive_amp_stop, count=amp_npts)
        # Define the conditional pi pulse
        # Define the experiment
        exp_out_and_back = Experiment(
            uid="out_and_back",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        conditional_uid = None

        with exp_out_and_back.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_out_and_back.sweep(uid="drive_amp_sweep", parameter=drive_amp_sweep):
                with exp_out_and_back.sweep(uid="phase_sweep", parameter=phase_sweep):
                    if init_state == "e":
                        conditional_uid = "qubit_excitation"
                        with exp_out_and_back.section(uid=conditional_uid):
                            exp_out_and_back.play(signal="drive", pulse=pi_pulse)
                    
                    with exp_out_and_back.section(uid="displacement", play_after=conditional_uid):
                        exp_out_and_back.play(signal="cavity_drive", pulse=cavity_drive_pulse,
                                              amplitude=drive_amp_sweep)
                        exp_out_and_back.delay(signal="cavity_drive", time=wait_time)
                        exp_out_and_back.play(signal="cavity_drive", pulse=cavity_drive_pulse, 
                                            phase=np.pi/180*phase_sweep+np.pi,
                                            amplitude=drive_amp_sweep)
                    
                    with exp_out_and_back.section(uid="cond_disp_pi", play_after="displacement"):
                        exp_out_and_back.play(signal="drive", pulse=cond_pi_pulse)
                    
                    with exp_out_and_back.section(uid="measure", play_after="cond_disp_pi"):
                        exp_out_and_back.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_out_and_back.acquire(signal="acquire",
                                                    handle="out_and_back",
                                                    kernel=readout_weighting_function)
                    with exp_out_and_back.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_out_and_back.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_out_and_back.set_signal_map(signal_map)
        compiled_exp_out_and_back = self.session.compile(exp_out_and_back)
        self.out_and_back_results = self.session.run(compiled_exp_out_and_back)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_out_and_back, start_time=0, length=20e-6)
            show_pulse_sheet("out_and_back", compiled_exp_out_and_back)


    def plot_out_and_back_measurement(self, fitting=False, x_threshold=None,y_threshold=None, wait_time=1e-6, init_state = "g",):
        from sklearn.linear_model import LinearRegression

        self.out_and_back_data = self.out_and_back_results.get_data("out_and_back")

        if self.which_data == "I":
            data = np.real(self.out_and_back_data)
        else:
            data = np.imag(self.out_and_back_data)

        drive_amp_sweep_list = self.out_and_back_results.acquired_results['out_and_back'].axis[0]
        phase_sweep_list = self.out_and_back_results.acquired_results['out_and_back'].axis[1]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        max_drive_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        alpha_1_cavity_drive_amp = cavity_parameters[cavity_component]['alpha_1_cavity_drive_amp']

        x_axis = np.array(drive_amp_sweep_list*max_drive_amp) ** 2 / alpha_1_cavity_drive_amp**2
        y_axis = phase_sweep_list
        X_grid, Y_grid = np.meshgrid(x_axis, y_axis, indexing='ij')  # shape = (M, N)
        Z_data = data  # shape = (M, N)

        X_flat = X_grid.flatten()
        Y_flat = Y_grid.flatten()
        Z_flat = Z_data.flatten()

        self.nopi_value

        if self.nopi_value > self.pi_value: # pi nopi measurement 값을 비교
            if init_state == "g":
                z_max = np.max(Z_flat) # 나중에 pi nopi measurement에서 z_max, z_min을 이용해서
                # z_range를 구하고, 그 범위에 해당하는 데이터만을 선택하여 fitting을 수행.
                z_min = np.min(Z_flat)
                z_range = (1/3) * (z_max - z_min)
                mask = (Z_flat >= z_min) & (Z_flat <= z_min + z_range)
            else:
                z_max = np.max(Z_flat)
                z_min = np.min(Z_flat)
                z_range = (1/3) * (z_max - z_min)
                mask = (Z_flat <= z_max) & (Z_flat >= z_max - z_range)
        
        else:
            if init_state == "g":
                z_max = np.max(Z_flat)
                z_min = np.min(Z_flat)
                z_range = (1/3) * (z_max - z_min)
                mask = (Z_flat <= z_max) & (Z_flat >= z_max - z_range)
            else:
                z_max = np.max(Z_flat)
                z_min = np.min(Z_flat)
                z_range = (1/3) * (z_max - z_min)
                mask = (Z_flat >= z_min) & (Z_flat <= z_min + z_range)

        if x_threshold is not None:
            mask = mask & (X_flat >= x_threshold)

        if y_threshold is not None:
            mask = mask & (Y_flat <= y_threshold)

        X_vals = X_flat[mask].reshape(-1, 1)
        Y_vals = Y_flat[mask]

        # y축 라인별로 x값 평균 구하고 scatter plot
        X_avgs = []
        Y_avgs = []
        for i in range(len(x_axis)):
            y_mask = (X_vals.flatten() == x_axis[i])
            if np.any(y_mask):
                mean_value = np.mean(Y_vals[y_mask]) 
                X_avgs.append(x_axis[i])
                Y_avgs.append(mean_value)

        model = LinearRegression()
        model.fit(np.array(X_avgs).reshape(-1, 1), np.array(Y_avgs))
        fit_line = model.predict(np.array(X_avgs).reshape(-1, 1))
        slope = model.coef_[0]
        intercept = model.intercept_
        print(f" relative frequency(Hz) = {slope/ (wait_time  * 360):.3f} * Photon # + {intercept/ (wait_time  * 360):.3f}")
        print(f" K_c = 2Pi X {-1 * slope/ (2 * wait_time  * 360):.3f} Hz")

        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        fig2.suptitle("Out and back measurement - Scatter Plot", fontsize=18)
        plt.scatter(X_avgs, np.array(Y_avgs)/ (wait_time  * 360) , label=f'Phase = {mean_value:.2f} deg', alpha=1, facecolors='none', edgecolors='black')
        ax2.plot(X_avgs, fit_line/ (wait_time  * 360), color='dodgerblue', label='Linear Fit')
        ax2.set_xlabel('Photon number')
        ax2.set_ylabel('relative frequency (Hz)')
        ax2.text(0.5, 0.05, f'K_c = {slope/ (2 * wait_time  * 360):.3f} Hz', transform=ax2.transAxes, fontsize=12, verticalalignment='top')
        plt.show()
        
        # Create a 2D plot with the original data
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle("Out and back measurement", fontsize=18)
        pcm = ax.pcolormesh(x_axis, y_axis, Z_data.T, shading='auto')
        plt.colorbar(pcm, ax=ax, label=f'[{self.which_data}] (a.u.)')
        ax.plot(X_vals, model.predict(np.array(X_vals).reshape(-1,1)), color='dodgerblue', label='Linear Fit')
        ax.legend()

        ax.set_xlabel('Photon number')
        ax.set_ylabel('Phase (deg)')
        plt.show()

# In[] to get chi between memory and qubit, ramsey-like experiment
    def qubit_state_revival(self, average_exponent=12, 
                            wait_time=1e-6,
                            wait_npts = 11,
                            is_plot_simulation=False):
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        cond_disp_pulse, cavity_drive_pulse, _, _ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                         qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        delay_sweep = LinearSweepParameter(uid="delay_sweep", start=0, stop=wait_time, count=wait_npts)

        # Define the experiment
        exp_qubit_state_revival = Experiment(
            uid="qubit_state_revival",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_qubit_state_revival.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_qubit_state_revival.sweep(uid="drive_amp_sweep", parameter=delay_sweep):
                with exp_qubit_state_revival.section(uid="displacement"):
                    exp_qubit_state_revival.play(signal="cavity_drive", pulse=cavity_drive_pulse)
                                                 
                with exp_qubit_state_revival.section(uid="qubit_excitation_1", play_after="displacement"):
                    exp_qubit_state_revival.play(signal="drive", pulse=pi2_pulse)
                    exp_qubit_state_revival.delay(signal="drive", time=delay_sweep)
                    exp_qubit_state_revival.play(signal="drive", pulse=pi2_pulse)
                
                with exp_qubit_state_revival.section(uid="measure", play_after="qubit_excitation_1"):
                    exp_qubit_state_revival.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_qubit_state_revival.acquire(signal="acquire",
                                                    handle="qubit_state_revival",
                                                    kernel=readout_weighting_function)
                with exp_qubit_state_revival.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_qubit_state_revival.reserve(signal="measure")
        
        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_qubit_state_revival.set_signal_map(signal_map)
        compiled_exp_qubit_state_revival = self.session.compile(exp_qubit_state_revival)
        self.qubit_state_revival_results = self.session.run(compiled_exp_qubit_state_revival)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_qubit_state_revival, start_time=0, length=20e-6)
            show_pulse_sheet("qubit_state_revival", compiled_exp_qubit_state_revival)
        
    def plot_qubit_state_revival(self, is_fitting=True):

        self.qubit_state_revival_data = self.qubit_state_revival_results.get_data("qubit_state_revival")

        if self.which_data == "I":
            data = np.real(self.qubit_state_revival_data)
        else:
            data = np.imag(self.qubit_state_revival_data)

        delay_sweep_list = self.qubit_state_revival_results.acquired_results['qubit_state_revival'].axis[0]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        cavity_drive_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        cavity_drive_length = cavity_parameters[cavity_component]["cavity_drive_length"]

        ############# Plotting the data #####################

        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        fig.suptitle(f"Qubit state revival using Ramsey interferometry", fontsize=18)
        
        ax.plot(delay_sweep_list, data, marker='o', linestyle=':', color='k')

        an = ax.annotate(f'cavity_drive_amp = {cavity_drive_amp:.4f}, cavity_drive_length = {cavity_drive_length*1e9:.0f}ns',
                        xy=(np.average(delay_sweep_list), np.average(data)),
                        size=12)
        an.draggable()

        ax.set_xlabel('Wait time (s)')
        ax.set_ylabel(f'[{self.which_data}] (a.u.)')

        plt.show()

# In[] storage mode characterization

    def storage_mode_characterization(self, average_exponent=12, 
                                    wait_time = 50e-6,
                                    wait_npts = 101,
                                    detuning = 0*1e6,
                                    init_state = "g",
                                    is_plot_simulation=False):
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        cond_disp_pulse, cavity_drive_pulse, _, _ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Sweep parameters
        delay_sweep = LinearSweepParameter(uid="delay_sweep", start=0, stop=wait_time, count=wait_npts)

        sweep_case = LinearSweepParameter(uid="correction", start=0, stop=1, count=2)

        def correction(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)

            return decorator

        # Define the experiment
        exp_storage_mode_characterization = Experiment(
            uid="storage_mode_characterization",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_storage_mode_characterization.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True
        ):
            with exp_storage_mode_characterization.sweep(uid="correction", parameter=sweep_case, reset_oscillator_phase=True):
                with exp_storage_mode_characterization.sweep(uid="delay_sweep", parameter=delay_sweep, reset_oscillator_phase=True):
                    with exp_storage_mode_characterization.section(uid="displacement", alignment=SectionAlignment.RIGHT):
                        exp_storage_mode_characterization.play(signal="cavity_drive", pulse=cavity_drive_pulse)
                        if init_state == "e":
                            exp_storage_mode_characterization.play(signal="drive", pulse=pi_pulse)
                                  
                    with exp_storage_mode_characterization.section(uid="delay_drive", play_after="displacement"):
                        exp_storage_mode_characterization.delay(signal="cavity_drive", time=delay_sweep)
                        exp_storage_mode_characterization.delay(signal="drive", time=delay_sweep)
                        exp_storage_mode_characterization.play(signal="drive", pulse=pi2_pulse)
                    
                    self.CNOD(exp = exp_storage_mode_characterization, cond_disp_pulse = cond_disp_pulse,
                                pi_pulse = pi_pulse, amp = 1, # amp of asym pulse is defined at "cond_disp_pulse_amp" 
                                prev_uid="delay_drive", uid2 = "cond_disp_pulse_2")

                    with exp_storage_mode_characterization.section(uid="qubit_excitation_2", play_after="cond_disp_pulse_2"):
                        @correction(sweep_case, exp=exp_storage_mode_characterization)
                        def play_correction(v):
                            if v == 0:
                                exp_storage_mode_characterization.play(signal="drive", pulse=pi2_pulse, phase = 0)
                            elif v == 1:
                                exp_storage_mode_characterization.play(signal="drive", pulse=pi2_pulse, phase = np.pi)
                    
                    with exp_storage_mode_characterization.section(uid="measure", play_after="qubit_excitation_2"):
                        exp_storage_mode_characterization.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_storage_mode_characterization.acquire(signal="acquire",
                                                    handle="storage_mode_characterization",
                                                    kernel=readout_weighting_function)
                    with exp_storage_mode_characterization.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_storage_mode_characterization.reserve(signal="measure")
        
        exp_calibration = Calibration()
        
        cavity_mode_oscillator = Oscillator(
            "cavity_drive_if_osc",
            frequency=self.cavity_parameters[cavity_component][f"{cavity_component}_freq_IF"] + detuning,
        )
        exp_calibration["cavity_drive"] = SignalCalibration( # experimental signal line 이름으로 signal calibration : 해당 실험 일시적 적용
            oscillator=cavity_mode_oscillator
        )
        exp_storage_mode_characterization.set_calibration(exp_calibration)

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_storage_mode_characterization.set_signal_map(signal_map)
        compiled_exp_storage_mode_characterization = self.session.compile(exp_storage_mode_characterization)
        self.storage_mode_characterization_results = self.session.run(compiled_exp_storage_mode_characterization)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_storage_mode_characterization, start_time=0, length=20e-6)
            show_pulse_sheet("storage_mode_characterization", compiled_exp_storage_mode_characterization)

# init_guess = [amplitude,omega,T1,detuning,offset]
    def plot_storage_mode_characterization(self, is_fit=True, init_guess=None):
        self.storage_mode_characterization_data = self.storage_mode_characterization_results.get_data("storage_mode_characterization")

        if self.which_data == "I":
            data = np.real(self.storage_mode_characterization_data)
        else:
            data = np.imag(self.storage_mode_characterization_data)

        delay_sweep_list = self.storage_mode_characterization_results.acquired_results['storage_mode_characterization'].axis[1]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        cnod_amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        cavity_drive_length = cavity_parameters[cavity_component]["cavity_drive_length"]
        cavity_drive_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]

        ############# Plotting the data #####################

        fig, ax = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle(f"Storage mode characterization", fontsize=18)
        
        ax[0].plot(delay_sweep_list, data[0], marker='o', linestyle=':', color='k', label='uncorrected')
        ax[0].plot(delay_sweep_list, data[1], marker='o', linestyle=':', color='r', label='correction')
        ax[1].plot(delay_sweep_list, data[0]-data[1], marker='o', linestyle=':', color='k', label='corrected')


        an = ax[1].annotate(f'cond_disp_pulse_amp = {cnod_amp:.4f}, disp_pulse_amp = {cavity_drive_amp:.4f}',
                        xy=(np.average(delay_sweep_list), np.average(data[0]-data[1])),
                        size=12)
        an.draggable()

        if is_fit :

            sfit1 = sFit('Storage_Characterization', delay_sweep_list, data[0]-data[1], init_guess=init_guess)
            # Scale the amplitude into alpha (photon number) by using alpha_1_CNOD_amp
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            print("popt:", popt)
            print("pcov:", pcov)

            ax[1].plot(delay_sweep_list, 
                    sfit1.func(delay_sweep_list, *popt), label='fit', color='g')

            an = ax[1].annotate(f'freq = {popt[3]:.4f}±{np.sqrt(np.diag(pcov))[3]:.4f}Hz'+'\n'
                                + f'omega = {popt[1]:.4f}±{np.sqrt(np.diag(pcov))[1]:.4f}'+'\n'
                                + f'T1 = {popt[2]:.4f}±{np.sqrt(np.diag(pcov))[2]:.4f}s',
                            xy=(np.average(delay_sweep_list), np.average(data[0]-data[1])*0.95))
            an.draggable()

        ax[0].set_xlabel('Wait time (s)')
        ax[0].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[0].legend()
        ax[1].set_xlabel('Wait time (s)')
        ax[1].set_ylabel(f'[{self.which_data}] (a.u.)')
        ax[1].legend()

        plt.show()

# In[] with post selection

    def wigner_characteristic_function_2D(self, average_exponent=12, 
                                    npts_x = 21,
                                    npts_y = 21,
                                    amplitude = 0,
                                    is_wigner_function=False,
                                    is_coherent_state=False,
                                    is_schrodinger_cat_state=False,
                                    is_schrodinger_cat_state_2=False,
                                    is_cat_state=False,
                                    is_cat_state_2=False,
                                    acquire_delay = 200e-9,
                                    alpha = 1, # for cat state
                                    beta = 1, # for cat state
                                    qubit_phase = 0,
                                    is_correction = False,
                                    is_plot_simulation=False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        self.exp_wigner_characteristic_function_amplitude = amplitude
        self.is_wigner_function = is_wigner_function

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse,_,_ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Create a 2D grid of complex amplitudes from (-amp_range-1j*amp_range) to (amp_range+1j*amp_range)
        real_vals = np.linspace(-0.5, 0.5, npts_x) # should not be over unity
        imag_vals = np.linspace(-0.5, 0.5, npts_y) # should not be over unity
        amplitude_grid = (real_vals[:, None] + 1j * imag_vals[None, :]).T
        
        self.amp_npts_x = npts_x
        self.amp_npts_y = npts_y

        # Sweep parameters
        amplitude_sweep = SweepParameter(uid="amp_sweep", values=amplitude_grid.flatten())

        if is_correction:
            sweep_case = LinearSweepParameter(uid="correction", start=0, stop=1, count=2)
        else:
            sweep_case = LinearSweepParameter(uid="correction", start=0, stop=0, count=1)
    
        # Define the experiment
        exp_wigner_characteristic_function = Experiment(
            uid="wigner_function",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        def displaced_coherent_state():

            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):

                exp_wigner_characteristic_function.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude)
        
        def schrodinger_cat_state():
            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_wigner_characteristic_function.section(uid="qubit_preparation"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                with exp_wigner_characteristic_function.section(uid="CNOD", play_after="qubit_preparation"):
                    self.CNOD(exp=exp_wigner_characteristic_function, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=1,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None)

        def schrodinger_cat_state_2():
            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):

                with exp_wigner_characteristic_function.section(uid="alpha"):
                    exp_wigner_characteristic_function.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude)
                with exp_wigner_characteristic_function.section(uid="qubit", play_after="alpha"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                    exp_wigner_characteristic_function.delay(signal="drive", time=np.abs(1/(2*cavity_parameters[cavity_component]["cavity_mode_chi"])/2)) # delay for cross Kerr effect

        def cat_state(): # Asaf's method
            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_wigner_characteristic_function.section(uid="qubit_preparation"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                with exp_wigner_characteristic_function.section(uid="CNOD_1", play_after="qubit_preparation"):
                    self.CNOD(exp=exp_wigner_characteristic_function, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=alpha,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid = "pi_pulse_1")
                with exp_wigner_characteristic_function.section(uid="qubit_1", play_after="CNOD_1"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                with exp_wigner_characteristic_function.section(uid="CNOD_2", play_after="qubit_1"):
                    self.CNOD(exp=exp_wigner_characteristic_function, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=1j*beta,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid = "pi_pulse_2")
                with exp_wigner_characteristic_function.section(uid="qubit_2", play_after="CNOD_2"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse, phase=np.pi/2)
        
        def cat_state_2(): # Yvonnes' method
            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_wigner_characteristic_function.section(uid="qubit_drive"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                with exp_wigner_characteristic_function.section(uid="CNOD_1", play_after="qubit_drive"):
                    self.CNOD(exp=exp_wigner_characteristic_function, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=alpha,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid = "pi_pulse_1")
                with exp_wigner_characteristic_function.section(uid="qubit_drive_2", play_after="CNOD_1"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)

        with exp_wigner_characteristic_function.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True,
        ):
            with exp_wigner_characteristic_function.sweep(uid="correction", parameter=sweep_case):
                with exp_wigner_characteristic_function.sweep(uid="amplitude_sweep", parameter=amplitude_sweep, auto_chunking = True, reset_oscillator_phase=True):

                    if is_coherent_state:
                        displaced_coherent_state()
                    elif is_schrodinger_cat_state:
                        schrodinger_cat_state()
                    elif is_schrodinger_cat_state_2:
                        schrodinger_cat_state_2()
                    elif is_cat_state:
                        cat_state()
                    elif is_cat_state_2:
                        cat_state_2()

                    #### post-selection ####

                    with exp_wigner_characteristic_function.section(uid="post_measure", play_after="preparation"):
                        exp_wigner_characteristic_function.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_wigner_characteristic_function.acquire(signal="acquire",
                                                handle="post_selection",
                                                kernel=readout_weighting_function)
                    
                    with exp_wigner_characteristic_function.section(uid="acquire_delay", length=acquire_delay):
                        exp_wigner_characteristic_function.reserve(signal="acquire")


                    #### wigner function measurement ####

                    if is_wigner_function:

                        self.wigner_function(exp=exp_wigner_characteristic_function, cavity_drive_pulse=cavity_drive_pulse, 
                                            pi2_pulse=pi2_pulse, amplitude_sweep=amplitude_sweep, sweep_case=sweep_case, prev_uid="acquire_delay")
                    else:

                        self.characteristic_function(exp=exp_wigner_characteristic_function,
                                                    pi2_pulse=pi2_pulse, pi_pulse=pi_pulse, 
                                                    cond_disp_pulse=cond_disp_pulse,
                                                    amplitude_sweep=amplitude_sweep, qubit_phase = qubit_phase, prev_uid="acquire_delay")

                    with exp_wigner_characteristic_function.section(uid="measure", play_after="qubit_excitation_2"):
                        exp_wigner_characteristic_function.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_wigner_characteristic_function.acquire(signal="acquire",
                                                handle="wigner_characteristic_function_measurement",
                                                kernel=readout_weighting_function)
                        
                    with exp_wigner_characteristic_function.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_wigner_characteristic_function.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_wigner_characteristic_function.set_signal_map(signal_map)
        compiled_exp_wigner_characteristic_function = self.session.compile(exp_wigner_characteristic_function)
        self.wigner_characteristic_function_results = self.session.run(compiled_exp_wigner_characteristic_function)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_wigner_characteristic_function, start_time=0, length=20e-6)
            # show_pulse_sheet("wigner_characteristic_function", compiled_exp_wigner_characteristic_function)

    def plot_wigner_characteristic_function_measurement_2D(self, vmin, vmax, threshold, 
                                                           is_g_smaller_than_e=True, is_plot_G = False, is_plot_E = False,
                                                           is_wo_post_selection = False):

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        
        alpha_1_cavity_drive_amp = cavity_parameters[cavity_component]["alpha_1_cavity_drive_amp"]
        cavity_drive_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        cond_disp_pulse_amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        alpha_1_CNOD_amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"]

        length = cavity_parameters[cavity_component]["cavity_drive_length"]
        # amp = self.exp_wigner_function_amplitude * cavity_drive_amp

        self.post_selected_data = self.wigner_characteristic_function_results.get_data("post_selection")
        self.wigner_characteristic_function_data = self.wigner_characteristic_function_results.get_data("wigner_characteristic_function_measurement")

        if self.which_data == "I":
            post_selected_data = np.real(self.post_selected_data)
            data = np.real(self.wigner_characteristic_function_data)
        else:
            post_selected_data = np.imag(self.post_selected_data)
            data = np.imag(self.wigner_characteristic_function_data)

        ## post-selection and make G and E separate
        mask = (post_selected_data < threshold) # threshold for discriminating g and e state

        true_count = np.sum(mask)
        false_count = np.sum(~mask)

        # make new map depending on ground and excited state
        if is_g_smaller_than_e:
            _G = np.where(mask, data, np.nan)  
            _E = np.where(~mask, data, np.nan)

            print(f"Ground state ratio is : {true_count / (true_count + false_count):.4f}")
        else:
            _G = np.where(~mask, data, np.nan)
            _E = np.where(mask, data, np.nan)

            print(f"Ground state ratio is : {false_count / (true_count + false_count):.4f}")

        G = np.nanmean(_G, axis=0, keepdims=False)
        E = np.nanmean(_E, axis=0, keepdims=False)

        amplitude_sweep_list = self.wigner_characteristic_function_results.acquired_results['wigner_characteristic_function_measurement'].axis[2]

        if self.is_wigner_function:
            Z = amplitude_sweep_list.reshape((self.amp_npts_y, self.amp_npts_x))
            x = np.real(Z) * cavity_drive_amp/alpha_1_cavity_drive_amp # scaling to alpha (photon number)
            y = np.imag(Z) * cavity_drive_amp/alpha_1_cavity_drive_amp   # scaling to alpha (photon number)
            G = G.reshape((self.amp_npts_y, self.amp_npts_x))
            E = E.reshape((self.amp_npts_y, self.amp_npts_x))

        else:
            Z = amplitude_sweep_list.reshape((self.amp_npts_y, self.amp_npts_x))
            x = np.real(Z) * cond_disp_pulse_amp/alpha_1_CNOD_amp # scaling to alpha (photon number)
            y = np.imag(Z) * cond_disp_pulse_amp/alpha_1_CNOD_amp   # scaling to alpha (photon number)
            G = G.reshape((self.amp_npts_y, self.amp_npts_x))
            E = E.reshape((self.amp_npts_y, self.amp_npts_x))
        
        if is_plot_G:

            plt.figure(figsize=(10, 10))
            pcm = plt.pcolormesh(x, y, G, shading='auto', cmap='bwr', vmin = vmin, vmax = vmax)
            plt.colorbar(pcm, label=f'[{self.which_data}] (a.u.)')
            if self.is_wigner_function:
                plt.title(f"Wigner function (G)")
            else:
                plt.title(f"Characteristic function (G)")
            plt.xlabel(r'Re($\alpha$)')
            plt.ylabel(r'Im($\alpha$)')

        if is_plot_E:

            plt.figure(figsize=(10, 10))
            pcm = plt.pcolormesh(x, y, E, shading='auto', cmap='bwr', vmin = vmin, vmax = vmax)
            plt.colorbar(pcm, label=f'[{self.which_data}] (a.u.)')
            if self.is_wigner_function:
                plt.title(f"Wigner function (E)")
            else:
                plt.title(f"Characteristic function (E)")
            plt.xlabel(r'Re($\alpha$)')
            plt.ylabel(r'Im($\alpha$)')
        
        if is_wo_post_selection:

            data_wo_post_selection = np.average(data, axis=0, keepdims=False)

            data_wo_post_selection = data_wo_post_selection.reshape((self.amp_npts_y, self.amp_npts_x))

            plt.figure(figsize=(10, 10))
            pcm = plt.pcolormesh(x, y, data_wo_post_selection, shading='auto', cmap='bwr', vmin = vmin, vmax = vmax)
            plt.colorbar(pcm, label=f'[{self.which_data}] (a.u.)')
            if self.is_wigner_function:
                plt.title(f"Wigner function (E)")
            else:
                plt.title(f"Characteristic function (E)")
            plt.xlabel(r'Re($\alpha$)')
            plt.ylabel(r'Im($\alpha$)')

# In[] without post selection
    def _wigner_characteristic_function_2D(self, average_exponent=12, 
                                    npts_x = 21,
                                    npts_y = 21,
                                    amplitude = 0,
                                    wait_length = 0,
                                    is_wigner_function=False,
                                    is_coherent_state=False,
                                    is_schrodinger_cat_state=False,
                                    is_schrodinger_cat_state_2=False,
                                    is_cat_state=False,
                                    alpha = 1, # for cat state
                                    beta = 1, # for cat state
                                    qubit_phase = 0,
                                    is_autochunking = False,
                                    is_correction = False,
                                    is_plot_simulation=False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        self.exp_wigner_characteristic_function_amplitude = amplitude
        self.is_wigner_function = is_wigner_function

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse,_,_ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        # Create a 2D grid of complex amplitudes from (-amp_range-1j*amp_range) to (amp_range+1j*amp_range)
        real_vals = np.linspace(-0.5, 0.5, npts_x) # should not be over unity
        imag_vals = np.linspace(-0.5, 0.5, npts_y) # should not be over unity
        amplitude_grid = (real_vals[:, None] + 1j * imag_vals[None, :]).T
        
        self.amp_npts_x = npts_x
        self.amp_npts_y = npts_y

        # Sweep parameters
        amplitude_sweep = SweepParameter(uid="amp_sweep", values=amplitude_grid.flatten())

        if is_correction:
            sweep_case = LinearSweepParameter(uid="correction", start=0, stop=1, count=2)
        else:
            sweep_case = LinearSweepParameter(uid="correction", start=0, stop=0, count=1)
    
        # Define the experiment
        exp_wigner_characteristic_function = Experiment(
            uid="wigner_function",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        def displaced_coherent_state():

            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):

                exp_wigner_characteristic_function.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude)
        
        def schrodinger_cat_state():
            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_wigner_characteristic_function.section(uid="qubit_preparation"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                with exp_wigner_characteristic_function.section(uid="CNOD", play_after="qubit_preparation"):
                    self.CNOD(exp=exp_wigner_characteristic_function, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=1,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None)

        def schrodinger_cat_state_2():
            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):

                with exp_wigner_characteristic_function.section(uid="alpha"):
                    exp_wigner_characteristic_function.play(signal="cavity_drive", pulse=cavity_drive_pulse, amplitude=amplitude)
                with exp_wigner_characteristic_function.section(uid="qubit", play_after="alpha"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                    exp_wigner_characteristic_function.delay(signal="drive", time=np.abs(1/(2*cavity_parameters[cavity_component]["cavity_mode_chi"])/2)) # delay for cross Kerr effect

        def cat_state():
            with exp_wigner_characteristic_function.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                with exp_wigner_characteristic_function.section(uid="qubit_preparation"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                with exp_wigner_characteristic_function.section(uid="CNOD_1", play_after="qubit_preparation"):
                    self.CNOD(exp=exp_wigner_characteristic_function, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=alpha,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid = "pi_pulse_1")
                with exp_wigner_characteristic_function.section(uid="qubit_1", play_after="CNOD_1"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse)
                with exp_wigner_characteristic_function.section(uid="CNOD_2", play_after="qubit_1"):
                    self.CNOD(exp=exp_wigner_characteristic_function, cond_disp_pulse=cond_disp_pulse,
                            pi_pulse=pi_pulse, amp=1j*beta,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                            prev_uid=None,
                            uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid = "pi_pulse_2")
                with exp_wigner_characteristic_function.section(uid="qubit_2", play_after="CNOD_2"):
                    exp_wigner_characteristic_function.play(signal="drive", pulse=pi2_pulse, phase=np.pi/2)


        with exp_wigner_characteristic_function.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True,
        ):
            # with exp_wigner_characteristic_function.sweep(uid="correction", parameter=sweep_case):
            with exp_wigner_characteristic_function.sweep(uid="amplitude_sweep", parameter=amplitude_sweep, auto_chunking = is_autochunking, reset_oscillator_phase=True):

                if is_coherent_state:
                    displaced_coherent_state()
                elif is_schrodinger_cat_state:
                    schrodinger_cat_state()
                elif is_schrodinger_cat_state_2:
                    schrodinger_cat_state_2()
                elif is_cat_state:
                    cat_state()

                #### wigner function measurement ####
                #### post-selection ####

                with exp_wigner_characteristic_function.section(uid="post_measure", play_after="preparation"):
                    exp_wigner_characteristic_function.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_wigner_characteristic_function.acquire(signal="acquire",
                                            handle="post_selection",
                                            kernel=readout_weighting_function)
                
                with exp_wigner_characteristic_function.section(uid="acquire_delay", length=wait_length):
                    exp_wigner_characteristic_function.reserve(signal="acquire")

                # with exp_wigner_characteristic_function.section(uid="wait", length = wait_length, play_after="preparation"):
                #     exp_wigner_characteristic_function.reserve(signal="cavity_drive")
                #     exp_wigner_characteristic_function.reserve(signal="drive")


                if is_wigner_function:

                    self.wigner_function(exp=exp_wigner_characteristic_function, cavity_drive_pulse=cavity_drive_pulse, 
                                        pi2_pulse=pi2_pulse, amplitude_sweep=amplitude_sweep, sweep_case=sweep_case,
                                        prev_uid="acquire_delay")
                else:

                    self.characteristic_function(exp=exp_wigner_characteristic_function,
                                                pi2_pulse=pi2_pulse, pi_pulse=pi_pulse, 
                                                cond_disp_pulse=cond_disp_pulse,
                                                amplitude_sweep=amplitude_sweep, qubit_phase = qubit_phase,
                                                prev_uid="acquire_delay")

                with exp_wigner_characteristic_function.section(uid="measure", play_after="qubit_excitation_2"):
                    exp_wigner_characteristic_function.play(signal="measure", pulse=readout_pulse, phase=phase)
                    exp_wigner_characteristic_function.acquire(signal="acquire",
                                            handle="wigner_characteristic_function_measurement",
                                            kernel=readout_weighting_function)
                    
                with exp_wigner_characteristic_function.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_wigner_characteristic_function.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_wigner_characteristic_function.set_signal_map(signal_map)
        compiled_exp_wigner_characteristic_function = self.session.compile(exp_wigner_characteristic_function)
        self.wigner_characteristic_function_results = self.session.run(compiled_exp_wigner_characteristic_function)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_wigner_characteristic_function, start_time=0, length=20e-6)
            # show_pulse_sheet("wigner_characteristic_function", compiled_exp_wigner_characteristic_function)

    def _plot_wigner_characteristic_function_measurement_2D(self, vmin, vmax):

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        
        alpha_1_cavity_drive_amp = cavity_parameters[cavity_component]["alpha_1_cavity_drive_amp"]
        cavity_drive_amp = cavity_parameters[cavity_component]["cavity_drive_amp"]
        cond_disp_pulse_amp = cavity_parameters[cavity_component]["cond_disp_pulse_amp"]
        alpha_1_CNOD_amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"]

        length = cavity_parameters[cavity_component]["cavity_drive_length"]
        # amp = self.exp_wigner_function_amplitude * cavity_drive_amp

        self.wigner_characteristic_function_data = self.wigner_characteristic_function_results.get_data("wigner_characteristic_function_measurement")
        
        # data = np.real(np.average(self.wigner_characteristic_function_data, axis=0, keepdims=True))

        # data = np.average(self.wigner_characteristic_function_data, axis=0, keepdims=False)

        if self.which_data == "I":
            data = np.real(np.mean(self.wigner_characteristic_function_data, axis = 0))
        else:
            data = np.imag(np.mean(self.wigner_characteristic_function_data, axis = 0))

        # averaged_nums = len(self.cavity_T1_results.acquired_results['cavity_T1'].axis[0])
        # # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

        # self.cavity_T1_data = self.cavity_T1_results.get_data("cavity_T1") # (2^N, npts) array
        # # time = self.cavity_T1_results.acquired_results['cavity_T1'].axis[1]
        # time = self.cavity_T1_results.acquired_results['cavity_T1'].axis[2]


        # data_0 = np.real(np.mean(self.cavity_T1_data, axis = 0)[0]) # v ==0
        # data_1 = np.real(np.mean(self.cavity_T1_data, axis = 0)[1]) # v ==1
        # std_data_0 = np.real(np.std(self.cavity_T1_data, axis = 0)[0]/np.sqrt(averaged_nums))
        # std_data_1 = np.real(np.std(self.cavity_T1_data, axis = 0)[1]/np.sqrt(averaged_nums))

        # data = np.average(data, axis=0, keepdims=False)

        # if self.which_data == "I":
        #     data = np.real(self.wigner_characteristic_function_data)
        # else:
        #     data = np.imag(self.wigner_characteristic_function_data)
        

        amplitude_sweep_list = self.wigner_characteristic_function_results.acquired_results['wigner_characteristic_function_measurement'].axis[1]


        if self.is_wigner_function:
            Z = amplitude_sweep_list.reshape((self.amp_npts_y, self.amp_npts_x))
            x = np.real(Z) * cavity_drive_amp/alpha_1_cavity_drive_amp # scaling to alpha (photon number)
            y = np.imag(Z) * cavity_drive_amp/alpha_1_cavity_drive_amp   # scaling to alpha (photon number)
            data = data.reshape((self.amp_npts_y, self.amp_npts_x))

        else:
            Z = amplitude_sweep_list.reshape((self.amp_npts_y, self.amp_npts_x))
            x = np.real(Z) * cond_disp_pulse_amp/alpha_1_CNOD_amp # scaling to alpha (photon number)
            y = np.imag(Z) * cond_disp_pulse_amp/alpha_1_CNOD_amp   # scaling to alpha (photon number)
            data = data.reshape((self.amp_npts_y, self.amp_npts_x))

        # np.save(f"wigner_characteristic_function", self.wigner_characteristic_function_data)
        # np.save(f"Z_value", Z)
        ############# Plotting the data #####################

        plt.figure(figsize=(10, 10))
        pcm = plt.pcolormesh(x, y, data, shading='auto', cmap='bwr', vmin = vmin, vmax = vmax)
        plt.colorbar(pcm, label=f'[{self.which_data}] (a.u.)')
        if self.is_wigner_function:
            plt.title(f"Wigner function")
        else:
            plt.title(f"Characteristic function")
        plt.xlabel(r'Re($\alpha$)')
        plt.ylabel(r'Im($\alpha$)')

        plt.show()




############ sweep beta for calibration of cnod amplitude used for disentanglement of the cat
    def disentangling_power_sweep(self, average_exponent=12,
                                  alpha = 1,
                                  beta_sweep_start = -1,
                                  beta_sweep_stop = 1,
                                  beta_sweep_count = 21,
                                  is_xyz=False,
                                  is_plot_simulation=False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters
        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        pi2_pulse, pi_pulse, _ = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        cond_disp_pulse, cavity_drive_pulse,_,_ = self.pulse_generator("cavity_control", qubits_parameters, cavity_parameters,
                                                                   qubits_component, cavity_component)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        self.disentangling_power_sweep_alpha = alpha

        beta_sweep = LinearSweepParameter(uid="beta_sweep", start=beta_sweep_start, stop=beta_sweep_stop, count=beta_sweep_count)

        if is_xyz:
            sweep_case = LinearSweepParameter(uid="correction", start=0, stop=2, count=3)
            self.is_xyz = True
        else:
            sweep_case = LinearSweepParameter(uid="correction", start=0, stop=0, count=1)
        
        def _xyz(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for v in sweep_case.values:
                            with exp.case(v):
                                f(v)
    
            return decorator

        # Define the experiment
        exp_disentangling_power_sweep = Experiment(
            uid="disentangling_power_sweep",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("cavity_drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_disentangling_power_sweep.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase=True,
        ):
            with exp_disentangling_power_sweep.sweep(uid="xyz", parameter=sweep_case):
                with exp_disentangling_power_sweep.sweep(uid="amplitude_sweep", parameter=beta_sweep, auto_chunking = True, reset_oscillator_phase=True):

                    with exp_disentangling_power_sweep.section(uid="preparation", alignment=SectionAlignment.RIGHT):
                        with exp_disentangling_power_sweep.section(uid="qubit_preparation"):
                            exp_disentangling_power_sweep.play(signal="drive", pulse=pi2_pulse)
                        with exp_disentangling_power_sweep.section(uid="CNOD_1", play_after="qubit_preparation"):
                            self.CNOD(exp=exp_disentangling_power_sweep, cond_disp_pulse=cond_disp_pulse,
                                    pi_pulse=pi_pulse, amp=alpha,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                                    prev_uid=None,
                                    uid1 = "cond_disp_pulse_1", uid2 = "cond_disp_pulse_2", pi_pulse_uid = "pi_pulse_1")
                        with exp_disentangling_power_sweep.section(uid="qubit_1", play_after="CNOD_1"):
                            exp_disentangling_power_sweep.play(signal="drive", pulse=pi2_pulse)
                        with exp_disentangling_power_sweep.section(uid="CNOD_2", play_after="qubit_1"):
                            self.CNOD(exp=exp_disentangling_power_sweep, cond_disp_pulse=cond_disp_pulse,
                                    pi_pulse=pi_pulse, amp=1j*beta_sweep,  # amp of asym pulse is defined at "cond_disp_pulse_amp"
                                    prev_uid=None,
                                    uid1 = "cond_disp_pulse_3", uid2 = "cond_disp_pulse_4", pi_pulse_uid = "pi_pulse_2")
                        with exp_disentangling_power_sweep.section(uid="qubit_2", play_after="CNOD_2"):
                            @_xyz(sweep_case, exp=exp_disentangling_power_sweep)
                            def play_drive(v):
                                if v == 0: # X
                                    exp_disentangling_power_sweep.play(signal="drive", pulse=pi2_pulse, phase = 0)
                                elif v == 1: # Y
                                    exp_disentangling_power_sweep.play(signal="drive", pulse=pi2_pulse, phase = np.pi/2)
                                elif v == 2: # Z
                                    pass
               
                    with exp_disentangling_power_sweep.section(uid="measure", play_after="preparation"):
                        exp_disentangling_power_sweep.play(signal="measure", pulse=readout_pulse, phase=phase)
                        exp_disentangling_power_sweep.acquire(signal="acquire",
                                                handle="disentangling_power_sweep",
                                                kernel=readout_weighting_function)
                        
                    with exp_disentangling_power_sweep.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                        exp_disentangling_power_sweep.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_disentangling_power_sweep.set_signal_map(signal_map)
        compiled_exp_disentangling_power_sweep = self.session.compile(exp_disentangling_power_sweep)
        self.disentangling_power_sweep_results = self.session.run(compiled_exp_disentangling_power_sweep)
        if is_plot_simulation:
            self.plot_results(self.disentangling_power_sweep_results)
            self.simulation_plot(compiled_exp_disentangling_power_sweep, start_time=0, length=20e-6)
            show_pulse_sheet("disentangling_power_sweep", compiled_exp_disentangling_power_sweep)
    
    def plot_disentangling_power_sweep(self):

        self.disentangling_power_sweep_data = self.disentangling_power_sweep_results.get_data("disentangling_power_sweep")

        if self.which_data == "I":
            data = np.real(self.disentangling_power_sweep_data)
        else:
            data = np.imag(self.disentangling_power_sweep_data)

        beta_sweep_list = self.disentangling_power_sweep_results.acquired_results['disentangling_power_sweep'].axis[1]

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        cond_disp_pulse_amp = cavity_parameters[cavity_component]['cond_disp_pulse_amp']
        alpha_1_CNOD_amp = cavity_parameters[cavity_component]["alpha_1_CNOD_amp"]

        ############# Plotting the data #####################

        if self.is_xyz:
            fig, ax = plt.subplots(3, 1, figsize=(20, 12))
            fig.suptitle(f"disentangling power sweep, alpha = {cond_disp_pulse_amp/alpha_1_CNOD_amp*self.disentangling_power_sweep_alpha}", fontsize=18)

            ax[0].plot(cond_disp_pulse_amp/alpha_1_CNOD_amp*beta_sweep_list, data[0], marker='o', linestyle=':', color='k', label='<X>')
            ax[1].plot(cond_disp_pulse_amp/alpha_1_CNOD_amp*beta_sweep_list, data[1], marker='o', linestyle=':', color='k', label='<Y>')
            ax[2].plot(cond_disp_pulse_amp/alpha_1_CNOD_amp*beta_sweep_list, data[2], marker='o', linestyle=':', color='k', label='<Z>')

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
        
        else:
            fig, ax = plt.subplots(1, 1, figsize=(20, 12))
            fig.suptitle(f"disentangling power sweep, alpha = {cond_disp_pulse_amp/alpha_1_CNOD_amp*self.disentangling_power_sweep_alpha}", fontsize=18)

            ax.plot(cond_disp_pulse_amp/alpha_1_CNOD_amp*beta_sweep_list, data, marker='o', linestyle=':', color='k', label='<X>')

            ax.legend()

        fig.text(0.5, 0.04, 'Beta (a.u.)', ha='center', fontsize=16)