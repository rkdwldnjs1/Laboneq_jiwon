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

# Helpers:
from laboneq.contrib.example_helpers.generate_device_setup import (
    generate_device_setup_qubits,
)

# LabOne Q:
from laboneq.simple import *
from laboneq.dsl.experiment.builtins import *
from laboneq.analysis import calculate_integration_kernels_thresholds

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

class Basic_qubit_characterization_experiments(ZI_QCCS):

# In[] Time Domain Measurements
    
    def nopi_pi(self, average_exponent, phase = 0, is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        drive_pulse = pulse_library.drag(uid="drive_pulse", 
                                             length = qubits_parameters[component]['drive_pulse_length'], 
                                             amplitude = qubits_parameters[component]["drive_amp"],
                                             beta = qubits_parameters[component]["drive_beta"],
                                             )
        

        phase = phase * np.pi / 180
        # nopi --------------------------------------------------------------------------------------------
        exp_nopi = Experiment(
                uid="nopi_experiment",
                signals=[
                    ExperimentSignal(uid="drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
            
        with exp_nopi.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.RAW,
            reset_oscillator_phase = True,
        ):
            with exp_nopi.section(uid="drive"):
                exp_nopi.reserve(signal="drive")
            with exp_nopi.section(uid="measure", play_after="drive"):
                exp_nopi.play(signal="measure", pulse=readout_pulse, phase = phase)
                exp_nopi.acquire(
                    signal="acquire", handle="ac_nopi", kernel=readout_weighting_function # can be acquired only there is a measure signal
                )
            # relax time after readout - for signal processing and qubit relaxation to ground state
            with exp_nopi.section(uid="relax_nopi", length=qubits_parameters[component]["reset_delay_length"]):
                exp_nopi.reserve(signal="measure")

        signal_map = self.signal_map(component)

        exp_nopi.set_signal_map(signal_map)
        
        compiled_experiment_nopi = self.session.compile(exp_nopi)

        self.nopi_results = self.session.run(compiled_experiment_nopi)


        # pi --------------------------------------------------------------------------------------------

        exp_pi = Experiment(
                uid="pi_experiment",
                signals=[
                    ExperimentSignal(uid="drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
            
        with exp_pi.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.RAW,
            reset_oscillator_phase = True,
        ):
            with exp_pi.section(uid="drive"):
                exp_pi.play(signal="drive", pulse=drive_pulse)
            with exp_pi.section(uid="measure", play_after="drive"):
                exp_pi.play(signal="measure", pulse=readout_pulse, phase = phase)
                exp_pi.acquire(
                    signal="acquire", handle="ac_pi", kernel=readout_weighting_function # can be acquired only there is a measure signal
                )
            # relax time after readout - for signal processing and qubit relaxation to ground state
            with exp_pi.section(uid="relax_pi", length=qubits_parameters[component]["reset_delay_length"]):
                exp_pi.reserve(signal="measure")

        signal_map = {
            "measure": device_setup.logical_signal_groups[component].logical_signals["measure"],
            "drive": device_setup.logical_signal_groups[component].logical_signals["drive"],
            "acquire": device_setup.logical_signal_groups[component].logical_signals["acquire"],
        }

        exp_pi.set_signal_map(signal_map)
        
        compiled_experiment_pi = self.session.compile(exp_pi)

        self.pi_results = self.session.run(compiled_experiment_pi)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_nopi, start_time=0, length=20e-6, component=component)
            self.simulation_plot(compiled_experiment_pi, start_time=0, length=20e-6, component=component)

    def single_shot_nopi_pi(self, npts_exponent, phase = 0, is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        drive_pulse = pulse_library.drag(uid="drive_pulse", 
                                             length = qubits_parameters[component]['drive_pulse_length'], 
                                             amplitude = qubits_parameters[component]["drive_amp"],
                                             beta = qubits_parameters[component]["drive_beta"],
                                             )
        
        phase = phase * np.pi / 180
        # nopi --------------------------------------------------------------------------------------------

        exp_ss_nopi = Experiment(
                uid="nopi_experiment",
                signals=[
                    ExperimentSignal(uid="drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
        
        with exp_ss_nopi.acquire_loop_rt(
            uid="shots",
            count=pow(2, npts_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION
        ):
            
            with exp_ss_nopi.section(uid="drive"):
                exp_ss_nopi.reserve(signal="drive")
            with exp_ss_nopi.section(uid="measure", play_after="drive"):
                exp_ss_nopi.play(signal="measure", pulse=readout_pulse, phase = phase)
                exp_ss_nopi.acquire(
                    signal="acquire", handle="ac_nopi", kernel=readout_weighting_function # can be acquired only there is a measure signal
                )
            # relax time after readout - for signal processing and qubit relaxation to ground state
            with exp_ss_nopi.section(uid="relax_nopi", length=qubits_parameters[component]["reset_delay_length"]):
                exp_ss_nopi.reserve(signal="measure")
        
        signal_map = self.signal_map(component)

        exp_ss_nopi.set_signal_map(signal_map)
        
        compiled_experiment_nopi = self.session.compile(exp_ss_nopi)

        nopi_results = self.session.run(compiled_experiment_nopi)

        self.ss_nopi_results = nopi_results.get_data("ac_nopi")

        # pi --------------------------------------------------------------------------------------------

        exp_ss_pi = Experiment(
                uid="pi_experiment",
                signals=[
                    ExperimentSignal(uid="drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
        
        with exp_ss_pi.acquire_loop_rt(
            uid="shots",
            count=pow(2, npts_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION
        ):
            with exp_ss_pi.section(uid="drive"):
                exp_ss_pi.play(signal="drive", pulse=drive_pulse)
            with exp_ss_pi.section(uid="measure", play_after="drive"):
                exp_ss_pi.play(signal="measure", pulse=readout_pulse, phase = phase)
                exp_ss_pi.acquire(
                    signal="acquire", handle="ac_pi", kernel=readout_weighting_function # can be acquired only there is a measure signal
                )
            # relax time after readout - for signal processing and qubit relaxation to ground state
            with exp_ss_pi.section(uid="relax_pi", length=qubits_parameters[component]["reset_delay_length"]):
                exp_ss_pi.reserve(signal="measure")
        
        signal_map = self.signal_map(component)

        exp_ss_pi.set_signal_map(signal_map)

        compiled_experiment_pi = self.session.compile(exp_ss_pi)

        pi_results = self.session.run(compiled_experiment_pi)

        self.ss_pi_results = pi_results.get_data("ac_pi")

        
    def plot_nopi_pi(self, npts = 100, num_of_bins = 300, is_gaussian_fit = True):

        self.raw_nopi = self.nopi_results.get_data("ac_nopi")
        self.raw_pi = self.pi_results.get_data("ac_pi")

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]

        ### data acquisition and processing ##########################################

        I_demod_nopi, Q_demod_nopi = self.IF_demodulation(np.real(self.raw_nopi), np.imag(self.raw_nopi), self.qubits_parameters[component]["readout_freq_IF"])
        I_demod_pi, Q_demod_pi = self.IF_demodulation(np.real(self.raw_pi), np.imag(self.raw_pi), self.qubits_parameters[component]["readout_freq_IF"])
        
        num_of_pnts = npts
        data_len = len(self.raw_nopi)
        num_avg_pnts = data_len // num_of_pnts

        time = np.linspace(0, data_len/2 - data_len/2/num_of_pnts, num_of_pnts) # 2GSa/s acquisition rate

        average_I_demod_nopi = [np.mean(I_demod_nopi[i:i+num_avg_pnts])* data_len for i in range(0, len(I_demod_nopi), num_avg_pnts)]
        average_Q_demod_nopi = [np.mean(Q_demod_nopi[i:i+num_avg_pnts])* data_len for i in range(0, len(Q_demod_nopi), num_avg_pnts)]
        
        average_I_demod_pi = [np.mean(I_demod_pi[i:i+num_avg_pnts])* data_len for i in range(0, len(I_demod_pi), num_avg_pnts)]
        average_Q_demod_pi = [np.mean(Q_demod_pi[i:i+num_avg_pnts])* data_len for i in range(0, len(Q_demod_pi), num_avg_pnts)]
        
        ### single shot data ##########################
        ss_nopi_results = self.ss_nopi_results
        ss_pi_results = self.ss_pi_results

        I_nopi_data = np.real(ss_nopi_results)
        Q_nopi_data = np.imag(ss_nopi_results)

        I_nopi_data_mean = np.mean(I_nopi_data)
        Q_nopi_data_mean = np.mean(Q_nopi_data)

        nopi_noise = np.var(np.sqrt((I_nopi_data-I_nopi_data_mean)**2 + 
                                    (Q_nopi_data-Q_nopi_data_mean)**2))

        I_pi_data = np.real(ss_pi_results)
        Q_pi_data = np.imag(ss_pi_results)

        I_pi_data_mean = np.mean(I_pi_data)
        Q_pi_data_mean = np.mean(Q_pi_data)

        SNR = np.sqrt((I_nopi_data_mean-I_pi_data_mean)**2 + (Q_nopi_data_mean-Q_pi_data_mean)**2)/np.sqrt(nopi_noise)

        #### figure plot ###############################################################

        measurement_length = qubits_parameters[component]["readout_pulse_length"]

        fig, ax = plt.subplots(2, 2, figsize=(16, 10))

        # fig.suptitle(f"nopi and pi \n r_amp:{qubits_parameters[component]["readout_amp"]}, r_freq:{qubits_parameters[component]["readout_frequency"]}", fontsize=20)
        fig.suptitle(f"\n r_amp:{qubits_parameters[component]["readout_amp"]}, r_freq:{qubits_parameters[component]["readout_frequency"]}", fontsize=20)
        fig.text(0.45, 0.965, "NoPi", fontsize=20, color = "blue")
        fig.text(0.49, 0.965, "and", fontsize=20, color = "black")
        fig.text(0.52, 0.965, "Pi", fontsize=20, color = "red")

        ax[0,0].scatter(time, average_I_demod_nopi, color="b", marker = 'o')
        ax[0,0].scatter(time, average_I_demod_pi, color="r", marker = 'o')
        ax[0,0].tick_params(axis='both', which='major', labelsize=10)
        ax[0,0].tick_params(axis='both', which='minor', labelsize=10)
        ax[0,0].set_xlabel("Time (ns)", fontsize=20)
        ax[0,0].set_ylabel("I (a.u.)", fontsize=20)

        ax[0,1].scatter(time, average_Q_demod_nopi, color="b", marker = 'o')
        ax[0,1].scatter(time, average_Q_demod_pi, color="r", marker = 'o')
        ax[0,1].tick_params(axis='both', which='major', labelsize=10)
        ax[0,1].tick_params(axis='both', which='minor', labelsize=10)
        ax[0,1].set_xlabel("Time (ns)", fontsize=20)
        ax[0,1].set_ylabel("Q (a.u.)", fontsize=20)

        axis_range = np.max([np.abs(average_I_demod_nopi), np.abs(average_Q_demod_nopi), np.abs(average_I_demod_pi), np.abs(average_Q_demod_pi)])
        ax[1,0].plot(average_I_demod_nopi, average_Q_demod_nopi, color="b", marker = 'o', markeredgecolor='black')
        ax[1,0].plot(average_I_demod_pi, average_Q_demod_pi, color="r", marker = 'o', markeredgecolor='black')
        ax[1,0].plot(I_nopi_data_mean, Q_nopi_data_mean, color="k", alpha = 0.3, marker = 'x')
        ax[1,0].plot(I_pi_data_mean, Q_pi_data_mean, color="k", alpha = 0.3, marker = 'x')

        ax[1,0].scatter(I_nopi_data, Q_nopi_data, color="b", alpha = 0.3, marker = '.')
        ax[1,0].scatter(I_pi_data, Q_pi_data, color="r", alpha = 0.3, marker = '.')
        ax[1,0].set_xlim([-np.max([np.sqrt(I_nopi_data**2 + Q_nopi_data**2), np.sqrt(I_pi_data**2 + Q_pi_data**2)])*1.3,
                    np.max([np.sqrt(I_nopi_data**2 + Q_nopi_data**2), np.sqrt(I_pi_data**2 + Q_pi_data**2)])*1.3])
        ax[1,0].set_ylim([-np.max([np.sqrt(I_nopi_data**2 + Q_nopi_data**2), np.sqrt(I_pi_data**2 + Q_pi_data**2)])*1.3,
                    np.max([np.sqrt(I_nopi_data**2 + Q_nopi_data**2), np.sqrt(I_pi_data**2 + Q_pi_data**2)])*1.3])
        ax[1,0].set_aspect('equal', 'box')

        ax[1,0].tick_params(axis='both', which='major', labelsize=10)
        ax[1,0].tick_params(axis='both', which='minor', labelsize=10)
        ax[1,0].set_xlabel("I (a.u.)", fontsize=20)
        ax[1,0].set_ylabel("Q (a.u.)", fontsize=20)

        an1 = ax[1,0].annotate((f'SNR = {SNR:.3f}'), xy = (0,0))
        an1.draggable()

        # np.save(f"nopi_pi_data_{self.which_qubit}", [I_nopi_data, Q_nopi_data, I_pi_data, Q_pi_data])

        if self.which_data == "I":
            nopi_data = I_nopi_data
            pi_data = I_pi_data
            IQ = "I"
        else:
            nopi_data = Q_nopi_data
            pi_data = Q_pi_data
            IQ = "Q"

        nopi_hist_data = ax[1,1].hist(nopi_data, bins = num_of_bins, color = "b", alpha = 0.5)
        pi_hist_data = ax[1,1].hist(pi_data, bins = num_of_bins, color = "r", alpha = 0.5)

        self.nopi_value = np.mean(nopi_data)
        self.pi_value = np.mean(pi_data)

        ax[1,1].plot([np.mean(nopi_data), np.mean(nopi_data)], [0, max([max(nopi_hist_data[0]),max(pi_hist_data[0])]) + 5], '-k')
        ax[1,1].plot([np.mean(pi_data), np.mean(pi_data)], [0, max([max(nopi_hist_data[0]),max(pi_hist_data[0])]) + 5], '--k')

        ax[1,1].tick_params(axis='both', which='major', labelsize=10)

        ax[1,1].set_xlabel(f"{IQ} (a.u.)", fontsize=20)
        ax[1,1].set_ylabel("Counts", fontsize=20)

        fdlty, thresh = self.hist_fidelity(nopi_data, pi_data)

        an2 = ax[1,1].annotate((f'Fidelity = {fdlty:.3f}'), xy = (np.average(nopi_data), max(nopi_hist_data[0])))
        an2.draggable()

        plt.show()

        if is_gaussian_fit:

            ground_state = np.mean(nopi_data)
            excited_state = np.mean(pi_data)

            nopi_h = max(nopi_hist_data[0])
            pi_h = max(pi_hist_data[0])

            plt.figure()
            plt.plot(nopi_hist_data[1][:-1], nopi_hist_data[0], color = "b", alpha = 0.5, label = "nopi")
            plt.plot(pi_hist_data[1][:-1], pi_hist_data[0], color = "r", alpha = 0.5, label = "pi")

            init_guess_1 = [nopi_h, ground_state, 0.1, 0, excited_state, 0.1]

            popt1, pcov1 = scipy.optimize.curve_fit(self.double_gaussian, nopi_hist_data[1][:-1], nopi_hist_data[0], p0=init_guess_1)
            plt.plot(nopi_hist_data[1][:-1], self.double_gaussian(nopi_hist_data[1][:-1], *popt1), color = "b", alpha = 0.5, label = "nopi_fit")
            plt.plot([popt1[1],popt1[1]], [0, nopi_h], color = "b", linestyle = "--", alpha = 0.5)

            init_guess_2 = [pi_h*0.2, ground_state, 0.1, pi_h, excited_state, 0.1]

            popt2, pcov2 = scipy.optimize.curve_fit(self.double_gaussian, pi_hist_data[1][:-1], pi_hist_data[0], p0=init_guess_2)
            plt.plot(pi_hist_data[1][:-1], self.double_gaussian(pi_hist_data[1][:-1], *popt2), color = "r", alpha = 0.5, label = "pi_fit")
            plt.plot([popt2[4],popt2[4]], [0, pi_h], color = "r", linestyle = "--", alpha = 0.5)
            # plt.annotate(f"ground state value :{popt1[1]:.2f}", xy = (popt1[1], nopi_h))
            # plt.annotate(f"excited state value :{popt2[4]:.2f}", xy = (popt2[4], pi_h))

            print(f"nopi : {popt1}")
            print(f"pi : {popt2}")
            print(f"Thermal population : {min(popt1[0]/popt1[3], popt1[3]/popt1[0])}")
            print(f"residual population : {min(popt2[0]/popt2[3], popt2[3]/popt2[0])}")

            plt.legend()


    def test_consecutive_measurements(self, npts_exponent, phase = 0, acquire_delay = 0,
                                      first_amp = 0,
                                      second_amp = 0, is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        drive_pulse = pulse_library.drag(uid="drive_pulse", 
                                             length = qubits_parameters[component]['drive_pulse_length'], 
                                             amplitude = qubits_parameters[component]["drive_amp"],
                                             beta = qubits_parameters[component]["drive_beta"],
                                             )
        
        phase = phase * np.pi / 180

        exp_ss_pi = Experiment(
                uid="pi_experiment",
                signals=[
                    ExperimentSignal(uid="drive"),
                    ExperimentSignal(uid="measure"),
                    ExperimentSignal(uid="acquire"),
                ],
            )
        
        with exp_ss_pi.acquire_loop_rt(
            uid="shots",
            count=pow(2, npts_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION
        ):
            
            with exp_ss_pi.section(uid="drive"):
                exp_ss_pi.play(signal="drive", pulse=drive_pulse, amplitude=first_amp)
            with exp_ss_pi.section(uid="measure_1", play_after="drive"):
                exp_ss_pi.play(signal="measure", pulse=readout_pulse, phase = phase)
                exp_ss_pi.acquire(
                    signal="acquire", handle="ac_pi_1", kernel=readout_weighting_function # can be acquired only there is a measure signal
                )
            if acquire_delay > 0:
                with exp_ss_pi.section(uid="acquire_delay", length=acquire_delay):
                    exp_ss_pi.reserve(signal="acquire")

            with exp_ss_pi.section(uid="drive_2", play_after="acquire_delay"):
                exp_ss_pi.play(signal="drive", pulse=drive_pulse, amplitude=second_amp)

            with exp_ss_pi.section(uid="measure_2", play_after="drive_2"):
                exp_ss_pi.play(signal="measure", pulse=readout_pulse, phase = phase)
                exp_ss_pi.acquire(
                    signal="acquire", handle="ac_pi_2", kernel=readout_weighting_function # can be acquired only there is a measure signal
                )

            # relax time after readout - for signal processing and qubit relaxation to ground state
            with exp_ss_pi.section(uid="relax_pi", length=qubits_parameters[component]["reset_delay_length"]):
                exp_ss_pi.reserve(signal="measure")
        
        signal_map = self.signal_map(component)

        exp_ss_pi.set_signal_map(signal_map)

        compiled_experiment_test_pi = self.session.compile(exp_ss_pi)

        test_pi_results = self.session.run(compiled_experiment_test_pi)

        self.test_pi_results_1 = test_pi_results.get_data("ac_pi_1")
        self.test_pi_results_2 = test_pi_results.get_data("ac_pi_2")

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_test_pi, start_time=0, length=20e-6, component=component)
            show_pulse_sheet("test_consecutive_measure", compiled_experiment_test_pi)

    
    def plot_test_consecutive_measure(self, num_of_bins = 300, threshold = 0):
        
        test_pi_results_1 = self.test_pi_results_1
        test_pi_results_2 = self.test_pi_results_2

        I_results_1 = np.real(test_pi_results_1)
        Q_results_1 = np.imag(test_pi_results_1)

        I_results_2 = np.real(test_pi_results_2)
        Q_results_2 = np.imag(test_pi_results_2)

        fig, ax = plt.subplots(1, 2, figsize=(16, 10))

        ax[0].scatter(I_results_1, Q_results_1, color="b", alpha = 0.3, marker = '.', label = "1st measurement")
        ax[0].scatter(I_results_2, Q_results_2, color="r", alpha = 0.3, marker = '.', label = "2nd measurement")
        ax[0].set_xlim([-np.max([np.sqrt(I_results_1**2 + Q_results_1**2), np.sqrt(I_results_2**2 + Q_results_2**2)])*1.3,
                    np.max([np.sqrt(I_results_1**2 + Q_results_1**2), np.sqrt(I_results_2**2 + Q_results_2**2)])*1.3])
        ax[0].set_ylim([-np.max([np.sqrt(I_results_1**2 + Q_results_1**2), np.sqrt(I_results_2**2 + Q_results_2**2)])*1.3,
                    np.max([np.sqrt(I_results_1**2 + Q_results_1**2), np.sqrt(I_results_2**2 + Q_results_2**2)])*1.3])
        ax[0].set_aspect('equal', 'box')

        ax[0].tick_params(axis='both', which='major', labelsize=10)
        ax[0].tick_params(axis='both', which='minor', labelsize=10)
        ax[0].set_xlabel("I (a.u.)", fontsize=20)
        ax[0].set_ylabel("Q (a.u.)", fontsize=20)

        ax[0].legend()

        if self.which_data == "I":
            data_1 = I_results_1
            data_2 = I_results_2
            IQ = "I"
        else:
            data_1 = I_results_2
            data_2 = Q_results_2
            IQ = "Q"

        nopi_hist_data = ax[1].hist(data_1, bins = num_of_bins, color = "b", alpha = 0.5)
        pi_hist_data = ax[1].hist(data_2, bins = num_of_bins, color = "r", alpha = 0.5)

        self.data_1 = np.mean(data_1)
        self.data_2 = np.mean(data_2)

        ax[1].plot([np.mean(data_1), np.mean(data_1)], [0, max([max(nopi_hist_data[0]),max(pi_hist_data[0])]) + 5], '-k')
        ax[1].plot([np.mean(data_2), np.mean(data_2)], [0, max([max(nopi_hist_data[0]),max(pi_hist_data[0])]) + 5], '--k')

        ax[1].tick_params(axis='both', which='major', labelsize=10)

        ax[1].set_xlabel(f"{IQ} (a.u.)", fontsize=20)
        ax[1].set_ylabel("Counts", fontsize=20)


# In[] T1 Measurement

    def T1(self, average_exponent = 12, duration = 100e-6, npts = 101, is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        self.exp_T1_dict = {
            "duration": duration,
            "npts": npts,
        }
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        drive_pulse = pulse_library.gaussian(uid="drive_pulse", 
                                             length = qubits_parameters[component]['drive_pulse_length'], 
                                             amplitude = qubits_parameters[component]["drive_amp"])
        
        phase = qubits_parameters[component]["readout_phase"]

        delay_sweep = LinearSweepParameter(uid="delay", start=0, stop=duration, count=npts)

        exp_T1 = Experiment(
            uid="T1 experiment",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        
        with exp_T1.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - sweep over delay between qubit excitation and readout pulse
            # right alignment makes sure the readout section follows a fixed timing, being the last element in each shot
            with exp_T1.sweep(
                uid="sweep", parameter=delay_sweep
            ):
                # qubit drive pulse followed by variable delay
                with exp_T1.section(uid="qubit_excitation", alignment=SectionAlignment.RIGHT):
                    exp_T1.play(signal="drive", pulse=drive_pulse)
                    exp_T1.delay(signal="drive", time=delay_sweep)
                # qubit readout pulse and data acquisition
                with exp_T1.section(uid="qubit_readout", play_after="qubit_excitation"):
                    # play readout pulse
                    exp_T1.play(signal="measure", pulse=readout_pulse, phase = phase)
                    # signal data acquisition
                    exp_T1.acquire(
                        signal="acquire",
                        handle="ac_T1",
                        kernel=readout_weighting_function,
                    )
                # relax time after readout - for signal processing and qubit relaxation to ground state
                with exp_T1.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_T1.reserve(signal="measure")
                
        
        signal_map = self.signal_map(component)
        
        exp_T1.set_signal_map(signal_map)
        
        compiled_experiment_T1 = self.session.compile(exp_T1)

        self.T1_results = self.session.run(compiled_experiment_T1)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_T1, start_time=0, length=20e-6, component=component)
            show_pulse_sheet("T1", compiled_experiment_T1)
    
    def plot_T1(self, is_fit = True):

        ### data processing ###############################################################

        averaged_nums = len(self.T1_results.acquired_results['ac_T1'].axis[0])
        # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

        self.T1_data = self.T1_results.get_data("ac_T1") # (2^N, npts) array
        time = self.T1_results.acquired_results['ac_T1'].axis[1]

        if self.which_data == "I":
            
            data = np.real(np.mean(self.T1_data, axis = 0))
            std_data = np.real(np.std(self.T1_data, axis = 0)/np.sqrt(averaged_nums))

        else:
            data = np.imag(np.mean(self.T1_data, axis = 0))
            std_data = np.imag(np.std(self.T1_data, axis = 0)/np.sqrt(averaged_nums))


        ### data plot ######################################################################

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        ax.errorbar(time*1e6, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))

        if is_fit :
            sfit1 = sFit('Exp', time, data)
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            _,decay_rate,_ = popt
            _,decay_rate_err,_ = np.sqrt(np.diag(pcov))

            ax.plot(time*1e6, sfit1.func(time, *popt))
            an = ax.annotate((f'T1 = {(1/decay_rate*1e6):.2f}±{(1/(decay_rate)**2*decay_rate_err*1e6):.2f}[us]'), 
                             xy = (np.average(time*1e6), np.average(data[0:10]) ),
                             size = 16)
            an.draggable()

        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=15)

        ax.set_title("T1 measurement", fontsize=20)
        ax.set_xlabel("Time (us)", fontsize=20)
        ax.set_ylabel(f"{self.which_data} (a.u.)", fontsize=20)

        plt.show()
# In[]            
    def Pi2_cal(self, average_exponent = 12, start = 0, npts = 12, is_plot_simulation = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        pi2_pulse = pulse_library.drag(
            uid="drive_pulse", 
            length = qubits_parameters[component]['pi2_length'], 
            amplitude = qubits_parameters[component]["pi2_amp"],
            beta = qubits_parameters[component]["pi2_beta"]
        )
        
        phase = qubits_parameters[component]["readout_phase"]

        pulse_count = LinearSweepParameter(uid="pulses", start = start, stop= start+npts-1, count=npts)
        
        def repeat(count: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                for _ in range(int(v)):
                                    f()
                else:
                    for _ in range(count):
                        f()

            return decorator
        
                            
        exp_pi2_cal = Experiment(
            uid="Pi2 calibration",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        
        
        with exp_pi2_cal.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_pi2_cal.sweep(uid="sweep", parameter = pulse_count, alignment=SectionAlignment.RIGHT):
                with exp_pi2_cal.section(uid="pi2_pulses", alignment=SectionAlignment.RIGHT):
                    @repeat(pulse_count, exp_pi2_cal)
                    def play_pi2():
                        exp_pi2_cal.play(signal = "drive", 
                                        pulse = pi2_pulse)
            
                with exp_pi2_cal.section(uid="measure", play_after="pi2_pulses"):
                    exp_pi2_cal.play(signal="measure", pulse=readout_pulse, phase = phase)
                    exp_pi2_cal.acquire(
                        signal="acquire", handle="ac_pi2_cal", kernel=readout_weighting_function
                    )
                
                with exp_pi2_cal.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_pi2_cal.reserve(signal="measure")
        
        signal_map = self.signal_map(component)
        
        exp_pi2_cal.set_signal_map(signal_map)
        
        compiled_experiment_pi2_cal = self.session.compile(exp_pi2_cal)

        self.pi2_cal_results = self.session.run(compiled_experiment_pi2_cal)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_pi2_cal, start_time=0, length=20e-6, component=component)
    
    def plot_Pi2_cal(self):

        qubits_parameters = self.qubits_parameters
        component = list(qubits_parameters.keys())[self.which_qubit]

        ### data processing ###############################################################

        averaged_nums = len(self.pi2_cal_results.acquired_results['ac_pi2_cal'].axis[0])
        pi2_nums = self.pi2_cal_results.acquired_results['ac_pi2_cal'].axis[1]

        self.pi2_cal_data = self.pi2_cal_results.get_data("ac_pi2_cal") # (2^N, npts) array

        if self.which_data == "I":
            data = np.real(np.mean(self.pi2_cal_data, axis = 0))
            std_data = np.real(np.std(self.pi2_cal_data, axis = 0)/np.sqrt(averaged_nums))

        else:
            data = np.imag(np.mean(self.pi2_cal_data, axis = 0))
            std_data = np.imag(np.std(self.pi2_cal_data, axis = 0)/np.sqrt(averaged_nums))


        ### data plot ######################################################################
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        middle_points = []
        middle_indexes = []
        for i in range(len(data)//2):
            middle_indexes.append(pi2_nums[0] + 2*i+1)
            middle_points.append(data[2*i+1])

        ax.errorbar(pi2_nums, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, 
                    ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
        ax.plot(middle_indexes, middle_points, color="r", marker="o", linestyle = '--', markeredgecolor='black')
        
        an = ax.annotate((f'pi2_amp = {qubits_parameters[component]["pi2_amp"]}, length = {qubits_parameters[component]["pi2_length"]*1e9}[ns]'),
                            xy = (0, np.average(data)),
                            size = 16)
        an.draggable()

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title("Pi/2 calibration", fontsize=20)
        ax.set_xlabel("Number of pi/2 pulses", fontsize=20)
        ax.set_ylabel(f"{self.which_data} (a.u.)", fontsize=20)

        plt.show()
    
    # In[]            
    def Pi_cal(self, average_exponent = 12, npts = 12, is_plot_simulation = False, is_cond_pulse = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        self.is_cond_pulse = is_cond_pulse
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        pi2_pulse = pulse_library.drag(
            uid="pi2_pulse", 
            length = qubits_parameters[component]['pi2_length'], 
            amplitude = qubits_parameters[component]["pi2_amp"],
            beta = qubits_parameters[component]["pi2_beta"]
        )

        if is_cond_pulse:
            pi_pulse = pulse_library.drag(
                uid="pi_pulse", 
                length = qubits_parameters[component]['cond_pi_length'], 
                amplitude = qubits_parameters[component]["cond_pi_amp"],
                beta = qubits_parameters[component]["cond_pi_beta"],
                conditional=True
            )
        else :
            pi_pulse = pulse_library.drag(
                uid="pi_pulse", 
                length = qubits_parameters[component]['pi_length'], 
                amplitude = qubits_parameters[component]["pi_amp"],
                beta = qubits_parameters[component]["pi_beta"]
            )

        phase = qubits_parameters[component]["readout_phase"]
        
        pulse_count = LinearSweepParameter(uid="pulses", start=0, stop=npts-1, count=npts)
        
        def repeat(count: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                if v%2 == 0:
                                    for _ in range(int(v)//2):
                                        f()
                                else:
                                    exp.play(signal = "drive", pulse = pi2_pulse)
                                    for _ in range(int(v)//2):
                                        f()
                                
            
            return decorator
        
        exp_pi_cal = Experiment(
            uid="Pi calibration",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        
        with exp_pi_cal.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_pi_cal.sweep(uid="sweep", parameter = pulse_count, alignment=SectionAlignment.RIGHT):
                with exp_pi_cal.section(uid="pi_pulses", alignment=SectionAlignment.RIGHT):
                    @repeat(pulse_count, exp_pi_cal)
                    def play_pi():
                        exp_pi_cal.play(signal = "drive", 
                                         pulse = pi_pulse)
                            
                with exp_pi_cal.section(uid="measure", play_after="pi_pulses"):
                    exp_pi_cal.play(signal="measure", pulse=readout_pulse, phase = phase)
                    exp_pi_cal.acquire(
                        signal="acquire", handle="ac_pi_cal", kernel=readout_weighting_function
                    )
                
                with exp_pi_cal.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_pi_cal.reserve(signal="measure")
        
        signal_map = self.signal_map(component)
        
        exp_pi_cal.set_signal_map(signal_map)
        
        compiled_experiment_pi_cal = self.session.compile(exp_pi_cal)

        self.pi_cal_results = self.session.run(compiled_experiment_pi_cal)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_pi_cal, start_time=0, length=20e-6, component=component)
    
    def plot_Pi_cal(self):

        qubits_parameters = self.qubits_parameters
        component = list(qubits_parameters.keys())[self.which_qubit]

        ### data processing ###############################################################

        averaged_nums = len(self.pi_cal_results.acquired_results['ac_pi_cal'].axis[0])
        pi_nums = self.pi_cal_results.acquired_results['ac_pi_cal'].axis[1]

        self.pi_cal_data = self.pi_cal_results.get_data("ac_pi_cal") # (2^N, npts) array

        if self.which_data == "I":
            
            data = np.real(np.mean(self.pi_cal_data, axis = 0))
            std_data = np.real(np.std(self.pi_cal_data, axis = 0)/np.sqrt(averaged_nums))

        else:
            data = np.imag(np.mean(self.pi_cal_data, axis = 0))
            std_data = np.imag(np.std(self.pi_cal_data, axis = 0)/np.sqrt(averaged_nums))


        ### data plot ######################################################################
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        middle_points = []
        middle_indexes = []
        for i in range(len(data)//2):
            middle_indexes.append(2*i+1)
            middle_points.append(data[2*i+1])

        ax.errorbar(pi_nums, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, 
                    ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
        ax.plot(middle_indexes, middle_points, color="r", marker="o", linestyle = '--', markeredgecolor='black')
        if self.is_cond_pulse:
            ax.set_title("Cond Pi calibration", fontsize=20)
        else:
            ax.set_title("Pi calibration", fontsize=20)
        ax.set_xlabel("Number of Pi pulses", fontsize=20)
        ax.set_ylabel(f'{self.which_data} (a.u.)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)

        if self.is_cond_pulse:
            an = ax.annotate((f'cond_pi_amp = {qubits_parameters[component]["cond_pi_amp"]}, length = {qubits_parameters[component]["cond_pi_length"]*1e9}[ns]'),
                            xy = (0, np.average(data)),
                            size = 16)
        else:
            an = ax.annotate((f'pi_amp = {qubits_parameters[component]["pi_amp"]}, length = {qubits_parameters[component]["pi_length"]*1e9}[ns]'),
                            xy = (0, np.average(data)),
                            size = 16)
        an.draggable()

        plt.show()
        
            
# In[]

    def Ramsey(self, detuning = 0, is_echo = False, n_pi_pulse = 1, qubit_phase = 0, # (phase : 0 CP, phase : pi/2 CPMG)
               average_exponent = 12, duration = 100e-6, npts = 101,
               is_zz_interaction = False,
               control_qubit = None,
               is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters

        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        if is_zz_interaction:
            control_component = list(qubits_parameters.keys())[control_qubit]
            control_drive_pulse_pi = pulse_library.gaussian(uid="control_drive_pulse", 
                                             length = qubits_parameters[control_component]['pi_length'], 
                                             amplitude = qubits_parameters[control_component]["pi_amp"])
        
        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        drive_pulse_pi2, drive_pulse_pi, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)
        
        
        time_sweep = LinearSweepParameter(uid="time_sweep", start=0, stop=duration, count=npts)

        phase = qubits_parameters[qubits_component]["readout_phase"]


        if is_zz_interaction:
            exp_ramsey = Experiment(
                uid="Ramsey experiment",
                signals=[
                    ExperimentSignal("drive"),
                    ExperimentSignal("control_drive"),
                    ExperimentSignal("measure"),
                    ExperimentSignal("acquire"),
                ],
            )
        else :
            exp_ramsey = Experiment(
                uid="Ramsey experiment",
                signals=[
                    ExperimentSignal("drive"),
                    ExperimentSignal("measure"),
                    ExperimentSignal("acquire"),
                ],
            )

        self.is_echo = is_echo
        self.n_pi_pulse = n_pi_pulse
        self.qubit_phase = qubit_phase
        
        def _CPMG(n_pi_pulse, qubit_phase):
            """CPMG sequence for Ramsey experiment"""
            if n_pi_pulse == 1:
                exp_ramsey.play(signal="drive", pulse=drive_pulse_pi2)
                exp_ramsey.delay(signal="drive", time=time_sweep/2)
                exp_ramsey.play(signal="drive", pulse=drive_pulse_pi, phase = qubit_phase)
                exp_ramsey.delay(signal="drive", time=time_sweep/2)
                exp_ramsey.play(signal="drive", 
                                pulse=drive_pulse_pi2,
                                phase = 2*np.pi*detuning*time_sweep)
            
            else :
                exp_ramsey.play(signal="drive", pulse=drive_pulse_pi2)
                exp_ramsey.delay(signal="drive", time=time_sweep/(2*n_pi_pulse))
                for _ in range(n_pi_pulse-1):
                    exp_ramsey.play(signal="drive", pulse=drive_pulse_pi, phase = qubit_phase)
                    exp_ramsey.delay(signal="drive", time=time_sweep/(n_pi_pulse))
                exp_ramsey.play(signal="drive", pulse=drive_pulse_pi, phase = qubit_phase)
                exp_ramsey.delay(signal="drive", time=time_sweep/(2*n_pi_pulse))
                exp_ramsey.play(signal="drive", 
                                pulse=drive_pulse_pi2,
                                phase = 2*np.pi*detuning*time_sweep)

        with exp_ramsey.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase = False,
        ):
            # inner loop - real time sweep of Ramsey time delays
            with exp_ramsey.sweep(
                uid="ramsey_sweep", parameter=time_sweep, alignment=SectionAlignment.RIGHT
            ):
                with exp_ramsey.section(uid="ZZ_interaction", alignment=SectionAlignment.RIGHT):
                    if is_zz_interaction:
                        exp_ramsey.play(signal="control_drive", pulse=control_drive_pulse_pi)
                    else:
                        pass
                # play qubit excitation pulse - pulse amplitude is swept
                with exp_ramsey.section(
                    uid="qubit_excitation", alignment=SectionAlignment.RIGHT, play_after="ZZ_interaction"
                ):  
                    if is_echo:
                        _CPMG(n_pi_pulse= n_pi_pulse, qubit_phase = qubit_phase)
                    else:
                        exp_ramsey.play(signal="drive", pulse=drive_pulse_pi2)
                        exp_ramsey.delay(signal="drive", time=time_sweep)
                        exp_ramsey.play(signal="drive", 
                                        pulse=drive_pulse_pi2, 
                                        phase = 2*np.pi*detuning*time_sweep)
                # readout pulse and data acquisition
                with exp_ramsey.section(
                    uid="readout_section", play_after="qubit_excitation"
                ):
                    # play readout pulse on measure line
                    exp_ramsey.play(signal="measure", pulse=readout_pulse, phase = phase)
                    # trigger signal data acquisition
                    exp_ramsey.acquire(
                        signal="acquire",
                        handle="ramsey",
                        kernel=readout_weighting_function,
                    )

                # relax time after readout - for qubit relaxation to groundstate and signal processing
                with exp_ramsey.section(uid="relax", length=qubits_parameters[qubits_component]["reset_delay_length"]):
                    exp_ramsey.reserve(signal="measure")
        
        if is_zz_interaction:
            signal_map_1 = self.signal_map(qubits_component)
            exp_ramsey.set_signal_map(signal_map_1)

            signal_map_2 = {
                "control_drive": device_setup.logical_signal_groups[control_component].logical_signals["drive_line"],
            }
            exp_ramsey.set_signal_map(signal_map_2)

        else:
            signal_map = self.signal_map(qubits_component)
            exp_ramsey.set_signal_map(signal_map)
        
        
        compiled_experiment_ramsey = self.session.compile(exp_ramsey)
        
        self.ramsey_results = self.session.run(compiled_experiment_ramsey)
        
        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_ramsey, start_time=0, length=20e-6, component=qubits_component)
            show_pulse_sheet("ramsey", compiled_experiment_ramsey)


    def Ramsey_with_photon(self, detuning = 0, is_echo = False,
               average_exponent = 12, duration = 100e-6, npts = 101,
               cavity_freq_detuning = 10e6,
               is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        cavity_parameters = self.cavity_parameters

        qubits_component = list(qubits_parameters.keys())[self.which_qubit]
        cavity_component = list(cavity_parameters.keys())[self.which_mode]

        self.cavity_freq_detuning = cavity_freq_detuning

        # Define pulses
        readout_pulse, readout_weighting_function = self.pulse_generator("readout", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        drive_pulse_pi2, drive_pulse_pi, cond_pi_pulse = self.pulse_generator("qubit_control", qubits_parameters, cavity_parameters, 
                        qubits_component, cavity_component)

        cavity_drive_constant_chunk = pulse_library.const(
            uid="cavity_drive_pulse_1",
            length=duration/npts,
            amplitude=cavity_parameters[cavity_component]["cavity_drive_amp"]
        )
        
        if is_echo:
            length = qubits_parameters[qubits_component]["pi_length"] + qubits_parameters[qubits_component]["pi2_length"]*2
        else :
            length = qubits_parameters[qubits_component]["pi2_length"]*2
        
        self.is_echo = is_echo

        cavity_drive_constant_chunk_2 = pulse_library.const(
            uid="cavity_drive_pulse_2",
            length=length,
            amplitude=cavity_parameters[cavity_component]["cavity_drive_amp"]
        )

        time_sweep = LinearSweepParameter(uid="time_sweep", start=0, stop=duration - duration/npts, count=npts)
        
        cavity_drive_length_sweep = LinearSweepParameter(uid="pulses", start=0, stop=npts-1, count=npts)

        phase = qubits_parameters[qubits_component]["readout_phase"]

        def repeat(count: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                for _ in range(int(v)):
                                    f()
                                    
                else:
                    for _ in range(count):
                        f()

            return decorator

        exp_ramsey_with_photon = Experiment(
                uid="Ramsey_photon experiment",
                signals=[
                    ExperimentSignal("drive"),
                    ExperimentSignal("measure"),
                    ExperimentSignal("acquire"),
                    ExperimentSignal("cavity_drive"),
                ],
            )

        with exp_ramsey_with_photon.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.SINGLE_SHOT,
            acquisition_type=AcquisitionType.INTEGRATION,
            reset_oscillator_phase = False,
        ):
            # inner loop - real time sweep of Ramsey time delays
            with exp_ramsey_with_photon.sweep(
                uid="ramsey_sweep", parameter=[time_sweep,cavity_drive_length_sweep], alignment=SectionAlignment.RIGHT
            ):
                with exp_ramsey_with_photon.section(uid="drive_section", alignment=SectionAlignment.RIGHT):
                    with exp_ramsey_with_photon.section(uid="cavity_drive_1"):
                        exp_ramsey_with_photon.play(signal = "cavity_drive", pulse = cavity_drive_constant_chunk_2) # pulse length for pi/2

                    with exp_ramsey_with_photon.section(uid="cavity_drive_2", play_after="cavity_drive_1"):
                        @repeat(cavity_drive_length_sweep, exp_ramsey_with_photon)
                        def play_cavity_drive():
                            exp_ramsey_with_photon.play(signal="cavity_drive", pulse=cavity_drive_constant_chunk)
                                                     # pulse length for delay time

                    # play qubit excitation pulse - pulse amplitude is swept
                    with exp_ramsey_with_photon.section(
                        uid="qubit_excitation", alignment=SectionAlignment.RIGHT
                    ):  
                        if is_echo:
                            exp_ramsey_with_photon.play(signal="drive", pulse=drive_pulse_pi2)
                            exp_ramsey_with_photon.delay(signal="drive", time=time_sweep/2)
                            exp_ramsey_with_photon.play(signal="drive", pulse=drive_pulse_pi)
                            exp_ramsey_with_photon.delay(signal="drive", time=time_sweep/2)
                            exp_ramsey_with_photon.play(signal="drive", 
                                            pulse=drive_pulse_pi2,
                                            phase = 2*np.pi*detuning*time_sweep)
                        else:
                            exp_ramsey_with_photon.play(signal="drive", pulse=drive_pulse_pi2)
                            exp_ramsey_with_photon.delay(signal="drive", time=time_sweep)
                            exp_ramsey_with_photon.play(signal="drive", 
                                            pulse=drive_pulse_pi2, 
                                            phase = 2*np.pi*detuning*time_sweep)
                
                # readout pulse and data acquisition
                with exp_ramsey_with_photon.section(
                    uid="readout_section", play_after="drive_section"
                ):
                    # play readout pulse on measure line
                    exp_ramsey_with_photon.play(signal="measure", pulse=readout_pulse, phase = phase)
                    # trigger signal data acquisition
                    exp_ramsey_with_photon.acquire(
                        signal="acquire",
                        handle="ramsey",
                        kernel=readout_weighting_function,
                    )

                # relax time after readout - for qubit relaxation to groundstate and signal processing
                with exp_ramsey_with_photon.section(uid="relax", length=cavity_parameters[cavity_component]["reset_delay_length"]):
                    exp_ramsey_with_photon.reserve(signal="measure")
        
        exp_calibration = Calibration()
        
        cavity_mode_oscillator = Oscillator(
            "cavity_drive_if_osc",
            frequency=self.cavity_parameters[cavity_component][f"{cavity_component}_freq_IF"] + cavity_freq_detuning,
        )
        exp_calibration["cavity_drive"] = SignalCalibration( # experimental signal line 이름으로 signal calibration : 해당 실험 일시적 적용
            oscillator=cavity_mode_oscillator,
            automute=True,
        )
        exp_ramsey_with_photon.set_calibration(exp_calibration)

        self.exp_calibration = exp_calibration

        signal_map = {
            "measure": device_setup.logical_signal_groups[qubits_component].logical_signals["measure"],
            "acquire": device_setup.logical_signal_groups[qubits_component].logical_signals["acquire"],
            "drive": device_setup.logical_signal_groups[qubits_component].logical_signals["drive"],
            "cavity_drive": device_setup.logical_signal_groups[cavity_component].logical_signals["cavity_drive_line"],
        }

        exp_ramsey_with_photon.set_signal_map(signal_map)
        compiled_exp_ramsey_with_photon = self.session.compile(exp_ramsey_with_photon)
        self.ramsey_with_photon_results = self.session.run(compiled_exp_ramsey_with_photon)
        if is_plot_simulation:
            self.simulation_plot(compiled_exp_ramsey_with_photon, start_time=0, length=20e-6)
            show_pulse_sheet("ramsey_with_photon", compiled_exp_ramsey_with_photon)

    
    def Ramsey_amp_sweep(self, detuning = 0, is_echo = False,
               average_exponent = 12, duration = 100e-6, npts = 101,
               cavity_freq_detuning = 10e6, amp_start = 0, amp_stop = 0.1, amp_npts = 11,
               kappa = 1e6, chi = 1e6,
               is_plot_figure = False,
               is_plot_simulation = False):

        def fit_func(x, A, B):
            return A*x**2 + B

        amp_values = np.linspace(amp_start, amp_stop, amp_npts)

        freq_detuning_list = []
        freq_err_list = []
        decay_rate_list = []
        decay_rate_err_list = []

        for amp in amp_values:
            self.cavity_parameters['m0']['cavity_drive_amp'] = amp
            self.Ramsey_with_photon(is_echo=is_echo, average_exponent=average_exponent,
                                    duration=duration, npts=npts, detuning=detuning,
                                    cavity_freq_detuning=cavity_freq_detuning,
                                    is_plot_simulation=is_plot_simulation)
            popt, popt_err = self.plot_Ramsey(is_ramsey_with_photon=True, is_fit=True, is_plot=is_plot_figure)
            
            freq_detuning_list.append((popt[1]-detuning)*1e-6) # [MHz]
            freq_err_list.append(popt_err[1]*1e-6) # [MHz]
            decay_rate_list.append(popt[2]*1e-6) # [MHz]
            decay_rate_err_list.append(popt_err[2]*1e-6) # [MHz]

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        popt_fit, pcov_fit = scipy.optimize.curve_fit(fit_func, amp_values, freq_detuning_list)

        ax[0].errorbar(amp_values, freq_detuning_list, yerr = freq_err_list, fmt = '--or', 
                       capsize = 5, markersize = 3, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
        ax[1].errorbar(amp_values, decay_rate_list, yerr = decay_rate_err_list, fmt = '--or',
                       capsize = 5, markersize = 3, ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
        ax[0].set_xlabel("Drive amplitude", fontsize=20)
        ax[0].set_ylabel("Frequency detuning (MHz)", fontsize=20)
        ax[0].set_title(f"{cavity_freq_detuning*1e-6}MHz away from center", fontsize=20)
        ax[0].plot(amp_values, fit_func(amp_values, *popt_fit), 'b--', label='Fit')
        ax[0].legend()

        an = ax[0].annotate(r"$\Delta f: AV^2 + B$"+'\n'\
                            +f'A = {popt_fit[0]:.4f} ± {np.sqrt(np.diag(pcov_fit))[0]:.4f} MHz/V^2' + '\n'\
                            +f'B = {popt_fit[1]:.4f} ± {np.sqrt(np.diag(pcov_fit))[1]:.4f} MHz',
                            xy=(0.0, 0.0), xycoords='axes fraction',
                            fontsize=12, ha='center', va='center', bbox=dict(boxstyle="round", fc="w"))
        
        an.draggable()
        
        A = popt_fit[0]

        Delta = cavity_freq_detuning

        a = np.sqrt( (A * (kappa**2 + chi**2)*(Delta**2+kappa**2/4))/(chi*kappa**2)  )

        an2 = ax[0].annotate(r"$\frac{\epsilon}{2\pi} = aV$"+'\n'\
                             + f"a={a:.4f} MHz/V",
                             xy=(0.0, 0.1), xycoords='axes fraction',
                             fontsize=12, ha='center', va='center', bbox=dict(boxstyle="round", fc="w"))

        an2.draggable()

        ax[1].set_xlabel("Drive amplitude", fontsize=20)
        ax[1].set_ylabel("Decay rate (MHz)", fontsize=20)
        # ax[1].set_title(f"Center from {cavity_freq_detuning*1e-6}MHz away", fontsize=20)
        ax[0].tick_params(axis='both', which='major', labelsize=16)
        ax[1].tick_params(axis='both', which='major', labelsize=16)




    def plot_Ramsey(self, is_ramsey_with_photon = False, is_fit = True, is_plot=True):

        cavity_parameters = self.cavity_parameters
        cavity_component = list(cavity_parameters.keys())[self.which_mode]
        
        ### data processing ###############################################################

        if is_ramsey_with_photon:
            results = self.ramsey_with_photon_results
            time = results.acquired_results['ramsey'].axis[1][0]

        else :
            results = self.ramsey_results
            time = results.acquired_results['ramsey'].axis[1]

        averaged_nums = len(results.acquired_results['ramsey'].axis[0])
        # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

        self.T2_data = results.get_data("ramsey") # (2^N, npts) array

        if self.which_data == "I":
            data = np.real(np.mean(self.T2_data, axis = 0))
            std_data = np.real(np.std(self.T2_data, axis = 0)/np.sqrt(averaged_nums))

        else:
            data = np.imag(np.mean(self.T2_data, axis = 0))
            std_data = np.imag(np.std(self.T2_data, axis = 0)/np.sqrt(averaged_nums))

        ### data plot ######################################################################
        if is_plot :

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.errorbar(time*1e6, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, 
                        ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))

            if is_fit :
                sfit1 = sFit('ExpCos', time, data)
                
                popt = sfit1._curve_fit()[0]
                pcov = sfit1._curve_fit()[1]

                _,freq,decay_rate,_,_ = popt
                _,freq_err,decay_rate_err,_,_ = np.sqrt(np.diag(pcov))

                # Var(1/X) = Var(X)/X^4 => std(1/X) = std(X)/X^2

                ax.plot(time*1e6, sfit1.func(time, *popt))
                an = ax.annotate((f'T2 = {(1/decay_rate*1e6):.2f}±{(1/(decay_rate)**2*decay_rate_err*1e6):.2f}[us], freq = {(freq*1e-6):.3f}±{(freq_err*1e-6):.3f}[MHz]'), 
                                    xy = (np.average(time), np.average(data)+(np.max(data)-np.min(data))*0.3),
                                    size = 16)
                an.draggable()
                ax.tick_params(axis='both', which='major', labelsize=16)

                if not is_ramsey_with_photon:
                    if self.is_echo:
                        if self.qubit_phase == 0:
                            ax.set_title(f"Ramsey measurement with CP : n_pi_pulse = {self.n_pi_pulse}", fontsize=20)
                        else:
                            ax.set_title(f"Ramsey measurement with CPMG : n_pi_pulse = {self.n_pi_pulse}", fontsize=20)

                    else:
                        ax.set_title("Ramsey measurement", fontsize=20)
                else :
                    if self.is_echo:
                        ax.set_title(f"Echo Ramsey measurement with photon (amp:{cavity_parameters[cavity_component]['cavity_drive_amp']}, cavity_detuning:{self.cavity_freq_detuning*1e-6}MHz)", fontsize=20)
                    else:
                        ax.set_title(f"Ramsey measurement with photon (amp:{cavity_parameters[cavity_component]['cavity_drive_amp']}, cavity_detuning:{self.cavity_freq_detuning*1e-6}MHz)", fontsize=20)

                ax.set_xlabel("Time (us)", fontsize=20)
                ax.set_ylabel(f'{self.which_data} (a.u.)', fontsize=20)
        
        return popt, np.sqrt(np.diag(pcov))
    
# In[] Rabi
    def Rabi_amplitude(self, average_exponent = 12, npts = 100, is_plot_simulation = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        
        drive_pulse = pulse_library.gaussian(uid="drive_pulse", 
                                             length = qubits_parameters[component]['drive_pulse_length'], 
                                             amplitude = qubits_parameters[component]["drive_amp"])
        
        amplitude_sweep = LinearSweepParameter(uid="amplitude_sweep", start=0, stop=1, count=npts)

        exp_rabi = Experiment(
            uid="Amplitude Rabi",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )
        
        with exp_rabi.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - real time sweep of Rabi ampitudes
            with exp_rabi.sweep(uid="rabi_sweep", parameter=amplitude_sweep):
                # play qubit excitation pulse - pulse amplitude is swept
                with exp_rabi.section(
                    uid="qubit_excitation", alignment=SectionAlignment.RIGHT
                ):
                    exp_rabi.play(
                        signal="drive", pulse=drive_pulse, amplitude=amplitude_sweep
                    )
                # readout pulse and data acquisition
                with exp_rabi.section(uid="readout_section", play_after="qubit_excitation"):
                    # play readout pulse on measure line
                    exp_rabi.play(signal="measure", pulse=readout_pulse)
                    # trigger signal data acquisition
                    exp_rabi.acquire(
                        signal="acquire",
                        handle="amp_rabi",
                        kernel=readout_weighting_function,
                    )
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                with exp_rabi.section(uid="reserve", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_rabi.reserve(signal="measure")
    
        signal_map = self.signal_map(component)
        
        exp_rabi.set_signal_map(signal_map)
        
        compiled_experiment_rabi = self.session.compile(exp_rabi)
        
        self.rabi_amp_results = self.session.run(compiled_experiment_rabi)
        
        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_rabi, start_time=0, length=20e-6, component=component)
    
    def plot_Rabi_amplitude(self):

        self.rabi_amp_data = self.rabi_amp_results.get_data("amp_rabi")
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))

        ax[0].plot(self.rabi_amp_results.acquired_results['amp_rabi'].axis[0], np.real(self.rabi_amp_data), color="r", marker="o")
        ax[1].plot(self.rabi_amp_results.acquired_results['amp_rabi'].axis[0], np.imag(self.rabi_amp_data), color="r", marker="o")

        plt.show()
            

# In[]
    def Rabi_length(self, average_exponent = 12, duration = 100e-6, npts = 100, is_single_shot = True, is_plot_simulation = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        self.exp_Rabi_length_dict = {
            "duration": duration,
            "npts": npts,
        }
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        
        rabi_drive_chunk = pulse_library.const(uid="drive_pulse", 
                                             length = duration/npts, 
                                             amplitude = qubits_parameters[component]["rabi_drive_amp"])
        
        ramp_up = pulse_library.gaussian_rise(uid="ramp_up", 
                                        length=qubits_parameters[component]["ramp_length"], 
                                        amplitude=qubits_parameters[component]["rabi_drive_amp"])
        ramp_down = pulse_library.gaussian_fall(uid="ramp_down", 
                                        length=qubits_parameters[component]["ramp_length"], 
                                        amplitude=qubits_parameters[component]["rabi_drive_amp"])
        
        rabi_length_sweep = LinearSweepParameter(uid="pulses", start=0, stop=npts-1, count=npts)

        
        def repeat(count: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                if v == 0:
                                    exp.play(signal="drive", pulse=ramp_up)
                                    exp.play(signal="drive", pulse=ramp_down)
                                else:
                                    exp.play(signal="drive", pulse=ramp_up)
                                    for _ in range(int(v)):
                                        f()
                                    exp.play(signal="drive", pulse=ramp_down)
                else:
                    for _ in range(count):
                        f()

            return decorator
        
        if is_single_shot :
            averaging_mode = AveragingMode.SINGLE_SHOT
            self.is_single_shot = True
        else:
            averaging_mode = AveragingMode.CYCLIC
            self.is_single_shot = False
        
        exp_rabi_length = Experiment(
            uid="Rabi_length",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_rabi_length.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=averaging_mode,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            
            with exp_rabi_length.sweep(uid="rabi_length_sweep", parameter= rabi_length_sweep, auto_chunking=True):
                with exp_rabi_length.section(uid="rabi_drives", alignment=SectionAlignment.RIGHT):
                    @repeat(rabi_length_sweep, exp_rabi_length)
                    def play_rabi():
                        exp_rabi_length.play(signal = "drive", 
                                            pulse = rabi_drive_chunk)
                                             
                # readout pulse and data acquisition
                with exp_rabi_length.section(uid="readout_section", play_after="rabi_drives"):
                    # play readout pulse on measure line
                    exp_rabi_length.play(signal="measure", pulse=readout_pulse, phase = qubits_parameters[component]["readout_phase"])
                    # trigger signal data acquisition
                    exp_rabi_length.acquire(
                        signal="acquire",
                        handle="rabi_length",
                        kernel=readout_weighting_function,
                    )
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                with exp_rabi_length.section(uid="reserve", length=qubits_parameters[component]["reset_delay_length"]):
                    exp_rabi_length.reserve(signal="measure")
    
        signal_map = self.signal_map(component)
        
        exp_rabi_length.set_signal_map(signal_map)
        
        compiled_experiment_rabi = self.session.compile(exp_rabi_length)
        
        self.rabi_length_results = self.session.run(compiled_experiment_rabi)
        
        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_rabi, start_time=0, length=20e-6, component=component)
            
    def plot_Rabi_length(self, is_fit = True):
        
        if self.is_single_shot:
        ### data processing ###############################################################

            averaged_nums = len(self.rabi_length_results.acquired_results['rabi_length'].axis[0])
            # npts = len(self.T1_results.acquired_results['ac_T1'].axis[1])

            self.rabi_length_data = self.rabi_length_results.get_data("rabi_length") # (2^N, npts) array
            # time = self.rabi_length_results.acquired_results['rabi_length'].axis[1]

            time = np.linspace(0, self.exp_Rabi_length_dict["duration"], self.exp_Rabi_length_dict["npts"])

            averaged_data = np.mean(self.rabi_length_data, axis = 0)

            if self.which_data == "I":
                data = np.real(averaged_data)
                std_data = np.real(np.std(self.rabi_length_data, axis = 0)/np.sqrt(averaged_nums))
            else:
                data = np.imag(averaged_data)
                std_data = np.imag(np.std(self.rabi_length_data, axis = 0)/np.sqrt(averaged_nums))

        ### data plot ######################################################################
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.errorbar(time*1e6, data, yerr = std_data, fmt = '--or', capsize = 5, markersize = 3, 
                        ecolor = 'k', mfc=(1,0,0,0.5), mec = (0,0,0,1))
        
        else:
        ### data processing ###############################################################
            self.rabi_length_data = self.rabi_length_results.get_data("rabi_length")

            if self.which_data == "I":
                data = np.real(self.rabi_length_data)
            else:
                data = np.imag(self.rabi_length_data)

            time = np.linspace(0, self.exp_Rabi_length_dict["duration"], self.exp_Rabi_length_dict["npts"])

        ### data plot ######################################################################

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.plot(time*1e6, data, color="r", marker="o", linestyle = '--', markeredgecolor='black')

        if is_fit :
            sfit1 = sFit('ExpCos', time, data)
            
            popt = sfit1._curve_fit()[0]
            pcov = sfit1._curve_fit()[1]

            _,freq,decay_rate,_,_ = popt
            _,freq_err,decay_rate_err,_,_ = np.sqrt(np.diag(pcov))

            # Var(1/X) = Var(X)/X^4 => std(1/X) = std(X)/X^2

            ax.plot(time*1e6, sfit1.func(time, *popt))
            an = ax.annotate((f'decay time = {(1/decay_rate*1e6):.2f}±{(1/(decay_rate)**2*decay_rate_err*1e6):.2f}[us], freq = {(freq*1e-6):.3f}±{(freq_err*1e-6):.3f}[MHz]'), 
                                xy = (np.average(time), np.average(data)+(np.max(data)-np.min(data))*0.3),
                                size = 16)
            an.draggable()
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_title("Rabi_length measurement", fontsize=20)
            ax.set_xlabel("Time (us)", fontsize=20)
            ax.set_ylabel(f'{self.which_data} (a.u.)', fontsize=20)

# In[] All XY

    def All_XY(self, average_exponent = 12, is_plot_simulation = False):

        self.AllXY_pulses = [["I", "I"],
                            ["X", "X"],
                            ["Y", "Y"],
                            ["X", "Y"],
                            ["Y", "X"],
                            ["X/2", "I"],
                            ["Y/2", "I"],
                            ["X/2", "Y/2"],
                            ["Y/2", "X/2"],
                            ["X/2", "Y"],
                            ["Y/2", "X"],
                            ["X", "Y/2"],
                            ["Y", "X/2"],
                            ["X/2", "X"],
                            ["X", "X/2"],
                            ["Y/2", "Y"],
                            ["Y", "Y/2"],
                            ["X", "I"],
                            ["Y", "I"],
                            ["X/2", "X/2"],
                            ["Y/2", "Y/2"]]
        
        AllXY_pulses = self.AllXY_pulses

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        pi2_pulse = pulse_library.drag(
            uid="pi2_pulse", 
            length = qubits_parameters[component]['pi2_length'], 
            amplitude = qubits_parameters[component]["pi2_amp"],
            beta = qubits_parameters[component]["pi2_beta"]
        )
        
        pi_pulse = pulse_library.drag(
            uid="pi_pulse", 
            length = qubits_parameters[component]['pi_length'], 
            amplitude = qubits_parameters[component]["pi_amp"],
            beta = qubits_parameters[component]["pi_beta"]
        )

        phase = qubits_parameters[component]["readout_phase"]

        def label_to_drag_play(name):
                if name == "I":
                    return False
                elif name == "X":
                    return pi_pulse, 0, {'beta': qubits_parameters[component]["pi_beta"]}
                elif name == "Y":
                    return pi_pulse, np.pi/2, {'beta': qubits_parameters[component]["pi_beta"]}
                elif name == "X/2":
                    return pi2_pulse, 0, {'beta': qubits_parameters[component]["pi2_beta"]}
                elif name == "Y/2":
                    return pi2_pulse, np.pi/2, {'beta': qubits_parameters[component]["pi2_beta"]}

        # Calculate the length of each experiment sections
        drive_section_length = (np.ceil((pi_pulse.length + pi2_pulse.length)/64e-9))*64e-9
        readout_section_length = (np.ceil(readout_pulse.length/64e-9))*64e-9
        reset_section_length = (np.ceil((qubits_parameters[component]["reset_delay_length"])/64e-9))*64e-9

        All_XY_sweep=LinearSweepParameter(uid="All_XY_sweep", start = 0, stop = len(AllXY_pulses)-1, count = len(AllXY_pulses))

        # define the sweep function
        def sweep_exp_cases(sweep_case: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for exp_case_num in sweep_case.values:
                            with exp.case(exp_case_num):
                                f(exp_case_num)
            return decorator
        

        exp_All_XY = Experiment(
            uid="All_XY",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_All_XY.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):

            with exp_All_XY.sweep(uid="sweep", parameter=All_XY_sweep, alignment=SectionAlignment.RIGHT):
                with exp_All_XY.section(uid="All_XY", length = drive_section_length, alignment=SectionAlignment.RIGHT):
                    @sweep_exp_cases(All_XY_sweep, exp = exp_All_XY)
                    def play_all_XY(exp_case_num):
                    
                        with exp_All_XY.section():
                            if not AllXY_pulses[round(exp_case_num)][0] == "I":
                                exp_All_XY.play(signal = "drive", 
                                                pulse = label_to_drag_play(AllXY_pulses[round(exp_case_num)][0])[0],
                                                phase = label_to_drag_play(AllXY_pulses[round(exp_case_num)][0])[1],
                                                pulse_parameters = label_to_drag_play(AllXY_pulses[round(exp_case_num)][0])[2],
                                                )
                            if not AllXY_pulses[round(exp_case_num)][1] == "I":
                                exp_All_XY.play(signal = "drive",
                                                pulse = label_to_drag_play(AllXY_pulses[round(exp_case_num)][1])[0],
                                                phase = label_to_drag_play(AllXY_pulses[round(exp_case_num)][1])[1],
                                                pulse_parameters = label_to_drag_play(AllXY_pulses[round(exp_case_num)][1])[2],
                                                )
                            if (AllXY_pulses[round(exp_case_num)][0] == "I") and (AllXY_pulses[round(exp_case_num)][1] == "I"):
                                exp_All_XY.reserve(signal = "drive")
                
                with exp_All_XY.section(uid="measure", play_after="All_XY", alignment=SectionAlignment.LEFT, length = readout_section_length):
                    exp_All_XY.play(signal="measure", pulse=readout_pulse, phase = phase)
                    exp_All_XY.acquire(
                        signal="acquire",
                        handle="All_XY",
                        kernel=readout_weighting_function,
                    )

                with exp_All_XY.section(uid="relax", length=reset_section_length):
                    exp_All_XY.reserve(signal="measure")

        signal_map = self.signal_map(component)

        exp_All_XY.set_signal_map(signal_map)

        compiled_experiment_All_XY = self.session.compile(exp_All_XY)

        self.All_XY_results = self.session.run(compiled_experiment_All_XY)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_All_XY, start_time=0, length=20e-6, component=component)
        
    def plot_All_XY(self):

        self.All_XY_data = self.All_XY_results.get_data("All_XY")
        
        if self.which_data == "I":
            data = np.real(self.All_XY_data)
        else:
            data = np.imag(self.All_XY_data)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        data_min = np.min(data)
        data_max = np.max(data)
        for grid_n in range(len(data)):
            ax.plot([np.arange(len(self.AllXY_pulses))[grid_n], np.arange(len(self.AllXY_pulses))[grid_n]], 
                    [data_min - 0.1*(data_max-data_min), data[grid_n]], linestyle='-', color='b', alpha=0.3)

        ax.plot(np.arange(len(self.AllXY_pulses)), data, label='I', marker='o', linestyle=':', color='k')
        
        # x axis labels: AllXY pulses names
        ax.set_xticks(np.arange(len(self.AllXY_pulses)))
        ax.set_xticklabels(self.AllXY_pulses)
        ax.set_title('AllXY data')
        ax.tick_params(axis='x', rotation=80, labelsize=10)

# In[]

    def drag_calibration(self, average_exponent = 12, beta_start = 0, beta_stop = 1, beta_count = 11, is_plot_simulation = False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters 
        component = list(qubits_parameters.keys())[self.which_qubit]
        
        ## define pulses used for experiment
        readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[component]["readout_pulse_length"], 
            amplitude=qubits_parameters[component]["readout_amp"], 
        )
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[component]["readout_integration_length"],
            amplitude=qubits_parameters[component]["readout_integration_amp"], 
        )
        
        pi2_pulse = pulse_library.drag(
            uid="pi2_pulse", 
            length = qubits_parameters[component]['pi2_length'], 
            amplitude = qubits_parameters[component]["pi2_amp"],
            beta = qubits_parameters[component]["pi2_beta"]
        )
        
        pi_pulse = pulse_library.drag(
            uid="pi_pulse", 
            length = qubits_parameters[component]['pi_length'], 
            amplitude = qubits_parameters[component]["pi_amp"],
            beta = qubits_parameters[component]["pi_beta"]
        )
        
        sweep_case = LinearSweepParameter(uid="sweep_case", start=0, stop=2, count=3)
        sweep_betas = LinearSweepParameter(uid="sweep_betas", start = beta_start, stop = beta_stop, count = beta_count)

        phase = qubits_parameters[component]["readout_phase"]
        # Calculate the length of each experiment sections
        drive_section_length = (np.ceil((pi_pulse.length + pi2_pulse.length)/64e-9))*64e-9
        readout_section_length = (np.ceil(readout_pulse.length/64e-9))*64e-9
        reset_section_length = (np.ceil((qubits_parameters[component]["reset_delay_length"])/64e-9))*64e-9

        def sweep_exp_cases(sweep_case: int | SweepParameter | LinearSweepParameter,
                            sweep_betas: int | SweepParameter | LinearSweepParameter,
                            exp):
            def decorator(f):
                if isinstance(sweep_case, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=sweep_case):
                        for exp_case_num in sweep_case.values:
                            with exp.case(exp_case_num):
                                with exp.match(sweep_parameter=sweep_betas):
                                    for sweep_value in sweep_betas.values:
                                        with exp.case(sweep_value):
                                            f(exp_case_num, sweep_value)

            return decorator
        
        exp_drag_calibration = Experiment(
            uid="drag_calibration",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with exp_drag_calibration.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_drag_calibration.sweep(uid="sweep", parameter= sweep_case):
                with exp_drag_calibration.sweep(uid="drag_sweep", parameter= sweep_betas):
                    with exp_drag_calibration.section(uid="drag_calibration", 
                                                      length = drive_section_length,
                                                      alignment=SectionAlignment.RIGHT):
                        @sweep_exp_cases(sweep_case, sweep_betas, exp = exp_drag_calibration)
                        def play_drag_calibration(exp_case_num, sweep_value):
                            
                            with exp_drag_calibration.section():
                                if exp_case_num == 0:
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi2_pulse,
                                        phase = 0, 
                                        pulse_parameters={"beta": sweep_value})
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi_pulse,
                                        phase = np.pi/2, 
                                        pulse_parameters={"beta": sweep_value})
                                elif exp_case_num == 1:
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi2_pulse,
                                        phase = 0, 
                                        pulse_parameters={"beta": sweep_value})
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi_pulse,
                                        phase = 0, 
                                        pulse_parameters={"beta": sweep_value})
                                elif exp_case_num == 2:
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi2_pulse,
                                        phase = 0, 
                                        pulse_parameters={"beta": sweep_value})
                                    exp_drag_calibration.play(signal = "drive", 
                                        pulse = pi_pulse,
                                        phase = -np.pi/2, 
                                        pulse_parameters={"beta": sweep_value})
                
                    with exp_drag_calibration.section(uid="measure", play_after="drag_calibration"):
                        exp_drag_calibration.play(signal="measure", pulse=readout_pulse, phase = phase)
                        exp_drag_calibration.acquire(
                            signal="acquire",
                            handle="drag_calibration",
                            kernel=readout_weighting_function,
                        )
                    with exp_drag_calibration.section(uid="relax", length=reset_section_length):
                        exp_drag_calibration.reserve(signal="measure")
        
        signal_map = self.signal_map(component)

        exp_drag_calibration.set_signal_map(signal_map)

        compiled_experiment_drag_calibration = self.session.compile(exp_drag_calibration)

        self.drag_calibration_results = self.session.run(compiled_experiment_drag_calibration)

        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_drag_calibration, start_time=0, length=20e-6, component=component)

    def plot_drag_calibration(self):

        self.drag_calibration_data = self.drag_calibration_results.get_data("drag_calibration")

        if self.which_data == "I":
            data = np.real(self.drag_calibration_data)
        else:
            data = np.imag(self.drag_calibration_data)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        beta_sweep = self.drag_calibration_results.acquired_results['drag_calibration'].axis[1]

        ax.plot(beta_sweep, data[0], label='Y(+pi)', marker='o', linestyle=':', color='r')
        ax.plot(beta_sweep, data[1], label='X(+pi)', marker='o', linestyle=':', color='m')
        ax.plot(beta_sweep, data[2], label='Y(-pi)', marker='o', linestyle=':', color='y')
        ax.set_title('beta sweep - I data')
        ax.set_xlabel('beta')
        ax.set_ylabel('I')
        ax.legend()

        # calculate crossing point
        dist_I_data = (data[0]-data[1])**2 + (data[1]-data[2])**2 + (data[2]-data[0])**2
        cross_idx = np.argmin(dist_I_data)
        ax.scatter([beta_sweep[cross_idx]],[data[0][cross_idx]], color='k', marker='s', s=100)
        ax.plot([beta_sweep[cross_idx],beta_sweep[cross_idx]],[np.min(np.concatenate((data[0],data[1],data[2]))), np.max(np.concatenate((data[0],data[1],data[2])))], 'k')
        ax.text(beta_sweep[cross_idx]*0.9+np.max(beta_sweep)*0.1,np.min(np.concatenate((data[0],data[1],data[2])))*0.9+np.max(np.concatenate((data[0],data[1],data[2])))*0.1,s=f"best beta={beta_sweep[cross_idx]:.6f}")

# In[]
    def error_amplification(self, average_exponent=12, pulse_npts=12, amp_npts=10, start_amp=0, end_amp=1,
                            is_drag_beta_calibration=False, 
                            is_plot_simulation=False):
        
        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        component = list(qubits_parameters.keys())[self.which_qubit]

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

        pi_pulse = pulse_library.drag(
            uid="pi_pulse",
            length=qubits_parameters[component]["pi_length"],
            amplitude=qubits_parameters[component]["pi_amp"],
            beta=qubits_parameters[component]["pi_beta"]
        )

        phase = qubits_parameters[component]["readout_phase"]

        # Sweep parameters
        pulse_count = LinearSweepParameter(uid="pulses", start=0, stop=pulse_npts - 1, count=pulse_npts)

        amp_sweep = LinearSweepParameter(uid="amplitude", start=start_amp, stop=end_amp, count=amp_npts)

        # Repeat decorator for custom looping
        def repeat(count, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                for _ in range(int(v)):
                                    f()
                else:
                    for _ in range(count):
                        f()

            return decorator

        # Define the experiment
        error_amplification = Experiment(
            uid="error_amplification",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        with error_amplification.acquire_loop_rt(
            uid="shots",
            count=2**average_exponent,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with error_amplification.sweep(uid="sweep1", parameter=pulse_count):
                with error_amplification.sweep(uid="sweep2", parameter=amp_sweep):

                    with error_amplification.section(uid="pi_section", alignment=SectionAlignment.RIGHT):
                        @repeat(pulse_count, error_amplification)
                        def play_pi():
                            if not is_drag_beta_calibration:
                                error_amplification.play(signal="drive", pulse=pi_pulse, 
                                                        amplitude=amp_sweep)
                                error_amplification.play(signal="drive", pulse=pi_pulse, 
                                                        amplitude=amp_sweep)
                            else:
                                error_amplification.play(signal="drive", pulse=pi_pulse,
                                                         phase = 0,
                                                        pulse_parameters={"beta": amp_sweep})
                                error_amplification.play(signal="drive", pulse=pi_pulse,
                                                         phase = np.pi,
                                                        pulse_parameters={"beta": amp_sweep})

                    with error_amplification.section(uid="measure", play_after="pi_section"):
                        error_amplification.play(signal="measure", pulse=readout_pulse, phase=phase)
                        error_amplification.acquire(
                            signal="acquire",
                            handle="ac_error_amplification",
                            kernel=readout_weighting_function
                        )

                    with error_amplification.section(uid="relax", length=qubits_parameters[component]["reset_delay_length"]):
                        error_amplification.reserve(signal="measure")

        signal_map = self.signal_map(component)
        error_amplification.set_signal_map(signal_map)

        compiled_error_amplification = self.session.compile(error_amplification)
        self.error_amplification_results = self.session.run(compiled_error_amplification)

        if is_plot_simulation:
            self.simulation_plot(compiled_error_amplification, start_time=0, length=40e-6, component=component)
        
    def plot_error_amplification(self, is_drag_beta_calibration=True):
        qubits_parameters = self.qubits_parameters
        component = list(qubits_parameters.keys())[self.which_qubit]

        self.error_amplification_data = self.error_amplification_results.get_data("ac_error_amplification")  # (2^N, npts) array
        data = np.real(self.error_amplification_data)

        x_vals = np.round(self.error_amplification_results.get_axis("ac_error_amplification")[1], 3)

        plt.figure(figsize=(14, 10))
        plt.imshow(data, cmap='viridis', origin='lower', aspect='auto')
        plt.title("Error amplification")

        step = 3  
        tick_positions = np.arange(0, len(x_vals), step)
        tick_labels = [x_vals[i] for i in tick_positions]

        plt.xticks(tick_positions, tick_labels, rotation=60)

        if is_drag_beta_calibration:
            plt.xlabel("drag beta")
        else:
            plt.xlabel("drive amplitude")

        plt.ylabel("# of Rabi pulses(pi x 2)")
        plt.colorbar(label='I value')
        plt.tight_layout()
        plt.show()

# In[] CR Calibration

    def cr_calibration_amp(self, average_exponent = 12, npts = 12, control_qubit = 0, target_qubit = 1, is_plot_simulation = False):

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        
        control_qubit_component = list(qubits_parameters.keys())[control_qubit]
        target_qubit_component = list(qubits_parameters.keys())[target_qubit]
        
        ## define pulses used for experiment
        target_qubit_readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[target_qubit_component]["readout_pulse_length"], 
            amplitude=qubits_parameters[target_qubit_component]["readout_amp"], 
        )
        
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        target_qubit_readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[target_qubit_component]["readout_integration_length"],
            amplitude=qubits_parameters[target_qubit_component]["readout_integration_amp"], 
        )
        
        
        target_qubit_drive_pulse = pulse_library.gaussian(uid="target_qubit_drive_pulse", 
                                length = qubits_parameters[target_qubit_component]['drive_pulse_length'], 
                                amplitude = qubits_parameters[target_qubit_component]["drive_amp"])
        
        control_qubit_drive_pulse = pulse_library.gaussian(uid="control_qubit_drive_pulse", 
                                length = qubits_parameters[control_qubit_component]['drive_pulse_length'], 
                                amplitude = qubits_parameters[control_qubit_component]["drive_amp"])
        
        amplitude_sweep = LinearSweepParameter(uid="amplitude_sweep", start=0, stop=1, count=npts)
        on_off_cases = LinearSweepParameter(uid="on_off_case", start=0, stop=1, count=2)

        phase = qubits_parameters[target_qubit_component]["readout_phase"]

        exp_cr = Experiment(
            uid="cr",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("drive_ef"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],
        )

        def on_off_control_drive(on_off_cases: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(on_off_cases, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=on_off_cases):
                        for v in on_off_cases.values:
                            with exp.case(v):
                                f(v)

            return decorator
        
   
        with exp_cr.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - real time sweep of Rabi ampitudes
            with exp_cr.sweep(uid="rabi_sweep", parameter=amplitude_sweep):
                # play qubit excitation pulse - pulse amplitude is swept
                with exp_cr.sweep(uid="on_off_case", parameter=on_off_cases):
                    with exp_cr.section(
                        uid="control_qubit_excitation", alignment=SectionAlignment.RIGHT
                    ):
                        
                        @on_off_control_drive(on_off_cases, exp = exp_cr)
                        def play_control_drive(on_off_case):
                            with exp_cr.section():
                                if on_off_case == 0:
                                    pass
                                else:
                                    exp_cr.play(signal="drive", pulse=control_qubit_drive_pulse)

                    with exp_cr.section(
                        uid="target_qubit_drive", play_after = "control_qubit_excitation"
                    ):
                        exp_cr.play(
                            signal="drive_ef", pulse=target_qubit_drive_pulse, amplitude=amplitude_sweep
                        )
                    # readout pulse and data acquisition
                    with exp_cr.section(uid="readout_section", play_after="target_qubit_drive"):
                        # play readout pulse on measure line
                        exp_cr.play(signal="measure", pulse=target_qubit_readout_pulse, phase = phase)
                        # trigger signal data acquisition
                        exp_cr.acquire(
                            signal="acquire",
                            handle="cr_caliibration",
                            kernel=target_qubit_readout_weighting_function,
                        )
                    # relax time after readout - for qubit relaxation to groundstate and signal processing
                    with exp_cr.section(uid="reserve", length=qubits_parameters[target_qubit_component]["reset_delay_length"]):
                        exp_cr.reserve(signal="measure")
    
        signal_map = self.signal_map(control_qubit_component, which_qubit = "control")
        exp_cr.set_signal_map(signal_map)
        signal_map = self.signal_map(target_qubit_component, which_qubit= "target")
        exp_cr.set_signal_map(signal_map)
        
        compiled_experiment_cr_calib_amp = self.session.compile(exp_cr)
        
        self.cr_calibration_results_amp = self.session.run(compiled_experiment_cr_calib_amp)
        
        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_cr_calib_amp, start_time=0, length=20e-6)

    def plot_cr_calibration_amp(self, is_fit = True):

        ### data processing ###############################################################

        control_off_data = self.cr_calibration_results_amp.get_data("cr_caliibration")[:,0]
        control_on_data = self.cr_calibration_results_amp.get_data("cr_caliibration")[:,1]

        amp_sweep = self.cr_calibration_results_amp.acquired_results['cr_caliibration'].axis[0]

        if self.which_data == "I":
            control_off_data = np.real(control_off_data)
            control_on_data = np.real(control_on_data)
        else:
            control_off_data = np.imag(control_off_data)
            control_on_data = np.imag(control_on_data)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(amp_sweep, control_off_data, color="b", marker="o", label = "control qubit off")
        ax.plot(amp_sweep, control_on_data, color="r", marker="o", label = "control qubit on")
        ax.set_title("CR calibration", fontsize=20)
        ax.set_xlabel("Amplitude", fontsize=20)
        ax.set_ylabel(f'{self.which_data} (a.u.)', fontsize=20)
        ax.legend()



    def cr_calibration_length(self, average_exponent = 12, npts = 100, duration = 1e-6, target_qubit_amp = 0.5,
                            control_qubit = 0, target_qubit = 1, is_plot_simulation = False):
        
        self.exp_cr_calibration_length = {
            "duration": duration,
            "target_qubit_amp": target_qubit_amp,
            "npts": npts,
        }

        device_setup = self.device_setup
        qubits_parameters = self.qubits_parameters
        
        control_qubit_component = list(qubits_parameters.keys())[control_qubit]
        target_qubit_component = list(qubits_parameters.keys())[target_qubit]
        
        ## define pulses used for experiment
        target_qubit_readout_pulse = pulse_library.gaussian_square(
            uid="readout_pulse", 
            length=qubits_parameters[target_qubit_component]["readout_pulse_length"], 
            amplitude=qubits_parameters[target_qubit_component]["readout_amp"], 
        )
        
        # readout integration weights - here simple square pulse, i.e. same weights at all times
        target_qubit_readout_weighting_function = pulse_library.gaussian_square(
            uid="readout_weighting_function", 
            length=qubits_parameters[target_qubit_component]["readout_integration_length"],
            amplitude=qubits_parameters[target_qubit_component]["readout_integration_amp"], 
        )
        
        control_qubit_drive_pi_pulse = pulse_library.gaussian(uid="drive_pulse", 
                                length = qubits_parameters[control_qubit_component]['pi_length'], 
                                amplitude = qubits_parameters[control_qubit_component]["pi_amp"])
        
        target_qubit_pi2_pulse = pulse_library.gaussian(uid="target_qubit_pulse", 
                                length = qubits_parameters[target_qubit_component]['pi2_length'], 
                                amplitude = qubits_parameters[target_qubit_component]["pi2_amp"])
        
        target_drive_chunk = pulse_library.const(uid="target_drive_chunk", 
                                             length = duration/npts, 
                                             amplitude = target_qubit_amp)
        
        ramp_up = pulse_library.gaussian_rise(uid="ramp_up", 
                                        length=qubits_parameters[target_qubit_component]["ramp_length"], 
                                        amplitude=target_qubit_amp)
        ramp_down = pulse_library.gaussian_fall(uid="ramp_down", 
                                        length=qubits_parameters[target_qubit_component]["ramp_length"], 
                                        amplitude=target_qubit_amp)
        
        target_length_sweep = LinearSweepParameter(uid="pulses", start=0, stop=npts-1, count=npts)
        on_off_cases = LinearSweepParameter(uid="on_off_case", start=0, stop=1, count=2)
        x_y_z_cases = LinearSweepParameter(uid="x_y_z_case", start=0, stop=2, count=3)

        phase = qubits_parameters[target_qubit_component]["readout_phase"]

        exp_cr = Experiment(
            uid="cr",
            signals=[
                ExperimentSignal("control_drive"),
                ExperimentSignal("target_drive_1"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
                ExperimentSignal("drive"),
            ],
        )

        def repeat(count: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(count, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=count):
                        for v in count.values:
                            with exp.case(v):
                                if v == 0:
                                    exp.play(signal="target_drive_1", pulse=ramp_up)
                                    exp.play(signal="target_drive_1", pulse=ramp_down)
                                else:
                                    exp.play(signal="target_drive_1", pulse=ramp_up)
                                    for _ in range(int(v)):
                                        f()
                                    exp.play(signal="target_drive_1", pulse=ramp_down)
                else:
                    for _ in range(count):
                        f()

            return decorator


        def switch(cases: int | SweepParameter | LinearSweepParameter, exp):
            def decorator(f):
                if isinstance(cases, (LinearSweepParameter, SweepParameter)):
                    with exp.match(sweep_parameter=cases):
                        for v in cases.values:
                            with exp.case(v):
                                f(v)

            return decorator
        

        with exp_cr.acquire_loop_rt(
            uid="shots",
            count=pow(2, average_exponent),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - real time sweep of Rabi ampitudes
            with exp_cr.sweep(uid="rabi_sweep", parameter=target_length_sweep):
                # play qubit excitation pulse - pulse amplitude is swept
                with exp_cr.sweep(uid="on_off_case", parameter=on_off_cases):
                    with exp_cr.sweep(uid="x_y_z_case", parameter=x_y_z_cases):
                        with exp_cr.section(
                            uid="control_qubit_excitation", alignment=SectionAlignment.RIGHT
                        ):
                            
                            @switch(on_off_cases, exp = exp_cr)
                            def play_control_drive(on_off_case):
                                with exp_cr.section():
                                    if on_off_case == 0:
                                        pass
                                    else:
                                        exp_cr.play(signal="control_drive", pulse=control_qubit_drive_pi_pulse)

                        with exp_cr.section(
                            uid="target_qubit_drive", play_after = "control_qubit_excitation"
                        ):
                            @repeat(target_length_sweep, exp_cr)
                            def play_rabi():
                                exp_cr.play(signal = "target_drive_1", pulse = target_drive_chunk)

                        with exp_cr.section(uid = "x_y_z_measurement", play_after = "target_qubit_drive"):
                            @switch(x_y_z_cases, exp = exp_cr)
                            def play_x_y_z(x_y_z_case):
                                with exp_cr.section():
                                    if x_y_z_case == 0: #  X measure
                                        exp_cr.play(signal="drive", pulse=target_qubit_pi2_pulse, phase = 0)
                                    elif x_y_z_case == 1: # Y measure
                                        exp_cr.play(signal="drive", pulse=target_qubit_pi2_pulse, phase = np.pi/2)
                                    else: # Z measure
                                        pass
                        # readout pulse and data acquisition
                        with exp_cr.section(uid="readout_section", play_after="x_y_z_measurement"):
                            # play readout pulse on measure line
                            exp_cr.play(signal="measure", pulse=target_qubit_readout_pulse, phase = phase)
                            # trigger signal data acquisition
                            exp_cr.acquire(
                                signal="acquire",
                                handle="cr_caliibration",
                                kernel=target_qubit_readout_weighting_function,
                            )
                        # relax time after readout - for qubit relaxation to groundstate and signal processing
                        with exp_cr.section(uid="reserve", length=qubits_parameters[target_qubit_component]["reset_delay_length"]):
                            exp_cr.reserve(signal="measure")
    
        signal_map = self.signal_map(control_qubit_component, which_qubit = "control")
        exp_cr.set_signal_map(signal_map)
        signal_map = self.signal_map(target_qubit_component)
        exp_cr.set_signal_map(signal_map)
        
        compiled_experiment_cr_calib_length = self.session.compile(exp_cr)
        
        self.cr_calibration_results_length = self.session.run(compiled_experiment_cr_calib_length)
        
        if is_plot_simulation:
            self.simulation_plot(compiled_experiment_cr_calib_length, start_time=0, length=20e-6)

    
    def plot_cr_calibration_length(self, is_fit = True):

        ### data processing ###############################################################

        control_off_X_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,0,0]
        control_on_X_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,1,0]

        control_off_Y_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,0,1]
        control_on_Y_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,1,1]

        control_off_Z_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,0,2]
        control_on_Z_data = self.cr_calibration_results_length.get_data("cr_caliibration")[:,1,2]

        time = np.linspace(0, self.exp_cr_calibration_length["duration"], self.exp_cr_calibration_length["npts"])

        if self.which_data == "I":
            control_off_X_data = np.real(control_off_X_data)
            control_on_X_data = np.real(control_on_X_data)
            control_off_Y_data = np.real(control_off_Y_data)
            control_on_Y_data = np.real(control_on_Y_data)
            control_off_Z_data = np.real(control_off_Z_data)
            control_on_Z_data = np.real(control_on_Z_data)
        else:
            control_off_X_data = np.imag(control_off_X_data)
            control_on_X_data = np.imag(control_on_X_data)
            control_off_Y_data = np.imag(control_off_Y_data)
            control_on_Y_data = np.imag(control_on_Y_data)
            control_off_Z_data = np.imag(control_off_Z_data)
            control_on_Z_data = np.imag(control_on_Z_data)

        fig, ax = plt.subplots(3, 1, figsize=(10, 20))

        ax[0].plot(time*1e6, control_off_X_data, color="b", marker="o", label = "control qubit off <X>")
        ax[0].plot(time*1e6, control_on_X_data, color="r", marker="o", label = "control qubit on <X>")
        ax[1].plot(time*1e6, control_off_Y_data, color="b", marker="o", label = "control qubit off <Y>")
        ax[1].plot(time*1e6, control_on_Y_data, color="r", marker="o", label = "control qubit on <Y>")
        ax[2].plot(time*1e6, control_off_Z_data, color="b", marker="o", label = "control qubit off <Z>")
        ax[2].plot(time*1e6, control_on_Z_data, color="r", marker="o", label = "control qubit on <Z>")

        fig.suptitle(f"CR calibration, target qubit amp : {self.exp_cr_calibration_length["target_qubit_amp"]}", fontsize=20)

        fig.text(0.5, 0.04, "Time [us]", ha="center", fontsize=20)
        fig.text(0.04, 0.5, f"{self.which_data} (a.u.)", va="center", rotation="vertical", fontsize=20)

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()