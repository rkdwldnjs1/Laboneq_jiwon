# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Calibration, Oscillator, SignalCalibration
from laboneq.dsl.device import LogicalSignalGroup
from laboneq.dsl.device.io_units import LogicalSignal
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.quantum.quantum_element import QuantumElement, QuantumElement_cr, QuantumElement_mult, SignalType, SignalType_cr, SignalType_mult


@classformatter
@dataclass
class TransmonParameters:
    #: Resonance frequency of the qubits g-e transition.
    resonance_frequency_ge: Optional[float] = None
    #: Resonance frequency of the qubits e-f transition.
    resonance_frequency_ef: Optional[float] = None
    #: Local oscillator frequency for the drive signals.
    drive_lo_frequency: Optional[float] = None
    #: Readout resonantor frequency of the qubit.
    readout_resonator_frequency: Optional[float] = None
    #: local oscillator frequency for the readout lines.
    readout_lo_frequency: Optional[float] = None
    #: integration delay between readout pulse and data acquisition, defaults to 20 ns.
    readout_integration_delay: Optional[float] = 20e-9
    #: drive power setting, defaults to 10 dBm.
    drive_range: Optional[float] = 10
    #: readout output power setting, defaults to 5 dBm.
    readout_range_out: Optional[float] = 5
    #: readout input power setting, defaults to 10 dBm.
    readout_range_in: Optional[float] = 10
    #: offset voltage for flux control line - defaults to 0.
    flux_offset_voltage: Optional[float] = 0
    #: Free form dictionary of user defined parameters.
    user_defined: dict | None = field(default_factory=dict)

    @property
    def drive_frequency_ge(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_ge - self.drive_lo_frequency
        except TypeError:
            return None

    @property
    def drive_frequency_ef(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_ef - self.drive_lo_frequency
        except TypeError:
            return None

    @property
    def readout_frequency(self) -> float | None:
        """Readout baseband frequency."""
        try:
            return self.readout_resonator_frequency - self.readout_lo_frequency
        except TypeError:
            return None


@classformatter
@dataclass(init=False, repr=True, eq=False)
class Transmon(QuantumElement):
    """A class for a superconducting, flux-tuneable Transmon Qubit."""

    parameters: TransmonParameters

    def __init__(
        self,
        uid: str | None = None,
        signals: dict[str, LogicalSignal | str] | None = None,
        parameters: TransmonParameters | dict[str, Any] | None = None,
    ):
        """
        Initializes a new Transmon Qubit.

        Args:
            uid: A unique identifier for the Qubit.
            signals: A mapping of logical signals associated with the qubit.
                Qubit accepts the following keys in the mapping: 'drive', 'measure', 'acquire', 'flux'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
                Required for generating calibration and experiment signals via `calibration()` and `experiment_signals()`.
        """
        if parameters is None:
            self.parameters = TransmonParameters()
        elif isinstance(parameters, dict):
            self.parameters = TransmonParameters(**parameters)
        else:
            self.parameters = parameters
        super().__init__(uid=uid, signals=signals)

    @classmethod
    def from_logical_signal_group(
        cls,
        uid: str,
        lsg: LogicalSignalGroup,
        parameters: TransmonParameters | dict[str, Any] | None = None,
    ) -> "Transmon":
        """Transmon Qubit from logical signal group.

        Args:
            uid: A unique identifier for the Qubit.
            lsg: Logical signal group.
                Transmon Qubit understands the following signal line names:

                    - drive: 'drive', 'drive_line'
                    - drive_ef: 'drive_ef', 'drive_line_ef'
                    - measure: 'measure', 'measure_line'
                    - acquire: 'acquire', 'acquire_line'
                    - flux: 'flux', 'flux_line'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
        """
        signal_type_map = {
            SignalType.DRIVE: ["drive", "drive_line"],
            SignalType.DRIVE_EF: ["drive_ef", "drive_line_ef"],
            SignalType.MEASURE: ["measure", "measure_line"],
            SignalType.ACQUIRE: ["acquire", "acquire_line"],
            SignalType.FLUX: ["flux", "flux_line"],
        }
        if parameters is None:
            parameters = TransmonParameters()
        elif isinstance(parameters, dict):
            parameters = TransmonParameters(**parameters)
        return cls._from_logical_signal_group(
            uid=uid,
            lsg=lsg,
            parameters=parameters,
            signal_type_map=signal_type_map,
        )

    def calibration(self, set_local_oscillators=True) -> Calibration:
        """Generate calibration from the parameters and attached signal lines.

        `Qubit` requires `parameters` for it to be able to produce calibration objects.

        Args:
            set_local_oscillators (bool):
                If True, adds local oscillator settings to the calibration.

        Returns:
            calibration:
                Prefilled calibration object from Qubit parameters.
        """
        drive_lo = None
        readout_lo = None
        if set_local_oscillators:
            if self.parameters.drive_lo_frequency is not None:
                drive_lo = Oscillator(
                    uid=f"{self.uid}_drive_local_osc",
                    frequency=self.parameters.drive_lo_frequency,
                )
            if self.parameters.readout_lo_frequency is not None:
                readout_lo = Oscillator(
                    uid=f"{self.uid}_readout_local_osc",
                    frequency=self.parameters.readout_lo_frequency,
                )
        if self.parameters.readout_frequency is not None:
            readout_oscillator = Oscillator(
                uid=f"{self.uid}_readout_acquire_osc",
                frequency=self.parameters.readout_frequency,
                modulation_type=ModulationType.AUTO,
            )

        calib = {}
        if "drive" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_ge is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_ge_osc",
                    frequency=self.parameters.drive_frequency_ge,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive"]] = sig_cal
        if "drive_ef" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_ef is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_ef_osc",
                    frequency=self.parameters.drive_frequency_ef,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive_ef"]] = sig_cal
        if "measure" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.readout_frequency is not None:
                sig_cal.oscillator = readout_oscillator
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_out
            calib[self.signals["measure"]] = sig_cal
        if "acquire" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.readout_frequency is not None:
                sig_cal.oscillator = readout_oscillator
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_in
            sig_cal.port_delay = self.parameters.readout_integration_delay
            calib[self.signals["acquire"]] = sig_cal
        if "flux" in self.signals:
            calib[self.signals["flux"]] = SignalCalibration(
                voltage_offset=self.parameters.flux_offset_voltage,
            )
        return Calibration(calib)

# ------------------------------------------------------------------ ADD : 12/30/2024 ------------------------------------------------------------------



@classformatter
@dataclass
class TransmonParameters_cr:                                       # adaptable to N <= 10
    #: Resonance frequency of the qubits g-e transition.
    resonance_frequency_ge: Optional[float] = None
    #: Resonance frequency of the qubits e-f transition.
    resonance_frequency_ef: Optional[float] = None
    #: Resonance frequency of the qubit auxiliary transition 0.
    resonance_frequency_aux0: Optional[float] = None
    #: Resonance frequency of the qubit auxiliary transition 1.
    resonance_frequency_aux1: Optional[float] = None
    #: Resonance frequency of the qubit auxiliary transition 2.
    resonance_frequency_aux2: Optional[float] = None
    #: Resonance frequency of the qubit auxiliary transition 3.
    resonance_frequency_aux3: Optional[float] = None
    #: Resonance frequency of the qubit auxiliary transition 4.
    resonance_frequency_aux4: Optional[float] = None
    #: Resonance frequency of the qubit auxiliary transition 5.
    resonance_frequency_aux5: Optional[float] = None
    # #: Resonance frequency of the qubit auxiliary transition 6.
    # resonance_frequency_aux6: Optional[float] = None
    # #: Resonance frequency of the qubit auxiliary transition 7.
    # resonance_frequency_aux7: Optional[float] = None
    # #: Resonance frequency of the qubit auxiliary transition 8.
    # resonance_frequency_aux8: Optional[float] = None
    # #: Resonance frequency of the qubit auxiliary transition 9.
    # resonance_frequency_aux9: Optional[float] = None
    #: Local oscillator frequency for the drive signals.
    drive_lo_frequency: Optional[float] = None
    #: Readout resonantor frequency of the qubit.
    readout_resonator_frequency: Optional[float] = None
    #: local oscillator frequency for the readout lines.
    readout_lo_frequency: Optional[float] = None
    #: integration delay between readout pulse and data acquisition, defaults to 20 ns.
    readout_integration_delay: Optional[float] = 20e-9
    #: drive power setting, defaults to 10 dBm.
    drive_range: Optional[float] = 10
    #: readout output power setting, defaults to 5 dBm.
    readout_range_out: Optional[float] = 5
    #: readout input power setting, defaults to 10 dBm.
    readout_range_in: Optional[float] = 10
    #: offset voltage for flux control line - defaults to 0.
    flux_offset_voltage: Optional[float] = 0
    #: Free form dictionary of user defined parameters.
    user_defined: dict | None = field(default_factory=dict)

    @property
    def drive_frequency_ge(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_ge - self.drive_lo_frequency
        except TypeError:
            return None

    @property
    def drive_frequency_ef(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_ef - self.drive_lo_frequency
        except TypeError:
            return None
        
    @property
    def drive_frequency_aux0(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_aux0 - self.drive_lo_frequency
        except TypeError:
            return None
        
    @property
    def drive_frequency_aux1(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_aux1 - self.drive_lo_frequency
        except TypeError:
            return None
    
    @property
    def drive_frequency_aux2(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_aux2 - self.drive_lo_frequency
        except TypeError:
            return None
        
    @property
    def drive_frequency_aux3(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_aux3 - self.drive_lo_frequency
        except TypeError:
            return None
        
    @property
    def drive_frequency_aux4(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_aux4 - self.drive_lo_frequency
        except TypeError:
            return None
        
    @property
    def drive_frequency_aux5(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_aux5 - self.drive_lo_frequency
        except TypeError:
            return None
    
    # @property
    # def drive_frequency_aux6(self) -> float | None:
    #     """Qubit drive frequency."""
    #     try:
    #         return self.resonance_frequency_aux6 - self.drive_lo_frequency
    #     except TypeError:
    #         return None
        
    # @property
    # def drive_frequency_aux7(self) -> float | None:
    #     """Qubit drive frequency."""
    #     try:
    #         return self.resonance_frequency_aux7 - self.drive_lo_frequency
    #     except TypeError:
    #         return None
    
    # @property
    # def drive_frequency_aux8(self) -> float | None:
    #     """Qubit drive frequency."""
    #     try:
    #         return self.resonance_frequency_aux8 - self.drive_lo_frequency
    #     except TypeError:
    #         return None
    
    # @property
    # def drive_frequency_aux9(self) -> float | None:
    #     """Qubit drive frequency."""
    #     try:
    #         return self.resonance_frequency_aux9 - self.drive_lo_frequency
    #     except TypeError:
    #         return None

    @property
    def readout_frequency(self) -> float | None:
        """Readout baseband frequency."""
        try:
            return self.readout_resonator_frequency - self.readout_lo_frequency
        except TypeError:
            return None



@classformatter
@dataclass
class TransmonParameters_mult:                                       # adaptable to N <= 10
    #: Resonance frequency of the qubits g-e transition.
    resonance_frequency_ge: Optional[float] = None
    #: Local oscillator frequency for the drive signals.
    drive_lo_frequency: Optional[float] = None
    #: Readout resonantor frequency of the qubit.
    readout_resonator_frequency: Optional[float] = None
    #: local oscillator frequency for the readout lines.
    readout_lo_frequency: Optional[float] = None
    #: integration delay between readout pulse and data acquisition, defaults to 20 ns.
    readout_integration_delay: Optional[float] = 20e-9
    #: drive power setting, defaults to 10 dBm.
    drive_range: Optional[float] = 10
    #: readout output power setting, defaults to 5 dBm.
    readout_range_out: Optional[float] = 5
    #: readout input power setting, defaults to 10 dBm.
    readout_range_in: Optional[float] = 10
    #: offset voltage for flux control line - defaults to 0.
    flux_offset_voltage: Optional[float] = 0
    #: Free form dictionary of user defined parameters.
    user_defined: dict | None = field(default_factory=dict)

    @property
    def drive_frequency_ge(self) -> float | None:
        """Qubit drive frequency."""
        try:
            return self.resonance_frequency_ge - self.drive_lo_frequency
        except TypeError:
            return None

    @property
    def readout_frequency(self) -> float | None:
        """Readout baseband frequency."""
        try:
            return self.readout_resonator_frequency - self.readout_lo_frequency
        except TypeError:
            return None


@classformatter
@dataclass(init=False, repr=True, eq=False)
class Transmon_cr(QuantumElement):
    """A class for a superconducting, flux-tuneable Transmon Qubit."""

    parameters: TransmonParameters_cr

    def __init__(
        self,
        uid: str | None = None,
        signals: dict[str, LogicalSignal | str] | None = None,
        parameters: TransmonParameters_cr | dict[str, Any] | None = None,
    ):
        """
        Initializes a new Transmon Qubit.

        Args:
            uid: A unique identifier for the Qubit.
            signals: A mapping of logical signals associated with the qubit.
                Qubit accepts the following keys in the mapping: 'drive', 'measure', 'acquire', 'flux'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
                Required for generating calibration and experiment signals via `calibration()` and `experiment_signals()`.
        """
        if parameters is None:
            self.parameters = TransmonParameters_cr()
        elif isinstance(parameters, dict):
            self.parameters = TransmonParameters_cr(**parameters)
        else:
            self.parameters = parameters
        super().__init__(uid=uid, signals=signals)

    @classmethod
    def from_logical_signal_group(
        cls,
        uid: str,
        lsg: LogicalSignalGroup,
        parameters: TransmonParameters_cr | dict[str, Any] | None = None,
    ) -> "Transmon_cr":
        """Transmon Qubit from logical signal group.

        Args:
            uid: A unique identifier for the Qubit.
            lsg: Logical signal group.
                Transmon Qubit understands the following signal line names:

                    - drive: 'drive', 'drive_line'
                    - drive_ef: 'drive_ef', 'drive_line_ef'
                    - drive_aux0: 'drive_aux0', 'drive_line_aux0'
                    - drive_aux1: 'drive_aux1', 'drive_line_aux1'
                    ...
                    - drive_aux9: 'drive_aux9', 'drive_line_aux9'
                    - measure: 'measure', 'measure_line'
                    - acquire: 'acquire', 'acquire_line'
                    - flux: 'flux', 'flux_line'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
        """
        signal_type_map = {
            SignalType_cr.DRIVE: ["drive", "drive_line"],
            SignalType_cr.DRIVE_EF: ["drive_ef", "drive_line_ef"],
            SignalType_cr.DRIVE_AUX0: ["drive_aux0", "drive_line_aux0"],
            SignalType_cr.DRIVE_AUX1: ["drive_aux1", "drive_line_aux1"],
            SignalType_cr.DRIVE_AUX2: ["drive_aux2", "drive_line_aux2"],
            SignalType_cr.DRIVE_AUX3: ["drive_aux3", "drive_line_aux3"],
            SignalType_cr.DRIVE_AUX4: ["drive_aux4", "drive_line_aux4"],
            SignalType_cr.DRIVE_AUX5: ["drive_aux5", "drive_line_aux5"],
            # SignalType_cr.DRIVE_AUX6: ["drive_aux6", "drive_line_aux6"],
            # SignalType_cr.DRIVE_AUX7: ["drive_aux7", "drive_line_aux7"],
            # SignalType_cr.DRIVE_AUX8: ["drive_aux8", "drive_line_aux8"],
            # SignalType_cr.DRIVE_AUX9: ["drive_aux9", "drive_line_aux9"],
            SignalType_cr.MEASURE: ["measure", "measure_line"],
            SignalType_cr.ACQUIRE: ["acquire", "acquire_line"],
            SignalType_cr.FLUX: ["flux", "flux_line"],
        }
        if parameters is None:
            parameters = TransmonParameters_cr()
        elif isinstance(parameters, dict):
            parameters = TransmonParameters_cr(**parameters)
        return cls._from_logical_signal_group(
            uid=uid,
            lsg=lsg,
            parameters=parameters,
            signal_type_map=signal_type_map,
        )

    def calibration(self, set_local_oscillators=True) -> Calibration:
        """Generate calibration from the parameters and attached signal lines.

        `Qubit` requires `parameters` for it to be able to produce calibration objects.

        Args:
            set_local_oscillators (bool):
                If True, adds local oscillator settings to the calibration.

        Returns:
            calibration:
                Prefilled calibration object from Qubit parameters.
        """
        drive_lo = None
        readout_lo = None
        if set_local_oscillators:
            if self.parameters.drive_lo_frequency is not None:
                drive_lo = Oscillator(
                    uid=f"{self.uid}_drive_local_osc",
                    frequency=self.parameters.drive_lo_frequency,
                )
            if self.parameters.readout_lo_frequency is not None:
                readout_lo = Oscillator(
                    uid=f"{self.uid}_readout_local_osc",
                    frequency=self.parameters.readout_lo_frequency,
                )
        if self.parameters.readout_frequency is not None:
            readout_oscillator = Oscillator(
                uid=f"{self.uid}_readout_acquire_osc",
                frequency=self.parameters.readout_frequency,
                modulation_type=ModulationType.AUTO,
            )

        calib = {}
        if "drive" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_ge is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_ge_osc",
                    frequency=self.parameters.drive_frequency_ge,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive"]] = sig_cal
        if "drive_ef" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_ef is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_ef_osc",
                    frequency=self.parameters.drive_frequency_ef,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive_ef"]] = sig_cal
        if "drive_aux0" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_aux0 is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_aux0_osc",
                    frequency=self.parameters.drive_frequency_aux0,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive_aux0"]] = sig_cal
        if "drive_aux1" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_aux1 is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_aux1_osc",
                    frequency=self.parameters.drive_frequency_aux1,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive_aux1"]] = sig_cal
        if "drive_aux2" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_aux2 is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_aux2_osc",
                    frequency=self.parameters.drive_frequency_aux2,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive_aux2"]] = sig_cal
        if "drive_aux3" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_aux3 is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_aux3_osc",
                    frequency=self.parameters.drive_frequency_aux3,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive_aux3"]] = sig_cal
        if "drive_aux4" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_aux4 is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_aux4_osc",
                    frequency=self.parameters.drive_frequency_aux4,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive_aux4"]] = sig_cal
        if "drive_aux5" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_aux5 is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_aux5_osc",
                    frequency=self.parameters.drive_frequency_aux5,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = drive_lo
            sig_cal.range = self.parameters.drive_range
            calib[self.signals["drive_aux5"]] = sig_cal
        if "measure" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.readout_frequency is not None:
                sig_cal.oscillator = readout_oscillator
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_out
            calib[self.signals["measure"]] = sig_cal
        if "acquire" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.readout_frequency is not None:
                sig_cal.oscillator = readout_oscillator
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_in
            sig_cal.port_delay = self.parameters.readout_integration_delay
            calib[self.signals["acquire"]] = sig_cal
        if "flux" in self.signals:
            calib[self.signals["flux"]] = SignalCalibration(
                voltage_offset=self.parameters.flux_offset_voltage,
            )
        return Calibration(calib)

@classformatter
@dataclass(init=False, repr=True, eq=False)
class Transmon_mult(QuantumElement_mult):
    """A class for a superconducting, flux-tuneable Transmon Qubit."""

    parameters: TransmonParameters_mult

    def __init__(
        self,
        uid: str | None = None,
        signals: dict[str, LogicalSignal | str] | None = None,
        parameters: TransmonParameters_mult | dict[str, Any] | None = None,
    ):
        """
        Initializes a new Transmon Qubit.

        Args:
            uid: A unique identifier for the Qubit.
            signals: A mapping of logical signals associated with the qubit.
                Qubit accepts the following keys in the mapping: 'drive', 'measure', 'acquire', 'flux'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
                Required for generating calibration and experiment signals via `calibration()` and `experiment_signals()`.
        """
        if parameters is None:
            self.parameters = TransmonParameters_mult()
        elif isinstance(parameters, dict):
            self.parameters = TransmonParameters_mult(**parameters)
        else:
            self.parameters = parameters
        super().__init__(uid=uid, signals=signals)

    @classmethod
    def from_logical_signal_group(
        cls,
        uid: str,
        lsg: LogicalSignalGroup,
        parameters: TransmonParameters_mult | dict[str, Any] | None = None,
    ) -> "Transmon_mult":
        """Transmon Qubit from logical signal group.

        Args:
            uid: A unique identifier for the Qubit.
            lsg: Logical signal group.
                Transmon Qubit understands the following signal line names:

                    - drive: 'drive', 'drive_line'
                    - measure: 'measure', 'measure_line'
                    - acquire: 'acquire', 'acquire_line'
                    - flux: 'flux', 'flux_line'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
        """
        signal_type_map = {
            SignalType_mult.DRIVE: ["drive", "drive_line"],
            SignalType_mult.MEASURE: ["measure", "measure_line"],
            SignalType_mult.ACQUIRE: ["acquire", "acquire_line"],
            SignalType_mult.FLUX: ["flux", "flux_line"],
        }
        if parameters is None:
            parameters = TransmonParameters_mult()
        elif isinstance(parameters, dict):
            parameters = TransmonParameters_mult(**parameters)
        return cls._from_logical_signal_group(
            uid=uid,
            lsg=lsg,
            parameters=parameters,
            signal_type_map=signal_type_map,
        )

    def calibration(self, set_local_oscillators=True) -> Calibration:
        """Generate calibration from the parameters and attached signal lines.

        `Qubit` requires `parameters` for it to be able to produce calibration objects.

        Args:
            set_local_oscillators (bool):
                If True, adds local oscillator settings to the calibration.

        Returns:
            calibration:
                Prefilled calibration object from Qubit parameters.
        """
        drive_lo = None
        readout_lo = None
        if set_local_oscillators:
            if self.parameters.drive_lo_frequency is not None:
                drive_lo = Oscillator(
                    uid=f"{self.uid}_drive_local_osc",
                    frequency=self.parameters.drive_lo_frequency,
                )
            if self.parameters.readout_lo_frequency is not None:
                readout_lo = Oscillator(
                    uid=f"{self.uid}_readout_local_osc",
                    frequency=self.parameters.readout_lo_frequency,
                )
        if self.parameters.readout_frequency is not None:
            readout_oscillator = Oscillator(
                uid=f"{self.uid}_readout_acquire_osc",
                frequency=self.parameters.readout_frequency,
                modulation_type=ModulationType.AUTO,
            )

        calib = {}
        if "drive" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.drive_frequency_ge is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_drive_ge_osc",
                    frequency=self.parameters.drive_frequency_ge,
                    modulation_type=ModulationType.AUTO,
                )
        if "measure" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.readout_frequency is not None:
                sig_cal.oscillator = readout_oscillator
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_out
            calib[self.signals["measure"]] = sig_cal
        if "acquire" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.readout_frequency is not None:
                sig_cal.oscillator = readout_oscillator
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_in
            sig_cal.port_delay = self.parameters.readout_integration_delay
            calib[self.signals["acquire"]] = sig_cal
        if "flux" in self.signals:
            calib[self.signals["flux"]] = SignalCalibration(
                voltage_offset=self.parameters.flux_offset_voltage,
            )
        return Calibration(calib)
