from typing import List, Union
import numpy as np
from acnportal import acnsim
import cvxpy as cp
import pytz
from datetime import datetime, timedelta
import adacharge
from adacharge import SessionInfo, InfrastructureInfo, get_active_sessions, ObjectiveComponent
from acnportal.acnsim.network.sites import caltech_acn
from acnportal.signals import tariffs
from collections import namedtuple


TIMEZONE = pytz.timezone('America/Los_Angeles')
PERIOD = 5  # minute
VOLTAGE = 208  # volts
KW_TO_AMPS = 1000 / 208
KWH_TO_AMP_PERIODS = KW_TO_AMPS * (60 / 5)


def get_infrastructure(network):
    N = len(network.station_ids)
    return InfrastructureInfo(network.constraint_matrix,
                              network.magnitudes,
                              network._phase_angles,
                              network._voltages,
                              network.constraint_index,
                              network.station_ids,
                              min_pilot=np.array([0] * N),
                              max_pilot=np.array([32] * N))


class StaticInterface(acnsim.Interface):
    """ ACN-Portal Interface using static data. """
    def __init__(self,
                 last_applied_pilot_signals=None,
                 last_actual_charging_rate=None,
                 current_time=None,
                 period=None,
                 active_sessions=None,
                 infrastructure=None,
                 tariff=None,
                 start=None,
                 prev_peak=None
                 ):
        super().__init__(None)
        self._last_applied_pilot_signals = last_applied_pilot_signals
        self._last_actual_charging_rate = last_actual_charging_rate
        self._current_time = current_time
        self._period = period
        self._active_sessions = active_sessions
        self._infrastructure = infrastructure
        self._tariff = tariff
        self._start = start
        self._prev_peak = prev_peak

    @property
    def last_applied_pilot_signals(self):
        """ Return the pilot signals that were applied in the last _iteration of the simulation for all active EVs.

        Does not include EVs that arrived in the current _iteration.

        Returns:
            Dict[str, number]: A dictionary with the session ID as key and the pilot signal as value.
        """
        return self._last_applied_pilot_signals

    @property
    def last_actual_charging_rate(self):
        """ Return the actual charging rates in the last period for all active EVs.

        Returns:
            Dict[str, number]:  A dictionary with the session ID as key and actual charging rate as value.
        """
        return self._last_actual_charging_rate

    @property
    def current_time(self):
        """ Get the current time (the current _iteration) of the simulator.

        Returns:
            int: The current _iteration of the simulator.
        """
        return self._current_time

    @property
    def period(self):
        """ Return the length of each timestep in the simulation.

        Returns:
            int: Length of each time interval in the simulation. [minutes]
        """
        return self._period

    def active_sessions(self):
        """ Return a list of SessionInfo objects describing the currently charging EVs.

        Returns:
            List[SessionInfo]: List of currently active charging sessions.
        """
        return self._active_sessions

    def infrastructure_info(self):
        """ Returns an InfrastructureInfo object generated from interface.

        Returns:
            InfrastructureInfo: A description of the charging infrastructure.
        """
        return self._infrastructure

    def allowable_pilot_signals(self, station_id):
        """ Returns the allowable pilot signal levels for the specified EVSE.
        One may assume an EVSE pilot signal of 0 is allowed regardless
        of this function's return values.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            bool: If the range is continuous or not
            list[float]: The sorted set of acceptable pilot signals. If continuous this range will have 2 values
                the min and the max acceptable values. [A]
        """
        i = self.infrastructure_info().get_station_index(station_id)
        return (self.infrastructure_info().is_continuous[i],
                self.infrastructure_info().allowable_pilots[i])

    def max_pilot_signal(self, station_id):
        """ Returns the maximum allowable pilot signal level for the specified EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            float: the maximum pilot signal supported by this EVSE. [A]
        """
        i = self.infrastructure_info().get_station_index(station_id)
        return self.infrastructure_info().max_pilot[i]

    def min_pilot_signal(self, station_id):
        """ Returns the minimum allowable pilot signal level for the specified EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            float: the minimum pilot signal supported by this EVSE. [A]
        """
        i = self.infrastructure_info().get_station_index(station_id)
        return self.infrastructure_info().min_pilot[i]

    def evse_voltage(self, station_id):
        """ Returns the voltage of the EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            float: voltage of the EVSE. [V]
        """
        i = self.infrastructure_info().get_station_index(station_id)
        return self.infrastructure_info().voltages[i]

    def evse_phase(self, station_id):
        """ Returns the phase angle of the EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            float: phase angle of the EVSE. [degrees]
        """
        i = self.infrastructure_info().get_station_index(station_id)
        return self.infrastructure_info().phases[i]

    def remaining_amp_periods(self, ev):
        """ Return the EV's remaining demand in A*periods.

        Returns:
            float: the EV's remaining demand in A*periods.
        """
        return self._convert_to_amp_periods(ev.remaining_demand, ev.station_id)

    def _convert_to_amp_periods(self, kwh, station_id):
        """ Convert the given energy in kWh to A*periods based on the voltage at EVSE station_id.

        Returns:
            float: kwh in A*periods.

        """
        return kwh * 1000 / self.evse_voltage(station_id) * 60 / self.period

    def get_constraints(self):
        """ Get the constraint matrix and corresponding EVSE ids for the network.

        Returns:
            np.ndarray: Matrix representing the constraints of the network. Each row is a constraint and each
        """
        Constraint = namedtuple(
            "Constraint",
            ["constraint_matrix", "magnitudes", "constraint_ids", "evse_index"],
        )
        infrastructure = self.infrastructure_info()
        return Constraint(
            infrastructure.constraint_matrix,
            infrastructure.constraint_limits,
            infrastructure.constraint_ids,
            infrastructure.station_ids,
        )

    def is_feasible(
        self,
        load_currents,
        linear=False,
        violation_tolerance=1e-5,
        relative_tolerance=1e-7,
    ):
        """ Return if a set of current magnitudes for each load are feasible.

        For a given constraint, the larger of the violation_tolerance
        and relative_tolerance is used to evaluate feasibility.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.
            violation_tolerance (float): Absolute amount by which
                schedule may violate network constraints. Default
                None, in which case the network's violation_tolerance
                attribute is used.
            relative_tolerance (float): Relative amount by which
                schedule may violate network constraints. Default
                None, in which case the network's relative_tolerance
                attribute is used.

        Returns:
            bool: If load_currents is feasible at time t according to this set of constraints.
        """
        if len(load_currents) == 0:
            return True
        # Check that all schedules are the same length
        schedule_lengths = set(len(x) for x in load_currents.values())
        if len(schedule_lengths) > 1:
            raise acnsim.InvalidScheduleError(
                "All schedules should have the same length.")
        schedule_length = schedule_lengths.pop()

        # Convert input schedule into its matrix representation
        infrastructure = self.infrastructure_info()
        schedule_matrix = np.array(
            [
                load_currents[station_id]
                if station_id in load_currents
                else [0] * schedule_length
                for station_id in infrastructure.station_ids
            ]
        )
        return self.infrastructure_constraints_feasible(
            schedule_matrix, linear, violation_tolerance, relative_tolerance
        )

    def infrastructure_constraints_feasible(
        self, rates, linear, violation_tolerance=1e-5, relative_tolerance=1e-7
    ):
        infrastructure = self.infrastructure_info()
        tol = np.maximum(
            violation_tolerance, relative_tolerance * infrastructure.constraint_limits
        )
        if not linear:
            phase_in_rad = np.deg2rad(infrastructure.phases)
            for j, v in enumerate(infrastructure.constraint_matrix):
                a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
                line_currents = np.linalg.norm(a @ rates, axis=0)
                if not np.all(
                    line_currents <= infrastructure.constraint_limits[j] + tol[j]
                ):
                    return False
        else:
            for j, v in enumerate(infrastructure.constraint_matrix):
                line_currents = np.linalg.norm(np.abs(v) @ rates, axis=0)
                if not np.all(
                    line_currents <= infrastructure.constraint_limits[j] + tol[j]
                ):
                    return False
        return True

    def get_prices(self, length, start=None):
        """ Get a vector of prices beginning at time start and continuing for length periods. ($/kWh)

        Args:
            length (int): Number of elements in the prices vector. One entry per period.
            start (int): Time step of the simulation where price vector should begin. If None, uses the current timestep
                of the simulation. Default None.

        Returns:
            np.ndarray[float]: Array of floats where each entry is the price for the corresponding period. ($/kWh)
        """
        if self._tariff is not None:
            if start is None:
                start = self.current_time
            price_start = self._start + timedelta(minutes=self.period) * start
            return np.array(
                self._tariff.get_tariffs(
                    price_start, length, self.period
                )
            )
        else:
            raise ValueError("No pricing method is specified.")

    def get_demand_charge(self, start=None):
        """ Get the demand charge for the given period. ($/kW)

        Args:
            start (int): Time step of the simulation where price vector should begin. If None, uses the current timestep
                of the simulation. Default None.

        Returns:
            float: Demand charge for the given period. ($/kW)
        """
        if self._tariff is not None:
            if start is None:
                start = self.current_time
            price_start = self._start + timedelta(minutes=self.period) * start
            return self._tariff.get_demand_charge(price_start)
        else:
            raise ValueError("No pricing method is specified.")

    def get_prev_peak(self):
        """ Get the highest aggregate peak demand so far in the simulation.

        Returns:
            float: Peak demand so far in the simulation. (A)
        """
        return self._prev_peak