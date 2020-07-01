"""
Pricing EV Charging Services with Demand Charge
Zachary Lee
Created 4-9-2020
Last Updated: 4-9-2020

In this experiment we use the dual of an offline scheduling optimization to calculate
the prices paid by each user for each session.

For simplicity we will use the SessionInfo and InfrastructureInfo objects from adacharge.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from datetime import datetime
import adacharge
from adacharge import ObjectiveComponent
from acnportal.algorithms import enforce_pilot_limit
from utils import StaticInterface
from typing import List, Union
import warnings
from acnportal.acnsim.interface import SessionInfo, InfrastructureInfo

PERIOD = 5  # minute
VOLTAGE = 208  # volts
KW_TO_AMPS = 1000 / 208
KWH_TO_AMP_PERIODS = KW_TO_AMPS * (60 / 5)


class AdaptiveChargingFloatingPeak(adacharge.AdaptiveChargingOptimization):
    def build_problem(self, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo,
                      peak_limit:Union[float, List[float], np.ndarray]=None, prev_peak:float=0):
        """ Build parts of the optimization problem including variables, constraints, and objective function.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.
            prev_peak (float): Previous peak current draw during the current billing period.

        Returns:
            Dict[str: object]:
                'objective' : cvxpy expression for the objective of the optimization problem
                'constraints': list of all constraints for the optimization problem
                'variables': dict mapping variable name to cvxpy Variable.
        """
        if peak_limit is not None:
            warnings.warn('AdaptiveChargingFloatingPeak does not support '
                          'peak_limit.')
        optimization_horizon = max(s.arrival_offset + s.remaining_time for s in active_sessions)
        num_evses = len(infrastructure.station_ids)
        rates = cp.Variable(shape=(num_evses, optimization_horizon))
        peak = cp.Variable()
        constraints = {}

        # Rate constraints
        constraints.update(self.charging_rate_bounds(rates, active_sessions,
                                                     infrastructure.station_ids))

        # Energy Delivered Constraints
        constraints.update(self.energy_constraints(rates, active_sessions, infrastructure,
                                                   self.interface.period, self.enforce_energy_equality))

        # Infrastructure Constraints
        constraints.update(self.infrastructure_constraints(rates, infrastructure, self.constraint_type))

        # Track Peak
        constraints['floating_peak'] = peak >= cp.sum(rates, axis=0)

        # Objective Function
        objective = cp.Maximize(self.build_objective(rates, infrastructure,
                                                     prev_peak=prev_peak,
                                                     floating_peak=peak))
        return {'objective': objective,
                'constraints': constraints,
                'variables': {'rates': rates, 'peak': peak}}


def floating_demand_charge(rates, infrastructure, interface, floating_peak, **kwargs):
    kW_peak = floating_peak * infrastructure.voltages[0] / 1000
    dc = interface.get_demand_charge()
    return -dc * kW_peak


def primal_scheduling_and_pricing(sessions, infrastructure, tariff, period, start_time):
    """ Calculate prices directly by first solving the scheduling problem.

    Args:
        sessions (List[SessionInfo]): Sessions to determine prices for.
        infrastructure (InfrastructureInfo): Charging infrastructure
            description.
        tariff (TimeOfUseTariff): Tariff structure to consider.
        period (int): Length of each control interval. [min]
        start_time (Datetime): Start time of the control horizon. Typically
            12:00 AM the first day of the month.

    Returns:
        dict:
            "alpha" (float): Energy prices,
            "beta" (float): Infrastructure congestion prices,
            "gamma" (float): Charger congestion prices,
            "delta" (float): Disaggregated demand charge,
            "rates" (np.array): Schedule of charging rates [A],
            "peak' (float): Peak power draw [A],
            "cost" (float): Total cost of charging,
            "prob" (cvxpy.Problem): CVXPY optimization problem

    """
    # Since we are using the adacharge framework, we need to define an interface.
    interface = StaticInterface(active_sessions=sessions,
                                infrastructure=infrastructure,
                                period=period,
                                tariff=tariff,
                                start=start_time,
                                current_time=0,
                                prev_peak=0)

    # Modify session max charging rate to include EVSE pilot limit.
    sessions = enforce_pilot_limit(sessions, infrastructure)

    # Objective includes energy cost and demand charge.
    obj = [ObjectiveComponent(adacharge.tou_energy_cost, 1),
           ObjectiveComponent(floating_demand_charge, 1)]

    # We will use linear constraints to make analysis of the pricing rule easier.
    # Enforce energy equality ensures all EVs get their energy demand.
    # We default to using MOSEK, which is available for free for academics.
    alg = AdaptiveChargingFloatingPeak(obj,
                                       interface,
                                       constraint_type='LINEAR',
                                       enforce_energy_equality=True,
                                       solver='MOSEK')

    # Build and solve the problem.
    problem_dict = alg.build_problem(sessions, infrastructure)
    prob = cp.Problem(problem_dict['objective'],
                      list(problem_dict['constraints'].values()))
    prob.solve(solver=alg.solver, verbose=True)

    # Extract dual prices
    constraints = problem_dict['constraints']

    # Session Prices
    alpha = np.array([constraints[f'energy_constraints.{s.session_id}'].dual_value for s in sessions])

    # Infrastructure Congestion
    beta = np.vstack([constraints[f'infrastructure_constraints.{const_name}'].dual_value
                      for const_name in infrastructure.constraint_ids])

    # Charger Congestion
    gamma = constraints['charging_rate_bounds.ub'].dual_value

    # Dis-aggregated demand charge
    delta = constraints['floating_peak'].dual_value

    # Charging Schedule
    rates = problem_dict['variables']['rates'].value

    # Peak current draw
    peak = problem_dict['variables']['peak'].value

    return {'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'delta': delta,
            'rates': rates,
            'peak': peak,
            'cost' : -prob.value,
            'prob': prob}


def dual_scheduling_and_pricing(sessions, infrastructure, tou_price, demand_charge):
    """ Calculate prices directly by solving the dual problem.

        Note this function can be slower than prices obtained by
        optimal_schedule() because the process of building the problem is not
        vectorized.

    Args:
        sessions (List[SessionInfo]): Sessions to determine prices for.
        infrastructure (InfrastructureInfo): Charging infrastructure
            description.
        tou_prices (np.array[float]): List of energy prices. [$/kWh]
        demand_charge (float): Demand charge. [$/kW]
        solar_production (np.array[float]): Solar energy production in each time period.

    Returns:
        dict:
            "alpha" (float): Energy prices,
            "beta" (float): Infrastructure congestion prices,
            "gamma" (float): Charger congestion prices,
            "delta" (float): Disaggregated demand charge,
            "rates" (np.array): Schedule of charging rates [A],
            "peak' (float): Peak power draw [A],
            "cost" (float): Total cost of charging,
            "prob" (cvxpy.Problem): CVXPY optimization problem

    """
    num_stations = len(infrastructure.station_ids)
    num_sessions = len(sessions)
    max_t = max(sess.departure for sess in sessions)

    constraint_matrix = np.abs(infrastructure.constraint_matrix)
    constraint_limits = np.tile(infrastructure.constraint_limits, (max_t, 1)).T
    rate_ub = np.full((num_stations, max_t), 32)

    # Dual Variables
    alpha = cp.Variable(num_sessions, name='alpha')
    beta = cp.Variable((constraint_matrix.shape[0], max_t), name='beta', nonneg=True)
    gamma = cp.Variable((num_stations, max_t), 'gamma', nonneg=True)
    delta = cp.Variable(max_t, 'delta', nonneg=True)

    e = np.array([session.requested_energy for session in sessions])*KWH_TO_AMP_PERIODS
    obj = alpha @ e
    obj -= cp.sum(cp.multiply(constraint_limits, beta))
    obj -= cp.sum(cp.multiply(rate_ub, gamma))

    constraints = {}
    alpha_const_matrix = np.tile(tou_price / KWH_TO_AMP_PERIODS, (num_stations, 1))
    alpha_const_matrix += constraint_matrix.T @ beta
    alpha_const_matrix += gamma
    for i, session in enumerate(sessions):
        j = infrastructure.get_station_index(session.station_id)
        a, d = session.arrival, session.departure
        constraints[f'alpha_constraints.{session.session_id}'] = alpha_const_matrix[j, a:d] + delta[a:d] >= alpha[i]
    constraints['peak_constraint'] = demand_charge / KW_TO_AMPS >= cp.sum(delta)

    print('Solving...')
    prob = cp.Problem(cp.Maximize(obj), list(constraints.values()))
    prob.solve(solver=cp.MOSEK, verbose=True)
    # prob.solve(verbose=True)

    # Build rates matrix
    rates = np.zeros((num_stations, max_t))
    for i, session in enumerate(sessions):
        j = infrastructure.get_station_index(session.station_id)
        a, d = session.arrival, session.departure
        rates[j, a:d] = constraints[f'alpha_constraints.{session.session_id}'].dual_value
    peak = np.max(np.sum(rates, axis=0))
    return {'alpha': alpha.value * KWH_TO_AMP_PERIODS, 'beta': beta.value, 'gamma': gamma.value, 'delta': delta.value,
            'rates': rates, 'peak': peak, 'cost': prob.value, 'prob': prob}


def get_solar(csv_source, index_name, col_name, scale, init_time=datetime(2019, 1, 1)):
    raw_solar = pd.read_csv(csv_source, index_col=index_name)
    raw_solar.index = init_time + pd.to_timedelta(raw_solar.index, 'h')
    raw_solar = raw_solar.resample('{0}T'.format(PERIOD)).pad()  # should switch to interpolate...
    raw_solar = raw_solar[col_name].fillna(0).clip(lower=0)
    return raw_solar * scale


def dual_scheduling_and_pricing_with_solar(
        sessions, infrastructure, tou_price, demand_charge, solar_production):
    num_stations = len(infrastructure.evse_index)
    num_sessions = len(sessions)
    max_t = max(sess.departure for sess in sessions)

    constraint_matrix = np.abs(infrastructure.constraint_matrix)
    constraint_limits = np.tile(infrastructure.constraint_limits, (max_t, 1)).T
    rate_ub = np.full((num_stations, max_t), 32)

    # Dual Variables
    alpha = cp.Variable(num_sessions, name='alpha')
    beta = cp.Variable((constraint_matrix.shape[0], max_t), name='beta', nonneg=True)
    gamma = cp.Variable((num_stations, max_t), 'gamma', nonneg=True)
    delta = cp.Variable(max_t, 'delta', nonneg=True)
    epsilon = cp.Variable(max_t, 'epsilon', nonneg=True)

    e = np.array([session.energy_requested for session in sessions]) * KWH_TO_AMP_PERIODS
    obj = alpha @ e
    obj -= cp.sum(cp.multiply(constraint_limits, beta))
    obj -= cp.sum(cp.multiply(rate_ub, gamma))
    obj -= epsilon @ solar_production

    constraints = {}
    tou_prices_amps = tou_price / KWH_TO_AMP_PERIODS
    alpha_const_matrix = constraint_matrix.T @ beta
    alpha_const_matrix += gamma
    for i, session in enumerate(sessions):
        j = infrastructure.get_station_index(session.station_id)
        a, d = session.arrival, session.departure
        constraints[f'alpha_constraints.1.{session.session_id}'] = alpha_const_matrix[j, a:d] + tou_prices_amps[a:d] + delta[a:d] >= alpha[i]
        constraints[f'alpha_constraints.2.{session.session_id}'] = alpha_const_matrix[j, a:d] + epsilon[a:d] >= alpha[i]
    constraints['peak_constraint'] = demand_charge / KW_TO_AMPS >= cp.sum(delta)
    print('Solving...')
    prob = cp.Problem(cp.Maximize(obj), list(constraints.values()))
    prob.solve(solver=cp.MOSEK, verbose=True)

    # Build rates matrix
    rates_s = np.zeros((num_stations, max_t))
    rates_g = np.zeros((num_stations, max_t))

    for i, session in enumerate(sessions):
        j = infrastructure.get_station_index(session.station_id)
        a, d = session.arrival, session.departure
        rates_s[j, a:d] = constraints[f'alpha_constraints.1.{session.session_id}'].dual_value
        rates_g[j, a:d] = constraints[f'alpha_constraints.2.{session.session_id}'].dual_value
    rates = rates_s + rates_g
    peak = np.max(np.sum(rates_g, axis=0))
    return {'alpha': alpha.value * KWH_TO_AMP_PERIODS, 'beta': beta.value, 'gamma': gamma.value, 'delta': delta.value,
            'epsilon': epsilon.value, 'solar_rates': rates_s, 'grid_rates': rates_g, 'rates': rates, 'peak': peak,
            'cost': prob.value, 'prob': prob}


def calculate_costs(results, tou_prices, demand_charge):
    """ Calculate costs paid to the utility.

    Args:
        results (dict): Output of get_dual_prices or optimal_schedule.
        tou_prices (np.array[float]): List of energy prices. [$/kWh]
        demand_charge (float): Demand charge. [$/kW]

    Returns:
        dict:
            "energy_cost" (float): Cost of energy.
            "demand_charge" (float): Demand charge.
            "total_cost" (float): Sum of energy_cost and demand_charge.
    """
    costs = dict()
    costs['energy_cost'] = tou_prices.T @ (np.sum(results['rates'], axis=0) / KWH_TO_AMP_PERIODS)
    costs['demand_charge'] = demand_charge * results['peak'] / KW_TO_AMPS
    costs['total_cost'] = costs['energy_cost'] + costs['demand_charge']
    return costs


def calculate_session_revenue(results, sessions, infrastructure, tou_prices):
    """ Calculate revenues from users, broken down by type.

    Args:
        results (dict): Output of get_dual_prices or optimal_schedule.
        sessions (List[SessionInfo]): Sessions to determine prices for.
        infrastructure (InfrastructureInfo): Charging infrastructure
            description.
        tou_prices (np.array[float]): List of energy prices. [$/kWh]

    Returns:
        dict
    """
    revenues = dict()
    network_congestion = np.abs(infrastructure.constraint_matrix).T @ results['beta']
    for i, s in enumerate(sessions):
        r = {}
        j = infrastructure.get_station_index(s.station_id)
        a, d = s.arrival, s.departure
        user_rates = results['rates'][j, a:d]
        r['energy'] = tou_prices[a:d] @ (user_rates / KWH_TO_AMP_PERIODS)
        r['demand_charge'] = results['delta'][a:d] @ user_rates
        r['charger_congestion'] = results['gamma'][j, a:d] @ user_rates
        r['network_congestion'] = network_congestion[j, a:d] @ user_rates
        r['session_price'] = -results['alpha'][i] * (np.sum(user_rates) /
                                                     KWH_TO_AMP_PERIODS)
        r['total'] = r['energy'] + r['demand_charge'] + r['charger_congestion'] + r['network_congestion']
        r['energy_delivered'] = user_rates.sum() / KWH_TO_AMP_PERIODS
        r['energy_requested'] = s.requested_energy
        revenues[s.session_id] = r
    return revenues


def calculate_aggregate_revenue(session_rev):
    """ Aggregate per session revenues into aggregate revenue from each
            source.
    """
    keys = ['energy', 'demand_charge', 'charger_congestion', 'network_congestion', 'session_price', 'total',
            'energy_delivered', 'energy_requested']
    agg = {key: 0 for key in keys}
    for rev in session_rev.values():
        for k in rev:
            agg[k] += rev[k]
    return agg
