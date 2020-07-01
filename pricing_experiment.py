"""
Pricing EV Charging Services with Demand Charge
Zachary Lee
Created 4-9-2020
Last Updated: 4-9-2020

In this experiment we use the dual of an offline scheduling optimization to calculate
the prices paid by each user for each session.

For simplicity we will use the SessionInfo and InfrastructureInfo objects from adacharge.
"""

import os
import json
import pickle
import numpy as np
from acnportal import acnsim
import pytz
from datetime import datetime
from adacharge import get_active_sessions
from acnportal.acnsim.network.sites import caltech_acn, jpl_acn
from acnportal.signals import tariffs
from utils import get_infrastructure
from pricing_rule import *

# EXPERIMENT_DIR = 'results/max_len_144/costs_and_prices_offline_dual_with_laxity_max_len_144/'
TIMEZONE = pytz.timezone('America/Los_Angeles')
PERIOD = 5  # minute
VOLTAGE = 208  # volts
KW_TO_AMPS = 1000 / 208
KWH_TO_AMP_PERIODS = KW_TO_AMPS * (60 / 5)
MAX_LEN = 144
FORCE_FEASIBLE = True


def get_sessions(start, end, site):
    """ Get charging sessions via the ACN-Data api. """
    API_KEY = 'DEMO_TOKEN'
    default_battery_power = 32 / KW_TO_AMPS
    start_time = TIMEZONE.localize(datetime.strptime(start, '%m-%d-%Y'))
    end_time = TIMEZONE.localize(datetime.strptime(end, '%m-%d-%Y'))
    evs = acnsim.acndata_events.get_evs(API_KEY, site, start_time, end_time, PERIOD, VOLTAGE, default_battery_power,
                                        force_feasible=FORCE_FEASIBLE, max_len=MAX_LEN)
    return get_active_sessions(evs, 0)


def run_and_store(start, end, site, tariff_name, primal=True):
    """ Run experiment and store results in files. """
    path = EXPERIMENT_DIR + f'{tariff_name}/{site}/{start}/'
    if os.path.exists(path + 'results.pkl'):
        print(f'Already Run - {path}')
        return

    # Get charging sessions
    start_time = TIMEZONE.localize(datetime.strptime(start, '%m-%d-%Y'))
    charging_sessions = get_sessions(start, end, site)
    for i in range(len(charging_sessions)):
        charging_sessions[i].requested_energy -= 1e-4
    max_t = max(sess.departure for sess in charging_sessions)

    # Get charging infrastructure limits
    if site == 'caltech':
        infrastructure_info = get_infrastructure(caltech_acn(basic_evse=True, voltage=VOLTAGE))
    elif site == 'jpl':
        infrastructure_info = get_infrastructure(jpl_acn(basic_evse=True, voltage=VOLTAGE))
    else:
        raise ValueError(f'Invalid site option - {site}')

    # Get tariff
    tariff = tariffs.TimeOfUseTariff(tariff_name)
    prices = np.array(tariff.get_tariffs(start_time, max_t, PERIOD))
    dc = tariff.get_demand_charge(start_time)

    if primal:
        results = primal_scheduling_and_pricing(charging_sessions,
                                                infrastructure_info,
                                                tariff,
                                                PERIOD,
                                                start_time)
    else:
        results = dual_scheduling_and_pricing(charging_sessions,
                                              infrastructure_info,
                                              prices,
                                              dc)
    session_rev = calculate_session_revenue(results,
                                            charging_sessions,
                                            infrastructure_info,
                                            prices)

    agg_rev = calculate_aggregate_revenue(session_rev)
    costs = calculate_costs(results, prices, dc)

    # Store Results
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'results.pkl', 'wb') as outfile:
        pickle.dump(results, outfile)

    with open(path + 'session_rev.json', 'w') as outfile:
        json.dump(session_rev, outfile)

    with open(path + 'agg_rev.json', 'w') as outfile:
        json.dump(agg_rev, outfile)

    with open(path + 'costs.json', 'w') as outfile:
        json.dump(costs, outfile)

    print(f'Done - {path}')


# Run experiments.
EXPERIMENT_DIR = 'test_results2/'
if __name__ == '__main__':
    dates = [f'{m}-1-2019' for m in range(1, 3)]
    sites = ['caltech', 'jpl']
    tariff_name = 'sce_tou_ev_4_march_2019'
    for site in sites:
        for i in range(len(dates)-1):
            start = dates[i]
            end = dates[i+1]
            run_and_store(start, end, site, tariff_name, primal=False)