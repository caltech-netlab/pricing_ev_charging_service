import warnings
from datetime import datetime
import pytz
import numpy as np
import os
import json
import cvxpy as cp

from acnportal import acnsim, algorithms
from acnportal.signals import tariffs
from acnportal.acnsim import analysis
import adacharge
from adacharge import AdaptiveSchedulingAlgorithm, ObjectiveComponent

DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
TIMEZONE = pytz.timezone('America/Los_Angeles')
PERIOD = 5  # minutes
VOLTAGE = 208  # volts
KW_TO_AMPS = 1000 / 208
KWH_TO_AMP_PERIODS = KW_TO_AMPS * (60 / 5)
MAX_LEN = 144
IDEAL_BATTERY = True
BASIC_EVSE = True
FORCE_FEASIBLE = True
VERBOSE = True

def get_events(start, end, voltage, site, period, ideal_battery, force_feasible, max_len):
    API_KEY = 'DEMO_TOKEN'
    start_time = TIMEZONE.localize(datetime.strptime(start, '%m-%d-%Y'))
    end_time = TIMEZONE.localize(datetime.strptime(end, '%m-%d-%Y'))
    default_battery_power = 32 * voltage / 1e3  # kW
    if ideal_battery:
        battery_params=None
    else:
        battery_params={'type': acnsim.Linear2StageBattery,
                        'capacity_fn': acnsim.models.battery.batt_cap_fn}
    events = acnsim.acndata_events.generate_events(API_KEY, site, start_time, end_time, period, voltage,
                                                    default_battery_power, force_feasible=force_feasible,
                                                    max_len=max_len, battery_params=battery_params)
    return events


def days_remaining_scale_demand_charge(rates, infrastructure, interface, month,
                                       baseline_peak=0, **kwargs):
    day_index = interface.current_time // ((60 / interface.period) * 24)
    days_in_month = DAYS_IN_MONTH[month]
    day_index = min(day_index, days_in_month - 1)
    scale = 1 / (days_in_month - day_index)
    dc = adacharge.demand_charge(rates, infrastructure, interface, baseline_peak,
                                 **kwargs)
    return scale * dc


# def days_remaining_peak(rates, infrastructure, demand_charges, current_time, baseline_peak=0, prev_peak=0, **kwargs):
#     prev_peak = prev_peak / KW_TO_AMPS  # prev_peak is currently in A but other peaks in kW
#     agg_power = adacharge.aggregate_power(rates, infrastructure)
#     max_power = cp.max(agg_power)
#     return demand_charges[current_time] * cp.maximum(max_power, baseline_peak, prev_peak)


def experiment(start, end, site, tariff_name, peak_hint=False, quick_charge_coeff=0., equal_share_coeff=0.):
    OPTIMAL_PEAKS = {}
    OPTIMAL_PEAKS['caltech'] = [38.41, 46.26, 39.19, 33.54, 34.04, 37.71, 31.62, 30.36]
    OPTIMAL_PEAKS['jpl'] = [85.14, 89.95, 130.15, 96.18, 106.87, 95.25, 111.28, 125.64]

    start_time = TIMEZONE.localize(datetime.strptime(start, '%m-%d-%Y'))
    events = get_events(start, end, VOLTAGE, site, PERIOD, IDEAL_BATTERY, FORCE_FEASIBLE, MAX_LEN)
    max_t = int((60 / PERIOD) * 24 * 35)

    # Get tariff
    tariff = tariffs.TimeOfUseTariff(tariff_name)
    month = int(start.split('-')[0])

    peak_baseline = .75 * OPTIMAL_PEAKS[site][month-1] if peak_hint else 0
    obj = [ObjectiveComponent(adacharge.tou_energy_cost, 1),
           ObjectiveComponent(days_remaining_scale_demand_charge, 1,
                              {'month': month, 'baseline_peak': peak_baseline}),
           ObjectiveComponent(adacharge.total_energy, 200)]
    if quick_charge_coeff > 0:
        obj.append(ObjectiveComponent(adacharge.quick_charge, quick_charge_coeff))
    if equal_share_coeff > 0:
        obj.append(ObjectiveComponent(adacharge.equal_share, equal_share_coeff))
    algorithm = AdaptiveSchedulingAlgorithm(obj, solver=SOLVER, constraint_type='LINEAR')

    # Get charging infrastructure limits
    if site == 'caltech':
        cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=VOLTAGE)
    elif site == 'jpl':
        cn = acnsim.sites.jpl_acn(basic_evse=True, voltage=VOLTAGE)
    else:
        raise ValueError(f'Invalid site option - {site}')

    signals = {'tariff': tariff}
    sim = acnsim.Simulator(cn, algorithm, events, start_time, period=PERIOD,
                           verbose=VERBOSE, signals=signals)
    return sim


def calc_metrics(sim):
    metrics = {
        'proportion_delivered': analysis.proportion_of_energy_delivered(sim) * 100,
        'demands_fully_met': analysis.proportion_of_demands_met(sim) * 100,
        'peak_current': sim.peak,
        'demand_charge': analysis.demand_charge(sim),
        'energy_cost': analysis.energy_cost(sim)
    }
    metrics['total_cost'] = metrics['demand_charge'] + metrics['energy_cost']
    return metrics


def run_and_store(start, end, site, tariff_name, peak_hint=False, quick_charge_coeff=0., equal_share_coeff=0.):
    path = OUTPUT_DIR + f'{tariff_name}/{site}/{start}:{end}/'
    if os.path.exists(path + 'sim.json'):
        print(f'Already Run - {path}...')
        return

    if not os.path.exists(path):
        os.makedirs(path)
    sim = experiment(start, end, site, tariff_name, peak_hint, quick_charge_coeff, equal_share_coeff)
    sim.run()
    sim.to_json(path + 'sim.json')
    with open(path + 'metrics.json', 'w') as outfile:
        json.dump(calc_metrics(sim), outfile)
    print(f'Done - {path}')


if __name__ == '__main__':
    SOLVER = 'ECOS'
    OUTPUT_DIR = 'results/max_len_144/online_cost_minimization-peak_hint-quick_charge_1e-4-max_len_144-rho_200/'
    dates = [f'{m}-1-2019' for m in range(1, 9)]
    sites = ['caltech', 'jpl']
    tariff_name = 'sce_tou_ev_4_march_2019'
    for site in sites:
        for i in range(len(dates)-1):
            start = dates[i]
            end = dates[i+1]
            # run_and_store(start, end, site, tariff_name, peak_hint=False)
            # run_and_store(start, end, site, tariff_name, peak_hint=True)
            run_and_store(start, end, site, tariff_name, peak_hint=True, quick_charge_coeff=1e-4)
            # run_and_store(start, end, site, tariff_name, peak_hint=True, quick_charge_coeff=1e-4, equal_share_coeff=-1e-7)