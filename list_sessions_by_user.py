from acnportal import acndata
from datetime import datetime
from collections import defaultdict
import pytz
import json

TIMEZONE = pytz.timezone('America/Los_Angeles')


def get_sessions_by_user(start, end, site):
    API_KEY = 'DEMO_TOKEN'
    dc = acndata.DataClient(API_KEY)
    start_time = TIMEZONE.localize(datetime.strptime(start, '%m-%d-%Y'))
    end_time = TIMEZONE.localize(datetime.strptime(end, '%m-%d-%Y'))
    sessions = dc.get_sessions_by_time(site, start_time, end_time)
    sessions_by_user = defaultdict(list)
    for s in sessions:
        if s['userID'] is not None:
            sessions_by_user[s['userID']].append(s['sessionID'])
    return dict(sessions_by_user)


if __name__ == '__main__':
    dates = [f'{m}-1-2019' for m in range(1, 12)]
    sites = ['caltech', 'jpl']
    sessions_by_user = dict()
    for site in sites:
        sessions_by_user[site] = dict()
        for i in range(len(dates)-1):
            start = dates[i]
            end = dates[i+1]
            sessions_by_user[site][start] = get_sessions_by_user(start, end, site)
    with open('pricing_wo_solar/sessions_by_user.json', 'w') as f:
        json.dump(sessions_by_user, f)