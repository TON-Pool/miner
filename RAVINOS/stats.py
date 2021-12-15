import ravinos
import os
import json

cfg = ravinos.get_config()
stats_ravin = ravinos.get_stats()

with open(os.path.join(cfg['work_dir'], 'stats.json')) as f:
    stats = json.loads(f.read())
with open(os.path.join(cfg['work_dir'], 'devices.json')) as f:
    devices = json.loads(f.read())

rates = [(stats['rates'][i], devices[i])for i in range(len(devices))]

for mpu in stats_ravin['mpu']:
    pos = None
    for i in range(len(rates)):
        if rates[i][1]['bus'] == mpu['pci_id']:
            pos = i
            break
    if pos is not None:
        mpu['hash_rate1'] = rates[pos][0] * 1e6
        rates.pop(pos)
    else:
        mpu['hash_rate1'] = 0
if len(rates):
    rates_sum = sum(x[0]for x in rates) * 1e6
    flag = False
    for mpu in stats_ravin['mpu']:
        if mpu['hash_rate1'] == 0:
            mpu['hash_rate1'] = rates_sum
            flag = True
            break
    if not flag:
        stats_ravin['mpu'][0]['hash_rate1'] += rates_sum

stats_ravin['shares'] = {
    'accpeted': stats['accepted'],
    'rejected': stats['rejected'],
    'invalid': 0,
}

ravinos.set_stats(stats_ravin)
