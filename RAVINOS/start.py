import ravinos
import os
cfg = ravinos.get_config()

pool = cfg['coins'][0]['pools'][0]
wallet = pool['user']
url = pool['url']
args = ' '.join(cfg['args'])
miner_path = os.path.join(cfg['miner_dir'], 'miner-linux')
cmd = miner_path + ' --stats --stats-devices %s %s' % (url, wallet)
if len(args) > 0:
    cmd += ' ' + args

ravinos.run(cmd)
