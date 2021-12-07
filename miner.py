import base64
import hashlib
import logging
import requests
import sha256
import sys
import time
import numpy as np
import pyopencl as cl
from threading import Thread, RLock
from urllib.parse import urljoin

pool_url = 'https://next.ton-pool.com'
wallet = 'EQBoG6BHwfFPTEUsxXW8y0TyHN9_5Z1_VIb2uctCd-NDmCbx'

share_count = 0
share_lock = RLock()
cur_task = None
task_lock = RLock()

logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')


def count_shares(num):
    global share_count
    with share_lock:
        share_count += num


def report_share(input, giver):
    r = requests.post(urljoin(pool_url, '/submit'), json={'inputs': [input], 'giver': giver, 'miner_addr': wallet})
    print(r.status_code, r.text)


def load_task():
    global cur_task
    try:
        r = requests.get(urljoin(pool_url, '/job'), timeout=10)
    except:
        return False
    r = r.json()

    wallet = base64.urlsafe_b64decode(r['wallet'])
    assert wallet[1] * 4 % 256 == 0
    input = b'\0\xf2Mine\0' + r['expire'].to_bytes(4, 'big') + wallet[2:34] + b'\0' * 32 + bytes.fromhex(r['seed']) + b'\0' * 32
    complexity = bytes.fromhex(r['complexity'])

    hash_state = np.array(sha256.generate_hash(input[:64])).astype(np.uint32)
    suffix = bytes(input[64:]) + b'\x80'
    suffix_arr = []
    for j in range(0, 60, 4):
        suffix_arr.append(int.from_bytes(suffix[j:j + 4], 'big'))
    suffix_np = np.array(suffix_arr).astype(np.uint32)
    with task_lock:
        cur_task = [0, input, r['giver'], complexity, hash_state, suffix_arr, suffix_np]


def update_task():
    while True:
        time.sleep(10)
        load_task()


def get_task(iterations):
    with task_lock:
        global_it, input, giver, complexity, hash_state, suffix_arr, suffix_np = cur_task
        cur_task[0] += 256
    return input, giver, complexity, suffix_arr, global_it, np.concatenate((np.array([iterations, global_it]).astype(np.uint32), hash_state, suffix_np))


class Worker:
    def __init__(self, device, program):
        self.context = cl.Context(devices=[device], dev_type=None)
        self.queue = cl.CommandQueue(self.context)
        self.kernel = cl.Program(self.context, program).build().hash_solver
        threads = device.max_compute_units * device.max_work_group_size
        if device.type & 4 == 0:
            threads = device.max_work_group_size
        self.threads = threads
        self.iterations = 131072

    def run(self):
        mf = cl.mem_flags
        while True:
            input, giver, complexity, suffix_arr, global_it, args = get_task(self.iterations)
            args_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args)
            res_g = cl.Buffer(self.context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=np.full(2048, 0xffffffff, np.uint32))
            self.kernel(self.queue, (self.threads,), None, args_g, res_g)
            res = np.empty(2048, np.uint32)
            e = cl.enqueue_copy(self.queue, res, res_g, is_blocking=False)

            while e.get_info(cl.event_info.COMMAND_EXECUTION_STATUS) != cl.command_execution_status.COMPLETE:
                time.sleep(0.005)

            os = list(np.where(res != 0xffffffff))[0]
            if len(os):
                for j in range(0, len(os), 2):
                    p = os[j]
                    assert os[j + 1] == p + 1
                    a = res[p]
                    b = res[p + 1]
                    suf = suffix_arr[:]
                    suf[0] ^= b
                    suf[12] ^= b
                    suf[1] ^= a
                    suf[13] ^= a
                    suf[2] ^= global_it
                    suf[14] ^= global_it
                    input_new = input[:64]
                    for x in suf:
                        input_new += int(x).to_bytes(4, 'big')
                    h = hashlib.sha256(input_new[:123]).digest()
                    if h < complexity:
                        logging.info('FOUND: ' + h.hex())
                        Thread(target=report_share, args=(input_new[:123].hex(), giver)).start()
            count_shares(self.threads * self.iterations)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python miner.py [pool addr] [wallet address]')
        exit()
    pool_url = sys.argv[1]
    wallet = sys.argv[2]
    try:
        r = requests.get(urljoin(pool_url, '/wallet/' + wallet), timeout=10)
    except Exception as e:
        print('failed to connect to pool:', e)
        exit()
    r = r.json()
    if 'ok' not in r:
        print('please check your wallet address:', r['msg'])
        exit()
    load_task()
    Thread(target=update_task).start()

    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()

    prog = open('sha256.cl').read()
    for device in devices:
        w = Worker(device, prog)
        Thread(target=w.run).start()

    ss = []
    ss.append((time.time(), share_count))
    while True:
        time.sleep(5)
        ss.append((time.time(), share_count))
        if len(ss) > 13:
            ss.pop(0)
        ct = ss[-1][0] - ss[0][0]
        logging.info('hashrate: %.2fMH/s in %.2fs' % (((ss[-1][1] - ss[0][1]) / ct / 10**6), ct))
