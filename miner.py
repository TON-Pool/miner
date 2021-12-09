# This file belongs to TON-Pool.com Miner (https://github.com/TON-Pool/miner)
# License: GPLv3

import argparse
import base64
import hashlib
import json
import logging
import os
import requests
import sha256
import sys
import time
import numpy as np
import pyopencl as cl
from queue import Queue
from threading import Thread, RLock
from urllib.parse import urljoin

pool_url = 'https://next.ton-pool.com'
wallet = 'EQBoG6BHwfFPTEUsxXW8y0TyHN9_5Z1_VIb2uctCd-NDmCbx'
VERSION = '0.2'

hashes_count = 0
hashes_count_per_device = []
hashes_lock = RLock()
cur_task = None
task_lock = RLock()
share_report_queue = Queue()
shares_count = 0
shares_accepted = 0

logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')


def count_hashes(num, device_id):
    global hashes_count
    with hashes_lock:
        hashes_count += num
        hashes_count_per_device[device_id] += num


def report_share():
    global shares_count, shares_accepted
    n_tries = 5
    while True:
        input, giver, hash = share_report_queue.get(True)
        for i in range(n_tries + 1):
            try:
                r = requests.post(urljoin(pool_url, '/submit'), json={'inputs': [input], 'giver': giver, 'miner_addr': wallet}, timeout=10)
                d = r.json()
            except Exception as e:
                if i == n_tries:
                    logging.warning('failed to submit share %s: %s' % (hash.hex(), e))
                    break
                logging.warning('failed to submit share %s, retrying (%d/%d): %s' % (hash.hex(), i + 1, n_tries, e))
                time.sleep(0.5)
                continue
            if r.status_code == 200 and 'accepted' in d and d['accepted']:
                logging.info('successfully submitted share %s' % hash.hex())
                shares_accepted += 1
            else:
                logging.warning('share %s rejected, please check your network connection' % hash.hex())
            break
        shares_count += 1


def load_task():
    global cur_task
    try:
        r = requests.get(urljoin(pool_url, '/job'), timeout=10).json()
    except Exception as e:
        return False, e

    wallet = base64.urlsafe_b64decode(r['wallet'])
    assert wallet[1] * 4 % 256 == 0
    prefix = bytes(map(lambda x, y: x ^ y, b'\0' * 4 + os.urandom(28), bytes.fromhex(r['prefix']).ljust(32, b'\0')))
    input = b'\0\xf2Mine\0' + r['expire'].to_bytes(4, 'big') + wallet[2:34] + prefix + bytes.fromhex(r['seed']) + prefix
    complexity = bytes.fromhex(r['complexity'])

    hash_state = np.array(sha256.generate_hash(input[:64])).astype(np.uint32)
    suffix = bytes(input[64:]) + b'\x80'
    suffix_arr = []
    for j in range(0, 60, 4):
        suffix_arr.append(int.from_bytes(suffix[j:j + 4], 'big'))
    with task_lock:
        cur_task = [0, input, r['giver'], complexity, hash_state, suffix_arr]
    return True, None


def update_task():
    lst_ok = time.time()
    while True:
        time.sleep(10)
        st, e = load_task()
        if st:
            lst_ok = time.time()
        else:
            logging.warning('failed to fetch new job: %s' % e)
            if time.time() - lst_ok > 60:
                logging.error('failed to fetch new job for %.2fs, please check your network connection!' % (time.time() - lst_ok))


def get_task(iterations):
    with task_lock:
        global_it, input, giver, complexity, hash_state, suffix_arr = cur_task
        cur_task[0] += 256
    suffix_np = np.array(suffix_arr[:12] + [suffix_arr[14]]).astype(np.uint32)
    return input, giver, complexity, suffix_arr, global_it, np.concatenate((np.array([iterations, global_it]).astype(np.uint32), hash_state, suffix_np))


try:
    benchmark_data = json.load(open('benchmark_data.txt'))
except:
    benchmark_data = {}
benchmark_lock = RLock()


def report_benchmark(id, iterations):
    with benchmark_lock:
        benchmark_data[id] = iterations
        json.dump(benchmark_data, open('benchmark_data.txt', 'w'))


def get_device_id(device):
    name = device.name
    try:
        bus = device.get_info(0x4008)
        slot = device.get_info(0x4009)
        return name + ' on PCI bus %d slot %d' % (bus, slot)
    except cl.LogicError:
        pass
    try:
        topo = device.get_info(0x4037)
        return name + ' on PCI bus %d device %d function %d' % (topo.bus, topo.device, topo.function)
    except cl.LogicError:
        pass
    return name


class Worker:
    def __init__(self, device, program, threads, id):
        self.device = device
        self.device_id = id
        self.context = cl.Context(devices=[device], dev_type=None)
        self.queue = cl.CommandQueue(self.context)
        self.kernel = cl.Program(self.context, program).build().hash_solver
        if threads is None:
            threads = device.max_compute_units * device.max_work_group_size
            if device.type & 4 == 0:
                threads = device.max_work_group_size
        self.threads = threads
        self.iterations = 131072

    def run_task(self):
        mf = cl.mem_flags
        st = time.time()
        input, giver, complexity, suffix_arr, global_it, args = get_task(self.iterations)
        args_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args)
        res_g = cl.Buffer(self.context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=np.full(2048, 0xffffffff, np.uint32))
        self.kernel(self.queue, (self.threads,), None, args_g, res_g)
        res = np.empty(2048, np.uint32)
        e = cl.enqueue_copy(self.queue, res, res_g, is_blocking=False)

        while e.get_info(cl.event_info.COMMAND_EXECUTION_STATUS) != cl.command_execution_status.COMPLETE:
            time.sleep(0.005)
        elapsed = time.time() - st

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
                if h[:4] != b'\0\0\0\0':
                    logging.warning('hash integrity error, please check your graphics card drivers')
                if h < complexity:
                    share_report_queue.put((input_new[:123].hex(), giver, h))
        count_hashes(self.threads * self.iterations, self.device_id)
        return self.threads * self.iterations / elapsed, elapsed

    def run(self):
        dd = get_device_id(self.device)
        if dd not in benchmark_data:
            logging.info('benchmarking %s ...' % dd)
            logging.info('the hashrate may be not stable in one minute due to benchmarking')
            self.iterations = 2048
            max_hr = (0, 0)
            flag = False
            while True:
                self.iterations *= 2
                hrs = []
                for _ in range(5):
                    hr, tm = self.run_task()
                    if tm > 10:
                        flag = True
                        break
                    hrs.append(hr)
                if flag:
                    break
                hr = sum(hrs) / len(hrs)
                if hr > max_hr[0]:
                    max_hr = (hr, self.iterations)
                if self.iterations >= max_hr[1] * 8:
                    break
            self.iterations = max_hr[1]
            report_benchmark(dd, self.iterations)
        else:
            self.iterations = benchmark_data[dd]
        logging.info('%s: starting normal mining with %d iterations per thread' % (dd, self.iterations))
        while True:
            self.run_task()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('')
    if sys.argv[1] == 'info':
        print('TON-Pool.com Miner', VERSION)
        try:
            platforms = cl.get_platforms()
        except cl.LogicError:
            print('failed to get OpenCL platforms, check your graphics card drivers')
            os._exit(0)
        for i, platform in enumerate(platforms):
            print('Platform %d:' % i)
            for j, device in enumerate(platform.get_devices()):
                print('    Device %d: %s' % (j, get_device_id(device)))
        os._exit(0)
    if sys.argv[1] == 'run':
        run_args = sys.argv[2:]
    elif sys.argv[1].startswith('http') or sys.argv[1].startswith('-'):
        run_args = sys.argv[1:]
    else:
        print('TON-Pool.com Miner', VERSION)
        print('Usage: %s [pool url] [wallet address]' % sys.argv[0])
        print('Run "%s info" to check your system info' % sys.argv[0])
        print('Run "%s -h" to for detailed arguments' % sys.argv[0])
        os._exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='PLATFORM', help='Platform ID')
    parser.add_argument('-d', dest='DEVICE', help='Device ID')
    parser.add_argument('-t', dest='THREADS', help='Number of threads. This is applied for all devices.')
    parser.add_argument('--stats', dest='STATS', action='store_true', help='Dump stats to stats.json')
    parser.add_argument('POOL', help='Pool URL')
    parser.add_argument('WALLET', help='Your wallet address')
    args = parser.parse_args(run_args)

    pool_url = args.POOL
    wallet = args.WALLET
    logging.info('starting TON-Pool.com Miner %s on pool %s wallet %s ...' % (VERSION, pool_url, wallet))
    start_time = time.time()
    try:
        r = requests.get(urljoin(pool_url, '/wallet/' + wallet), timeout=10)
    except Exception as e:
        logging.info('failed to connect to pool: ' + str(e))
        os._exit(1)
    r = r.json()
    if 'ok' not in r:
        logging.info('please check your wallet address: ' + r['msg'])
        os._exit(1)
    load_task()
    th = Thread(target=update_task)
    th.setDaemon(True)
    th.start()
    th = Thread(target=report_share)
    th.setDaemon(True)
    th.start()

    platforms = cl.get_platforms()
    if args.PLATFORM is not None:
        t = int(args.PLATFORM)
        if t >= len(platforms):
            logging.info('wrong platform ID: %d' % t)
            os._exit(1)
        platforms = [platforms[t]]
    devices = []
    for platform in platforms:
        cur_devices = platform.get_devices()
        if args.DEVICE is not None:
            t = int(args.DEVICE)
            if t >= len(cur_devices):
                logging.info('wrong device ID: %d' % t)
                if args.PLATFORM is not None and len(platforms) > 1:
                    logging.info('you may want to specify a platform ID')
                os._exit(1)
            cur_devices = [cur_devices[t]]
        devices += cur_devices
    logging.info('total devices: %d' % len(devices))
    hashes_count_per_device = [0] * len(devices)

    path = os.path.dirname(os.path.abspath(__file__))
    try:
        prog = open(os.path.join(path, 'sha256.cl'), 'r').read() + '\n' + open(os.path.join(path, 'hash_solver.cl'), 'r').read()
    except:
        logging.info('failed to load opencl program')
        os._exit(1)
    for i, device in enumerate(devices):
        w = Worker(device, prog, args.THREADS, i)
        th = Thread(target=w.run)
        th.setDaemon(True)
        th.start()

    ss = []
    ss.append((time.time(), hashes_count, [0] * len(devices)))
    cnt = 0
    while True:
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            logging.info('exiting...')
            os._exit(0)
        ss.append((time.time(), hashes_count, hashes_count_per_device[:]))
        if len(ss) > 7:
            ss.pop(0)
        a, b = ss[-2], ss[-1]
        ct = b[0] - a[0]
        logging.info('total hashrate: %.2fMH/s in %.2fs, %d shares found, %d accepted' % (((b[1] - a[1]) / ct / 10**6), ct, shares_count, shares_accepted))
        cnt += 1
        if cnt >= 6 and cnt % 6 == 2:
            a, b = ss[0], ss[-1]
            ct = b[0] - a[0]
            logging.info('average hashrate in last minute: %.2fMH/s in %.2fs' % (((b[1] - a[1]) / ct / 10**6), ct))
        if (cnt < 8 or cnt % 6 == 2) and args.STATS:
            if cnt < 8:
                a, b = ss[-2], ss[-1]
            else:
                a, b = ss[0], ss[-1]
            ct = b[0] - a[0]
            rates = []
            for i in range(len(devices)):
                rates.append((b[2][i] - a[2][i]) / ct / 10**6)
            json.dump({
                'total': (b[1] - a[1]) / ct / 10**3,
                'rates': rates,
                'uptime': time.time() - start_time,
                'accepted': shares_accepted,
                'rejected': shares_count - shares_accepted,
            }, open('stats.json', 'w'))
