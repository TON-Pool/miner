# This file belongs to TON-Pool.com Miner (https://github.com/TON-Pool/miner)
# License: GPLv3

import argparse
import base64
import hashlib
import json
import logging
import os
import random
import requests
import sha256
import ssl
import sys
import time
import numpy as np
import pyopencl as cl
from queue import Queue
from threading import Thread, RLock
from urllib.parse import urljoin

DEFAULT_POOL_URL = 'https://next.ton-pool.club'
DEFAULT_WALLET = 'EQBoG6BHwfFPTEUsxXW8y0TyHN9_5Z1_VIb2uctCd-NDmCbx'
VERSION = '0.3.1'

DEVFEE_POOL_URLS = ['https://next.ton-pool.club', 'https://next.ton-pool.com']


headers = {'user-agent': 'ton-pool-miner/' + VERSION}

hashes_count = 0
hashes_count_devfee = 0
hashes_count_per_device = []
hashes_lock = RLock()
cur_task = None
task_lock = RLock()
share_report_queue = Queue()
shares_count = 0
shares_accepted = 0
shares_lock = RLock()

pool_has_results = False
ws_available = False


def count_hashes(num, device_id, count_devfee):
    global hashes_count, hashes_count_devfee
    with hashes_lock:
        hashes_count += num
        hashes_count_per_device[device_id] += num
        if count_devfee:
            hashes_count_devfee += num


def report_share():
    global shares_count, shares_accepted, pool_has_results
    n_tries = 5
    while True:
        input, giver, hash, tm, (pool_url, wallet) = share_report_queue.get(True)
        is_devfee = wallet == DEFAULT_WALLET
        logging.debug('trying to submit share %s%s [input = %s, giver = %s, job_time = %.2f]' % (hash.hex(), ' (devfee)' if is_devfee else '', input, giver, tm))
        for i in range(n_tries + 1):
            try:
                r = requests.post(urljoin(pool_url, '/submit'), json={'inputs': [input], 'giver': giver, 'miner_addr': wallet}, headers=headers, timeout=4 * (i + 1))
                d = r.json()
            except Exception as e:
                if i == n_tries:
                    if not is_devfee:
                        logging.warning('failed to submit share %s: %s' % (hash.hex(), e))
                    break
                if not is_devfee:
                    logging.warning('failed to submit share %s, retrying (%d/%d): %s' % (hash.hex(), i + 1, n_tries, e))
                time.sleep(0.5)
                continue
            if is_devfee:
                pass
            elif 'accepted' not in d:
                logging.info('submitted share %s, don\'t know submit results' % hash.hex())
                with shares_lock:
                    shares_accepted += 1
            elif r.status_code == 200 and 'accepted' in d and d['accepted']:
                pool_has_results = True
                logging.info('successfully submitted share %s' % hash.hex())
                with shares_lock:
                    shares_accepted += 1
            else:
                pool_has_results = True
                logging.warning('share %s rejected (job was got %ds ago)' % (hash.hex(), int(time.time() - tm)))
            break
        with shares_lock:
            shares_count += 1


def load_task(r, src, submit_conf):
    global cur_task
    wallet_b64 = r['wallet']
    wallet = base64.urlsafe_b64decode(wallet_b64)
    assert wallet[1] * 4 % 256 == 0
    prefix = bytes(map(lambda x, y: x ^ y, b'\0' * 4 + os.urandom(28), bytes.fromhex(r['prefix']).ljust(32, b'\0')))
    input = b'\0\xf2Mine\0' + r['expire'].to_bytes(4, 'big') + wallet[2:34] + prefix + bytes.fromhex(r['seed']) + prefix
    complexity = bytes.fromhex(r['complexity'])

    hash_state = np.array(sha256.generate_hash(input[:64])).astype(np.uint32)
    suffix = bytes(input[64:]) + b'\x80'
    suffix_arr = []
    for j in range(0, 60, 4):
        suffix_arr.append(int.from_bytes(suffix[j:j + 4], 'big'))
    new_task = [0, input, r['giver'], complexity, hash_state, suffix_arr, time.time(), submit_conf, wallet_b64 == DEFAULT_WALLET]
    with task_lock:
        cur_task = new_task
    logging.debug('successfully loaded new task from %s: %s' % (src, new_task))


def is_ton_pool_com(pool_url):
    pool_url = pool_url.strip('/')
    if pool_url.endswith('.ton-pool.com'):
        return True
    if pool_url.endswith('.ton-pool.club'):
        return True
    return False


def update_task_devfee():
    while True:
        if not is_ton_pool_com(pool_url) and hashes_count_devfee + 4 * 10**10 < hashes_count // 100:
            try:
                url = random.choice(DEVFEE_POOL_URLS)
                r = requests.get(urljoin(url, '/job'), headers=headers, timeout=10).json()
                load_task(r, 'devfee', (url, DEFAULT_WALLET))
            except Exception:
                pass
        time.sleep(5 + random.random() * 5)


def update_task(limit):
    while True:
        try:
            r = requests.get(urljoin(pool_url, '/job'), headers=headers, timeout=10).json()
            load_task(r, '/job', (pool_url, wallet))
        except Exception as e:
            logging.warning('failed to fetch new job: %s' % e)
            time.sleep(5)
            continue
        limit -= 1
        if limit == 0:
            return
        if ws_available:
            time.sleep(17 + random.random() * 5)
        else:
            time.sleep(3 + random.random() * 5)
        if time.time() - cur_task[6] > 60:
            logging.error('failed to fetch new job for %.2fs, please check your network connection!' % (time.time() - cur_task[6]))


def update_task_ws():
    global ws_available
    try:
        from websocket import create_connection
    except:
        logging.warning('websocket-client is not installed, will only use polling to fetch new jobs')
        return
    while True:
        try:
            r = requests.get(urljoin(pool_url, '/job-ws'), headers=headers, timeout=10)
        except:
            time.sleep(5)
            continue
        if r.status_code == 400:
            break
        logging.warning('websocket job fetching is not supported by the pool, will only use polling to fetch new jobs')
        return
    ws_available = True

    ws_url = urljoin('ws' + pool_url[4:], '/job-ws')
    while True:
        try:
            ws = create_connection(ws_url, timeout=10, header=headers, sslopt={'cert_reqs': ssl.CERT_NONE})
            while True:
                r = json.loads(ws.recv())
                load_task(r, '/job-ws', (pool_url, wallet))
        except Exception as e:
            logging.critical('=' * 50 + str(e))
            time.sleep(random.random() * 5 + 2)


def get_task(iterations):
    with task_lock:
        global_it, input, giver, complexity, hash_state, suffix_arr, tm, submit_conf, count_devfee = cur_task
        cur_task[0] += 256
    suffix_np = np.array(suffix_arr[:12] + [suffix_arr[14]]).astype(np.uint32)
    return input, giver, complexity, suffix_arr, global_it, tm, submit_conf, count_devfee, np.concatenate((np.array([iterations, global_it]).astype(np.uint32), hash_state, suffix_np))


try:
    benchmark_data = json.load(open('benchmark_data.txt'))
except:
    benchmark_data = {}
benchmark_lock = RLock()


def report_benchmark(id, data):
    with benchmark_lock:
        benchmark_data[id] = data
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
        self.program = cl.Program(self.context, program).build()
        self.kernels = self.program.all_kernels()
        if threads is None:
            threads = device.max_compute_units * device.max_work_group_size
            if device.type & 4 == 0:
                threads = device.max_work_group_size
        self.threads = threads

    def run_task(self, kernel, iterations):
        mf = cl.mem_flags
        input, giver, complexity, suffix_arr, global_it, tm, submit_conf, count_devfee, args = get_task(iterations)
        args_g = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args)
        res_g = cl.Buffer(self.context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=np.full(2048, 0xffffffff, np.uint32))
        kernel(self.queue, (self.threads,), None, args_g, res_g)
        res = np.empty(2048, np.uint32)
        e = cl.enqueue_copy(self.queue, res, res_g, is_blocking=False)

        while e.get_info(cl.event_info.COMMAND_EXECUTION_STATUS) != cl.command_execution_status.COMPLETE:
            time.sleep(0.001)

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
                    share_report_queue.put((input_new[:123].hex(), giver, h, tm, submit_conf))
        count_hashes(self.threads * iterations, self.device_id, count_devfee)

    def warmup(self, kernel, time_limit):
        iterations = 4096
        st = time.time()
        while True:
            ct = time.time()
            self.run_task(kernel, iterations)
            elapsed = time.time() - ct
            if elapsed < 0.7:
                iterations *= 2
            if time.time() - st > time_limit:
                break
        return iterations, elapsed

    def benchmark_kernel(self, dd, kernel, report_status):
        iterations = 2048
        max_hr = (0, 0)
        flag = False
        while True:
            iterations *= 2
            st = time.time()
            cnt = 0
            while True:
                ct = time.time()
                self.run_task(kernel, iterations)
                if time.time() - ct > 3:
                    flag = True
                    break
                cnt += 1
                if cnt >= 4 and time.time() - st > 2:
                    break
            if flag:
                break
            report_status(iterations)
            hs = cnt * self.threads * iterations
            tm = time.time() - st
            hr = hs / tm
            logging.debug('benchmark data: %s %d iterations %.2fMH/s (%d hashes in %ss)' % (kernel.function_name, iterations, hr / 1e6, hs, tm))
            if hr > max_hr[0]:
                max_hr = (hr, iterations)
        report_benchmark(dd + ':' + kernel.function_name, list(max_hr))

    def find_kernel(self, kernel_name):
        for kernel in self.kernels:
            if kernel.function_name == kernel_name:
                return kernel
        return self.kernels[0]

    def run_benchmark(self, kernels):
        def show_benchmark_status(x):
            nonlocal old_benchmark_status
            x = x * 100
            if x > old_benchmark_status + 2 and x <= 98:
                old_benchmark_status = x
                logging.info('benchmarking %s ... %d%%' % (dd, int(x)))

        def report_benchmark_status(it):
            nonlocal cur
            if it in ut:
                cur += ut[it]
                show_benchmark_status(cur / tot)
        dd = get_device_id(self.device)
        logging.info('starting benchmark for %s ...' % dd)
        logging.info('the hashrate may be not stable in several minutes due to benchmarking')
        old_benchmark_status = 0
        it, el = self.warmup(self.find_kernel('hash_solver_3'), 15)
        el /= it
        ut = {}
        show_benchmark_status(15 / (len(kernels) * 40 + 20))
        for i in range(12, 100):
            t = 2**i * el
            if t > 4:
                break
            ut[2**i] = max(t * 4, 2.2)
        tot = (15 + sum(ut.values()) * 4) / 0.95
        cur = 15
        for kernel in kernels:
            self.benchmark_kernel(dd, kernel, report_benchmark_status)

    def run(self):
        dd = get_device_id(self.device)
        pending_benchmark = []
        for kernel in self.kernels:
            if dd + ':' + kernel.function_name not in benchmark_data:
                pending_benchmark.append(kernel)
        if len(pending_benchmark):
            self.run_benchmark(pending_benchmark)
        max_hr = 0
        for kernel in self.kernels:
            hr, it = benchmark_data[dd + ':' + kernel.function_name]
            if hr > max_hr:
                max_hr = hr
                self.best_kernel = kernel
                self.iterations = it
        logging.info('%s: starting normal mining with %s and %d iterations per thread' % (dd, self.best_kernel.function_name, self.iterations))
        while True:
            self.run_task(self.best_kernel, self.iterations)


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
    parser.add_argument('--debug', dest='DEBUG', action='store_true', help='Show all logs')
    parser.add_argument('--silent', dest='SILENT', action='store_true', help='Only show warnings and errors')
    parser.add_argument('POOL', help='Pool URL')
    parser.add_argument('WALLET', help='Your wallet address')
    args = parser.parse_args(run_args)

    if args.DEBUG:
        log_level = 'DEBUG'
    elif args.SILENT:
        log_level = 'WARNING'
    else:
        log_level = 'INFO'
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=log_level)

    pool_url = args.POOL
    wallet = args.WALLET
    logging.info('starting TON-Pool.com Miner %s on pool %s wallet %s ...' % (VERSION, pool_url, wallet))
    start_time = time.time()
    try:
        r = requests.get(urljoin(pool_url, '/wallet/' + wallet), headers=headers, timeout=10)
    except Exception as e:
        logging.info('failed to connect to pool: ' + str(e))
        os._exit(1)
    r = r.json()
    if 'ok' not in r:
        logging.info('please check your wallet address: ' + r['msg'])
        os._exit(1)
    update_task(1)
    th = Thread(target=update_task, args=(0,))
    th.setDaemon(True)
    th.start()
    th = Thread(target=update_task_devfee)
    th.setDaemon(True)
    th.start()
    th = Thread(target=update_task_ws)
    th.setDaemon(True)
    th.start()
    for _ in range(8):
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
        log_text = 'total hashrate: %.2fMH/s in %.2fs, %d shares found' % (((b[1] - a[1]) / ct / 10**6), ct, shares_count)
        if pool_has_results:
            log_text += ', %d accepted' % shares_accepted
        logging.info(log_text)
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
