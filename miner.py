# This file belongs to TON-Pool.com Miner (https://github.com/TON-Pool/miner)
# Fork: https://github.com/MPCodeWriter21/miner
# License: GPLv3

import os
import ssl
import sys
import time
import json
import log21
import base64
import random
import sha256
import hashlib
import argparse
import requests
import numpy as np
import pyopencl as cl
from queue import Queue
from threading import Thread, RLock
from urllib.parse import urljoin


class Task:
    def __init__(self, global_it: int, input_: bytes, giver: str, complexity: bytes, hash_state: np.ndarray,
                 suffix_arr: list, load_time: float, submit_conf: tuple, is_devfee: bool):
        self.global_it: int = global_it
        self.input: bytes = input_
        self.giver: str = giver
        self.complexity: bytes = complexity
        self.hash_state: np.ndarray = hash_state
        self.suffix_arr: list = suffix_arr
        self.load_time: float = load_time
        self.submit_conf: tuple = submit_conf
        self.is_devfee: bool = is_devfee

    def list(self) -> list:
        return [self.global_it, self.input, self.giver, self.complexity, self.hash_state, self.suffix_arr,
                self.load_time, self.submit_conf, self.is_devfee]

    def __repr__(self):
        return '\n' + LYellow + 'global_it' + LRed + ':' + LWhite + str(self.global_it) + '\n' + \
               LYellow + 'input' + LRed + ': ' + LWhite + str(self.input) + '\n' + \
               LYellow + 'giver' + LRed + ': ' + LWhite + str(self.giver) + '\n' + \
               LYellow + 'complexity' + LRed + ': ' + LWhite + str(self.complexity) + '\n' + \
               LYellow + 'hash_state' + LRed + ': ' + LWhite + str(self.hash_state) + '\n' + \
               LYellow + 'suffix_arr' + LRed + ': ' + LWhite + str(self.suffix_arr) + '\n' + \
               LYellow + 'load_time' + LRed + ': ' + LWhite + str(self.load_time) + '\n' + \
               LYellow + 'submit_conf' + LRed + ': ' + LWhite + str(self.submit_conf) + '\n' + \
               LYellow + 'is_devfee' + LRed + ': ' + LWhite + str(self.is_devfee) + '\n'

    def __str__(self):
        return 'global_it:' + str(self.global_it) + ', ' + \
               'input: ' + str(self.input) + ', ' + \
               'giver: ' + str(self.giver) + ', ' + \
               'complexity: ' + str(self.complexity) + ', ' + \
               'hash_state: ' + str(self.hash_state) + ', ' + \
               'suffix_arr: ' + str(self.suffix_arr) + ', ' + \
               'load_time: ' + str(self.load_time) + ', ' + \
               'submit_conf: ' + str(self.submit_conf) + ', ' + \
               'is_devfee: ' + str(self.is_devfee)


# Colors
Cyan = log21.get_color('Cyan')
LCyan = log21.get_color('LightCyan')
LYellow = log21.get_color('LightYellow')
Red = log21.get_color('Red')
LRed = log21.get_color('LightRed')
Green = log21.get_color('Green')
LGreen = log21.get_color('LightGreen')
LBlue = log21.get_color('LightBlue')
LPink = log21.get_color('LightMagenta')
LWhite = log21.get_color('LightWhite')

logger = log21.get_logger('TON-Miner', level=log21.INFO)

DEFAULT_POOL_URL = 'https://next.ton-pool.club'
DEFAULT_WALLET = 'EQBoG6BHwfFPTEUsxXW8y0TyHN9_5Z1_VIb2uctCd-NDmCbx'
VERSION = '0.3.7'

DEVFEE_POOL_URLS = ['https://next.ton-pool.club', 'https://next.ton-pool.com']

headers = {'user-agent': 'ton-pool-miner/' + VERSION}

devfee: int = 1
hashes_count: int = 0
hashes_count_devfee: int = 0
hashes_count_per_device: list = []
hashes_lock: RLock = RLock()
cur_task: Task = None
task_lock: RLock = RLock()
share_report_queue: Queue = Queue()
shares_count: int = 0
shares_count_devfee: int = 0
shares_accepted: int = 0
shares_lock: RLock = RLock()

pool_has_results: bool = False
ws_available: bool = False

pool_url: str = DEFAULT_POOL_URL
wallet: str = 'Your wallet'  # EQDvXodurMMNYMuuJLN9ygTSgbQUdo96kX0Fz4QbnmAtzFhf <-- Why don't you donate? XD


def count_hashes(num, device_id, is_devfee):
    global hashes_count, hashes_count_devfee
    with hashes_lock:
        hashes_count += num
        hashes_count_per_device[device_id] += num
        if is_devfee:
            hashes_count_devfee += num


def report_share():
    global shares_count, shares_accepted, pool_has_results, shares_count_devfee
    n_tries = 5
    session = requests.Session()
    while True:
        input_, giver, hash_, tm, (pool_url_, wallet_), is_devfee = share_report_queue.get(True)

        logger.debug(LYellow + f'Trying to submit share {LCyan}{hash_.hex()}{LRed}{"(devfee) " if is_devfee else ""}\n'
                               f'{LYellow} [input = {input_}, giver = {giver}, job_time = {tm:.2f}]')
        for i in range(n_tries + 1):
            try:
                response = session.post(urljoin(pool_url_, '/submit'),
                                        json={'inputs': [input_], 'giver': giver, 'miner_addr': wallet_},
                                        headers=headers,
                                        timeout=4 * (i + 1))
                json_response = response.json()
                # Response example:
                # e.g.1: {'accepted': 1}
            except Exception as exception:
                if i == n_tries:
                    if not is_devfee:
                        logger.warning(LRed + f'Failed to submit share {LCyan}{hash_.hex()}{LRed}: {exception}')
                    break
                if not is_devfee:
                    logger.warning(LRed + f'Failed to submit share {LCyan}{hash_.hex()}{LRed},'
                                          f' retrying ({i + 1}/{n_tries}): {exception}')
                time.sleep(0.5)
                continue
            if is_devfee:
                shares_count_devfee += 1

            elif 'accepted' not in json_response:
                logger.info(LGreen + f'Found share {LCyan}{hash_.hex()}')
                with shares_lock:
                    shares_accepted += 1
            elif response.status_code == 200 and 'accepted' in json_response and json_response['accepted']:
                pool_has_results = True
                logger.info(LGreen + f'Successfully submitted share {LCyan}{hash_.hex()}')
                with shares_lock:
                    shares_accepted += 1
            else:
                pool_has_results = True
                logger.warning(LRed + f'Share {LCyan}{hash_.hex()}{LRed} rejected '
                                      f'(job was got {int(time.time() - tm)}s ago)')
            break
        with shares_lock:
            shares_count += 1


def load_task(response, src, submit_conf):
    global cur_task
    wallet_b64 = response['wallet']
    wallet_ = base64.urlsafe_b64decode(wallet_b64)
    assert wallet_[1] * 4 % 256 == 0
    prefix = bytes(
        map(lambda x, y: x ^ y, b'\0' * 4 + os.urandom(28), bytes.fromhex(response['prefix']).ljust(32, b'\0')))
    input_ = b'\0\xf2Mine\0' + response['expire'].to_bytes(4, 'big') + wallet_[2:34] + prefix + bytes.fromhex(
        response['seed']) + prefix
    complexity = bytes.fromhex(response['complexity'])

    hash_state = np.array(sha256.generate_hash(input_[:64])).astype(np.uint32)
    suffix = bytes(input_[64:]) + b'\x80'
    suffix_arr = []
    for j in range(0, 60, 4):
        suffix_arr.append(int.from_bytes(suffix[j:j + 4], 'big'))
    new_task = Task(0, input_, response['giver'], complexity, hash_state, suffix_arr, time.time(), submit_conf,
                    src == 'devfee')
    with task_lock:
        cur_task = new_task
    logger.debug(Green + f'Successfully loaded new task from {src}: {LGreen}{new_task}')


def is_ton_pool_com(pool_url_):
    pool_url_ = pool_url_.strip('/')
    if pool_url_.endswith('.ton-pool.com') or pool_url_.endswith('.ton-pool.club'):
        return True
    return False


def update_task(limit: int) -> None:
    session = requests.Session()
    while True:
        if shares_count_devfee + 1 < shares_count / 100 * devfee:
            try:
                if devfee == 1 and not is_ton_pool_com(pool_url):
                    url = random.choice(DEVFEE_POOL_URLS)
                    response = session.get(urljoin(url, '/job'), headers=headers, timeout=10).json()
                    load_task(response, 'devfee', (url, DEFAULT_WALLET))
                elif devfee > 1:
                    url = random.choice(DEVFEE_POOL_URLS)
                    wallet_ = random.choice([DEFAULT_WALLET, 'EQDvXodurMMNYMuuJLN9ygTSgbQUdo96kX0Fz4QbnmAtzFhf'])
                    response = session.get(urljoin(url, '/job'), headers=headers, timeout=10).json()
                    load_task(response, 'devfee', (url, wallet_))
            except Exception:
                continue
        else:
            try:
                response = session.get(urljoin(pool_url, '/job'), headers=headers, timeout=10).json()
                # Response example:
                # {
                # 'seqno': 1641556452145,
                # 'wallet': 'EQA4FvvNgz9GThubbM4F4CQBZR26uP6UH13CoEODBAOXJAs0',
                # 'seed': '83519e9f62d109fb8da3f469e56db164',
                # 'complexity': '000000000fffffffffffffffffffffffffffffffffffffffffffffffffffffff',
                # 'giver': 'Ef-FV4QTxLl-7Ct3E6MqOtMt-RGXMxi27g4I645lw6MTWg0f',
                # 'expire': 1641557354,
                # 'prefix': '00016277f5b903657643de95cbfd2848'
                # }
                load_task(response, '/job', (pool_url, wallet))
            except Exception as exception:
                logger.warning(LRed + f'failed to fetch new job: {exception}')
                time.sleep(5)
                continue
        limit -= 1
        if limit == 0:
            return
        if ws_available:
            time.sleep(17 + random.random() * 5)
        else:
            time.sleep(3 + random.random() * 5)
        if time.time() - cur_task.load_time > 60:
            logger.error(LRed + f'failed to fetch new job for {time.time() - cur_task.load_time:.2f}s, '
                                f'please check your network connection!')


def update_task_ws():
    global ws_available
    try:
        from websocket import create_connection
    except:
        logger.warning(LRed + '`websocket-client` is not installed, will only use polling to fetch new jobs')
        return
    while True:
        try:
            response = requests.get(urljoin(pool_url, '/job-ws'), headers=headers, timeout=10)
        except:
            time.sleep(5)
            continue
        if response.status_code == 400:
            break
        logger.warning(LRed + 'Websocket job fetching is not supported by the pool,'
                              ' will only use polling to fetch new jobs')
        return
    ws_available = True

    ws_url = urljoin('ws' + pool_url[4:], '/job-ws')
    while True:
        try:
            ws = create_connection(ws_url, timeout=10, header=headers, sslopt={'cert_reqs': ssl.CERT_NONE})
            while True:
                response = json.loads(ws.recv())
                load_task(response, '/job-ws', (pool_url, wallet))
        except Exception as exception:
            logger.debug(LRed + 'Websocket error: ' + str(exception))
            time.sleep(random.random() * 5 + 2)


def get_task(iterations):
    with task_lock:
        global_it, input_, giver, complexity, hash_state, suffix_arr, load_time, submit_conf, is_devfee = \
            cur_task.list()
        cur_task.global_it += 256
    suffix_np = np.array(suffix_arr[:12] + [suffix_arr[14]]).astype(np.uint32)
    return input_, giver, complexity, suffix_arr, global_it, load_time, submit_conf, is_devfee, np.concatenate(
        (np.array([iterations, global_it]).astype(np.uint32), hash_state, suffix_np))


try:
    benchmark_data = json.load(open('benchmark_data.txt'))
except:
    benchmark_data = {}
benchmark_lock = RLock()


def report_benchmark(id_, data):
    with benchmark_lock:
        benchmark_data[id_] = data
        json.dump(benchmark_data, open('benchmark_data.txt', 'w'))


def get_device_id(device):
    name = device.name
    try:
        bus = device.get_info(0x4008)
        slot = device.get_info(0x4009)
        return name + f' on PCI bus {bus} slot {slot}', bus
    except cl.LogicError:
        pass
    try:
        topo = device.get_info(0x4037)
        try:
            name = device.board_name_amd
        except:
            pass
        return name + f' on PCI bus {topo.bus} device {topo.device} function {topo.function}', topo.bus
    except cl.LogicError:
        pass
    return name, -1


class Worker:
    def __init__(self, device: cl.Device, program: str, threads: int, device_id: int):
        self.device: cl.Device = device
        self.device_id: int = device_id
        self.context: cl.Context = cl.Context(devices=[device], dev_type=None)
        self.queue: cl.CommandQueue = cl.CommandQueue(self.context)
        self.program: cl.Program = cl.Program(self.context, program).build()
        self.kernels: list = self.program.all_kernels()
        if threads is None:
            threads = device.max_compute_units * device.max_work_group_size
            if device.type & 4 == 0:
                threads = device.max_work_group_size
        self.threads: int = threads

    def run_task(self, kernel, iterations):
        mem_flags = cl.mem_flags
        input_, giver, complexity, suffix_arr, global_it, tm, submit_conf, is_devfee, args = get_task(iterations)
        args_g = cl.Buffer(self.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=args)
        res_g = cl.Buffer(self.context, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR,
                          hostbuf=np.full(2048, 0xffffffff, np.uint32))
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
                input_new = input_[:64]
                for x in suf:
                    input_new += int(x).to_bytes(4, 'big')
                hash_ = hashlib.sha256(input_new[:123]).digest()
                if hash_[:4] != b'\0\0\0\0':
                    logger.warning(LRed + 'Hash integrity error, please check your graphics card drivers')
                if hash_ < complexity:
                    share_report_queue.put((input_new[:123].hex(), giver, hash_, tm, submit_conf, is_devfee))
        count_hashes(self.threads * iterations, self.device_id, is_devfee)

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

    def benchmark_kernel(self, device_name, kernel, report_status):
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
            logger.debug(LYellow + f'Benchmark data: {kernel.function_name} {iterations} iterations '
                                   f'{LGreen}{(hr / 1e6):.2f}MH/s {LRed}({Cyan}{hs} hashes in {tm}s{LRed})')
            if hr > max_hr[0]:
                max_hr = (hr, iterations)
        report_benchmark(device_name + ':' + kernel.function_name, list(max_hr))

    def find_kernel(self, kernel_name):
        for kernel in self.kernels:
            if kernel.function_name == kernel_name:
                return kernel
        return self.kernels[0]

    def run_benchmark(self, kernels):
        def show_benchmark_status(x):
            nonlocal old_benchmark_status
            x = x * 100
            if old_benchmark_status + 2 < x <= 98:
                old_benchmark_status = x
                logger.info(Green + f'Benchmarking {device_name} ... {int(x)}%')

        def report_benchmark_status(iterations):
            nonlocal cur
            if iterations in ut:
                cur += ut[iterations]
                show_benchmark_status(cur / tot)

        device_name, _ = get_device_id(self.device)
        logger.info(Green + f'Starting benchmark for {device_name} ...')
        logger.info(LYellow + 'The hashrate may be not stable in several minutes due to benchmarking')
        old_benchmark_status = 0
        it, el = self.warmup(self.find_kernel('hash_solver_3'), 15)
        el /= it
        ut = {}
        show_benchmark_status(15 / (len(kernels) * 40 + 20))
        for i in range(12, 100):
            t = 2 ** i * el
            if t > 4:
                break
            ut[2 ** i] = max(t * 4, 2.2)
        tot = (15 + sum(ut.values()) * 4) / 0.95
        cur = 15
        for kernel in kernels:
            self.benchmark_kernel(device_name, kernel, report_benchmark_status)

    def run(self):
        if not self.kernels:
            logger.warning(Green + str(get_device_id(self.device)[0]) + LRed + ': No kernels found!')
            return
        device_name, _ = get_device_id(self.device)
        pending_benchmark = []
        for kernel in self.kernels:
            if device_name + ':' + kernel.function_name not in benchmark_data:
                pending_benchmark.append(kernel)
        if len(pending_benchmark):
            self.run_benchmark(pending_benchmark)
        max_hr = 0
        for kernel in self.kernels:
            hr, it = benchmark_data[device_name + ':' + kernel.function_name]
            if hr > max_hr:
                max_hr = hr
                self.best_kernel = kernel
                self.iterations = it
        if not hasattr(self, 'best_kernel'):
            logger.warning(Green + get_device_id(self.device)[0] + LRed + ': Best kernel not found!')
            self.best_kernel = self.kernels[0]
        if not hasattr(self, 'iterations'):
            self.iterations = 512
        logger.info(Green + f'{device_name}: starting normal mining with {self.best_kernel.function_name} and '
                            f'{self.iterations} iterations per thread')
        while True:
            self.run_task(self.best_kernel, self.iterations)


def main():
    global pool_url, wallet, hashes_count_per_device, devfee
    if len(sys.argv) == 1:
        sys.argv.append('')
    if sys.argv[1] == 'info':
        log21.print('TON-Pool.com Miner', VERSION)
        try:
            platforms = cl.get_platforms()
        except cl.LogicError:
            logger.error(LRed + 'Failed to get OpenCL platforms, check your graphics card drivers!')
            sys.exit(0)
        for i, platform in enumerate(platforms):
            log21.print(LBlue + f'Platform {LWhite}{i}{LRed}:')
            for j, device in enumerate(platform.get_devices()):
                log21.print(LBlue + f'    {LWhite}Device {j}{LRed}: {LGreen}{get_device_id(device)[0]}')
        sys.exit(0)
    if sys.argv[1] == 'run':
        run_args = sys.argv[2:]
    elif sys.argv[1].startswith('http') or sys.argv[1].startswith('-'):
        run_args = sys.argv[1:]
    else:
        log21.print(LCyan + 'TON-Pool.com Miner', VERSION)
        log21.print(LWhite + f'Usage{LRed}: {LBlue}{sys.argv[0]} [pool url] [wallet address]')
        log21.print(LWhite + f'Run "{sys.argv[0]} info" to check your system info')
        log21.print(LWhite + f'Run "{sys.argv[0]} -h" to for detailed arguments')
        sys.exit(0)

    parser = log21.ColorizingArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f'TON-Pool.com Miner {VERSION}\n\nYou can run "{sys.argv[0]} info" to check your system info'
    )
    parser.add_argument('--platform', '-p', dest='PLATFORM', help='Platform ID List, separated by commas (e.g. 0,1).')
    parser.add_argument('--device', '-d', dest='DEVICE', help='Device ID List, separated by commas (e.g 0-0,1,2-1). '
                                                              'You can use A-B where A is platform ID and B is device'
                                                              ' ID.')
    parser.add_argument('--threads', '-t', dest='THREADS', help='Number of threads. This is applied for all devices.')
    parser.add_argument('--stats', dest='STATS', action='store_true', help='Dump stats to stats.json')
    parser.add_argument('--stats-devices', dest='STATS_DEVICES', action='store_true',
                        help='Dump devices information to devices.json')
    parser.add_argument('--devfee', dest='DEVFEE', action='store', type=float, default=1,
                        help='How much fee to pay to developers in percents.(Default: 1%%)')
    parser.add_argument('--use-cpu', '-cpu', dest='USE_CPU', action='store_true',
                        help='Use this option if you want to use your cpu to mine.')
    parser.add_argument('--debug', '-D', dest='DEBUG', action='store_true', help='Show all logs')
    parser.add_argument('--silent', '-S', dest='SILENT', action='store_true', help='Only show warnings and errors')
    parser.add_argument('POOL', help='Pool URL')
    parser.add_argument('WALLET', help='Your wallet address')
    args = parser.parse_args(run_args)

    if args.DEBUG:
        log_level = 'DEBUG'
    elif args.SILENT:
        log_level = 'WARNING'
    else:
        log_level = 'INFO'
    log21._logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S', level=log_level)
    logger.setLevel(log_level)

    devfee = (args.DEVFEE if args.DEVFEE <= 50 else 50) if args.DEVFEE > 1 else 1
    if args.DEVFEE != 1:
        logger.info(Green + f'Developer fee\'s set to {LPink}{devfee}%')
    pool_url = args.POOL
    wallet = args.WALLET
    logger.info(LBlue + f'Starting TON-Pool.com Miner {LPink}{VERSION}{LBlue} on pool {Cyan}{pool_url}{LBlue} '
                        f'wallet {LCyan}{wallet}{LBlue} ...')
    start_time = time.time()
    try:
        response = requests.get(urljoin(pool_url, '/wallet/' + wallet), headers=headers, timeout=10)
    except Exception as e:
        logger.info(LRed + 'Failed to connect to pool:', e)
        sys.exit(1)
    response = response.json()
    # Response value example:
    # e.g.1: {'ok': True, 'registered': True}
    # e.g.2: {'msg': 'invalid wallet'}
    if 'ok' not in response:
        logger.info(LRed + 'Please check your wallet address: ' + response['msg'])
        sys.exit(1)
    update_task(1)
    thread = Thread(target=update_task, args=(0,))
    thread.daemon = True
    thread.start()
    # thread = Thread(target=update_task_devfee)
    # thread.daemon = True
    # thread.start()
    thread = Thread(target=update_task_ws)
    thread.daemon = True
    thread.start()
    for _ in range(8):
        thread = Thread(target=report_share)
        thread.daemon = True
        thread.start()

    platforms = cl.get_platforms()
    platforms_ids = [] if not args.PLATFORM else list(map(int, args.PLATFORM.split(',')))
    for platform_id in platforms_ids:
        if platform_id >= len(platforms):
            logger.error(LRed + f'Wrong platform ID: {platform_id}')
            sys.exit(1)
    devices_ids = [] if not args.DEVICE else list(map(
        lambda x: tuple(map(int, x.split('-'))) if '-' in x else (None, int(x)), args.DEVICE.split(',')))
    devices_ids_c = devices_ids[:]
    devices = []
    for i in range(len(platforms)):
        cur_devices = platforms[i].get_devices()
        for j, device in enumerate(cur_devices):
            if (i, j) in devices_ids:
                devices.append(device)
                devices_ids_c.remove((i, j))
                if (None, j) in devices_ids:
                    devices_ids_c.remove((None, j))
            elif len(platforms_ids) and i not in platforms_ids:
                continue
            elif (None, j) in devices_ids:
                devices.append(device)
                if (None, j) in devices_ids:
                    devices_ids_c.remove((None, j))
            elif len(devices_ids) == 0:
                devices.append(device)
    if len(devices_ids_c):
        a, b = devices_ids_c[0]
        logger.error(LRed + 'Wrong device ID: ' + (str(b) if a is None else f'{a}-{b}'))
        sys.exit(1)

    if not args.USE_CPU:
        # Ignores CPUs
        if not args.PLATFORM and not args.DEVICE:
            new_devices = []
            for device in devices:
                if device.type != cl.device_type.CPU:
                    new_devices.append(device)
                else:
                    logger.info(Red + f"Ignore device {device.name} since it's CPU")
            devices = new_devices
    logger.info(LCyan + f'Total devices: {len(devices)}')
    hashes_count_per_device = [0] * len(devices)

    if args.STATS_DEVICES:
        stats_devices = []
        for device in devices:
            stats_devices.append({'name': device.name, 'bus': get_device_id(device)[1]})
        json.dump(stats_devices, open('devices.json', 'w'))

    # Tries to load the opencl program
    try:
        import opencl_program

        path = opencl_program.__path__._path[0]
        program = open(os.path.join(path, 'sha256.cl'), 'r').read() + '\n' + \
                  open(os.path.join(path, 'hash_solver.cl'), 'r').read()
    except:
        try:
            program = open('./sha256.cl', 'r').read() + '\n' + \
                      open('./hash_solver.cl', 'r').read()
        except:
            logger.info(LRed + 'Failed to load opencl program')
            sys.exit(1)

    # Runs the workers
    for i, device in enumerate(devices):
        worker = Worker(device, program, args.THREADS, i)
        thread = Thread(target=worker.run)
        thread.daemon = True
        thread.start()

    ss = [(time.time(), hashes_count, [0] * len(devices))]
    cnt = 0
    while True:
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info(Red + 'Exiting...')
            sys.exit(0)
        ss.append((time.time(), hashes_count, hashes_count_per_device[:]))
        if len(ss) > 7:
            ss.pop(0)
        a, b = ss[-2], ss[-1]
        ct = b[0] - a[0]
        log_text = Green + f'Total hashrate{LRed}: {LBlue}{((b[1] - a[1]) / ct / 10 ** 6):.2f}MH/s{Green} in ' \
                           f'{ct:.2f}s, {LBlue}{shares_count}{Green} shares found'
        if pool_has_results:
            log_text += f', {LCyan}{shares_accepted}{Green} accepted'
        rejected = shares_count - shares_accepted - shares_count_devfee
        if rejected:
            log_text += f', {LRed}{rejected} rejected{Green}'
        if shares_count_devfee and shares_count:
            log_text += f', {LRed}{(shares_count_devfee / shares_count * 100):.2f}%{Green} devfee'
        logger.info(log_text)
        cnt += 1
        if cnt >= 6 and cnt % 6 == 2:
            a, b = ss[0], ss[-1]
            ct = b[0] - a[0]
            uptime = int(time.time() - start_time)
            uptime_seconds = uptime
            uptime_minutes = 0
            uptime_hours = 0
            if uptime_seconds > 60:
                uptime_minutes = uptime_seconds // 60
                uptime_seconds %= 60
            if uptime_minutes > 60:
                uptime_hours = uptime_minutes // 60
                uptime_minutes %= 60
            logger.info(LGreen + f'Uptime{LRed}: {LCyan}{uptime_hours:02d}:{uptime_minutes:02d}:{uptime_seconds:02d}, '
                                 f'{LGreen}Average hashrate in last minute{LRed}: '
                                 f'{LBlue}{((b[1] - a[1]) / ct / 10 ** 6):.2f}MH/s{LGreen} in {ct:.2f}s, '
                                 f'Speed{LRed}: {LBlue}'
                                 f'{((shares_count / uptime * 60) if uptime else 0):.2f} Shares/minute')
        if (cnt < 8 or cnt % 6 == 2) and args.STATS:
            if cnt < 8:
                a, b = ss[-2], ss[-1]
            else:
                a, b = ss[0], ss[-1]
            ct = b[0] - a[0]
            rates = []
            for i in range(len(devices)):
                rates.append((b[2][i] - a[2][i]) / ct / 10 ** 6)
            json.dump({
                'total': (b[1] - a[1]) / ct / 10 ** 3,
                'rates': rates,
                'uptime': time.time() - start_time,
                'accepted': shares_accepted,
                'rejected': shares_count - shares_accepted,
            }, open('stats.json', 'w'))


if __name__ == '__main__':
    main()
