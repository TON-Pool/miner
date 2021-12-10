# TON-Pool Miner

Miner from [TON-Pool.com](https://ton-pool.com/)

## Instructions

Download the latest release of our miner at https://github.com/TON-Pool/miner/releases , then run the corresponding command according to your operating system.

```
# Windows
miner-windows.exe run https://next.ton-pool.club <your_wallet>

# Linux
./miner-linux run https://next.ton-pool.club <your_wallet>
```

If you have updated your hardware settings (like overclocking) or if you accidentally run our miner twice, make sure to delete `benchmark_data.txt` before running it again to get optimal hashrate.

### Hive OS Configuration

Installation URL: go to [releases](https://github.com/TON-Pool/miner/releases), copy the link of any `hiveos` file.

Wallet and worker template: `%WAL%`

Pool URL: https://next.ton-pool.club

## Run Python code

If you want to debug the miner, you can run the Python code directly.

You need to have Python 3 and packages `pyopencl`, `numpy`, and `requests` installed.

For Linux users, you can run `pip3 install pyopencl numpy requests` to install the packages. If you are running old version of Python, try `pip3 install "pyopencl<2018.3"` and `pip3 install "numpy<1.15"`.

For Windows users, you can run `pip3 install numpy requests` to install the later two packages. You need to go to https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl and download an `pyopencl` binary.

The command is

```
python3 miner.py [pool addr] [wallet address]
```

An optional dependency is `websocket-client`, if you install it you can get more timely job fetching.

## Dev Fee

You are allowed to use our miner in any mining pool, but if you don't use [TON-Pool.com](https://ton-pool.com/), then 1% of your income will be donated to the developers.

## License

GPLv3
