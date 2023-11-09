import os
import queue
import stat
import threading
from itertools import product as itproduct
from os.path import expanduser
from pathlib import Path

import paramiko
from tqdm import tqdm

known_hosts = Path(expanduser("~/.ssh/known_hosts"))
hostname = "ilona.isti.cnr.it"
username = "volpi"

__exec_main = "cd tesi; /home/volpi/.local/bin/poetry run main"
__exec_log = "/usr/bin/tail -f -n 0 tesi/quacc.log"
__log_file = "quacc.log"
__target_dir = Path("/home/volpi/tesi")
__to_sync_up = {
    "dir": [
        "quacc",
        "baselines",
    ],
    "file": [
        "conf.yaml",
        "run.py",
    ],
}
__to_sync_down = {
    "dir": [
        "output",
    ],
    "file": [],
}


def prune_remote(sftp: paramiko.SFTPClient, remote: Path):
    _ex_list = []
    mode = sftp.stat(str(remote)).st_mode
    if stat.S_ISDIR(mode):
        for f in sftp.listdir(str(remote)):
            _ex_list.append([prune_remote, sftp, remote / f])
        _ex_list.append([sftp.rmdir, str(remote)])
    elif stat.S_ISREG(mode):
        _ex_list.append([sftp.remove, str(remote)])

    return _ex_list


def put_dir(sftp: paramiko.SFTPClient, from_: Path, to_: Path):
    _ex_list = []

    _ex_list.append([sftp.mkdir, str(to_)])

    from_list = os.listdir(from_)
    for f in from_list:
        if (from_ / f).is_file():
            _ex_list.append([sftp.put, str(from_ / f), str(to_ / f)])
        elif (from_ / f).is_dir():
            _ex_list += put_dir(sftp, from_ / f, to_ / f)

    to_list = sftp.listdir(str(to_))
    for f in to_list:
        if f not in from_list:
            _ex_list += prune_remote(sftp, to_ / f)

    return _ex_list


def get_dir(sftp: paramiko.SFTPClient, from_: Path, to_: Path):
    _ex_list = []

    if not (to_.exists() and to_.is_dir()):
        _ex_list.append([os.mkdir, to_])

    for f in sftp.listdir(str(from_)):
        mode = sftp.stat(str(from_ / f)).st_mode
        if stat.S_ISDIR(mode):
            _ex_list += get_dir(sftp, from_ / f, to_ / f)
            _ex_list.append([sftp.rmdir, str(from_ / f)])
        elif stat.S_ISREG(mode):
            _ex_list.append([sftp.get, str(from_ / f), str(to_ / f)])
            _ex_list.append([sftp.remove, str(from_ / f)])

    return _ex_list


def sync_code(*, ssh: paramiko.SSHClient = None, verbose=False):
    _was_ssh = ssh is not None
    if ssh is None:
        ssh = paramiko.SSHClient()
        ssh.load_host_keys(known_hosts)
        ssh.connect(hostname=hostname, username=username)

    sftp = ssh.open_sftp()

    to_move = [item for k, vs in __to_sync_up.items() for item in itproduct([k], vs)]
    _ex_list = []
    for mode, f in to_move:
        from_ = Path(f).absolute()
        to_ = __target_dir / f
        if mode == "dir":
            _ex_list += put_dir(sftp, from_, to_)
        elif mode == "file":
            _ex_list.append([sftp.put, str(from_), str(to_)])

    for _ex in tqdm(_ex_list, desc="synching code: "):
        fn_ = _ex[0]
        try:
            fn_(*_ex[1:])
        except IOError:
            if verbose:
                print(f"Info: directory {to_} already exists.")

    sftp.close()
    if not _was_ssh:
        ssh.close()


def sync_output(*, ssh: paramiko.SSHClient = None):
    _was_ssh = ssh is not None
    if ssh is None:
        ssh = paramiko.SSHClient()
        ssh.load_host_keys(known_hosts)
        ssh.connect(hostname=hostname, username=username)

    sftp = ssh.open_sftp()

    to_move = [item for k, vs in __to_sync_down.items() for item in itproduct([k], vs)]
    _ex_list = []
    for mode, f in to_move:
        from_ = __target_dir / f
        to_ = Path(f).absolute()
        if mode == "dir":
            _ex_list += get_dir(sftp, from_, to_)
        elif mode == "file":
            _ex_list.append([sftp.get, str(from_), str(to_)])

    for _ex in tqdm(_ex_list, desc="synching output: "):
        fn_ = _ex[0]
        fn_(*_ex[1:])

    sftp.close()
    if not _was_ssh:
        ssh.close()


def _echo_channel(ch: paramiko.ChannelFile):
    while line := ch.readline():
        print(line, end="")


def _echo_log(ssh: paramiko.SSHClient, q_: queue.Queue):
    _, rout, _ = ssh.exec_command(__exec_log, timeout=5.0)
    while True:
        try:
            with open(__log_file, "a") as f:
                f.write(rout.readline())
        except TimeoutError:
            pass

        try:
            q_.get_nowait()
            return
        except queue.Empty:
            pass


def remote():
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(known_hosts)
    ssh.connect(hostname=hostname, username=username)
    sync_code(ssh=ssh)

    _, rout, rerr = ssh.exec_command(__exec_main)
    q = queue.Queue()

    _tlog = threading.Thread(target=_echo_log, args=[ssh, q])
    _tlog.start()

    _tchans = [threading.Thread(target=_echo_channel, args=[ch]) for ch in [rout, rerr]]

    for th in _tchans:
        th.start()

    for th in _tchans:
        th.join()

    q.put(None)

    sync_output(ssh=ssh)

    _tlog.join()
    ssh.close()
