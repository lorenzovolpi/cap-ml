import argparse

from quacc.main import main as _main
from remote import remote as _remote


def run_local():
    _main()


def run_remote():
    _remote()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--local", action="store_true", dest="local")
    parser.add_argument("-r", "--remote", action="store_true", dest="remote")
    args = parser.parse_args()

    if args.local:
        run_local()
    elif args.remote:
        run_remote()
