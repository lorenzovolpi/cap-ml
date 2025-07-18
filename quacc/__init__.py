# fmt: off
import quacc.data.datasets as dataset
import quacc.error as error
import quacc.plot as plot
import quacc.utils.commons as commons
from quacc.environment import env


def _get_njobs(n_jobs):
    return env["N_JOBS"] if n_jobs is None else n_jobs

# fmt: on
