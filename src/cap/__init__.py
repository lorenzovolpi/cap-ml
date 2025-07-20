# fmt: off
import cap.data.datasets as dataset
import cap.error as error
import cap.plot as plot
import cap.utils.commons as commons
from cap.environment import env


def _get_njobs(n_jobs):
    return env["N_JOBS"] if n_jobs is None else n_jobs

# fmt: on
