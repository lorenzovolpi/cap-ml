from sys import platform
from traceback import print_exception as traceback

import quacc.evaluation.comp as comp
from quacc.dataset import Dataset
from quacc.environment import env
from quacc.logger import Logger
from quacc.utils import create_dataser_dir

CE = comp.CompEstimator()


def toast():
    if platform == "win32":
        import win11toast

        win11toast.notify("Comp", "Completed Execution")


def estimate_comparison():
    log = Logger.logger()
    for conf in env.get_confs():
        dataset = Dataset(
            env.DATASET_NAME,
            target=env.DATASET_TARGET,
            n_prevalences=env.DATASET_N_PREVS,
            prevs=env.DATASET_PREVS,
        )
        create_dataser_dir(dataset.name, update=env.DATASET_DIR_UPDATE)
        Logger.add_handler(env.OUT_DIR / f"{dataset.name}.log")
        try:
            dr = comp.evaluate_comparison(
                dataset,
                estimators=CE.name[env.COMP_ESTIMATORS],
            )
        except Exception as e:
            log.error(f"Evaluation over {dataset.name} failed. Exception: {e}")
            traceback(e)
        for plot_conf in env.get_plot_confs():
            for m in env.METRICS:
                output_path = env.OUT_DIR / f"{plot_conf}_{m}.md"
                try:
                    _repr = dr.to_md(
                        conf=plot_conf,
                        metric=m,
                        estimators=CE.name[env.PLOT_ESTIMATORS],
                        stdev=env.PLOT_STDEV,
                    )
                    with open(output_path, "w") as f:
                        f.write(_repr)
                except Exception as e:
                    log.error(
                        f"Failed while saving configuration {plot_conf} of {dataset.name}. Exception: {e}"
                    )
                    traceback(e)
        Logger.clear_handlers()

    # print(df.to_latex(float_format="{:.4f}".format))
    # print(utils.avg_group_report(df).to_latex(float_format="{:.4f}".format))


def main():
    log = Logger.logger()
    try:
        estimate_comparison()
    except Exception as e:
        log.error(f"estimate comparison failed. Exceprion: {e}")
        traceback(e)

    toast()
    Logger.close()


if __name__ == "__main__":
    main()
