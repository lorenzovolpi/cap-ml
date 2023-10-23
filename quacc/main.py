import os
import shutil
from pathlib import Path

import quacc.evaluation.comp as comp
from quacc.dataset import Dataset
from quacc.environ import env


def create_out_dir(dir_name):
    base_out_dir = Path(env.OUT_DIR_NAME)
    if not base_out_dir.exists():
        os.mkdir(base_out_dir)
    dir_path = base_out_dir / dir_name
    env.OUT_DIR = dir_path
    shutil.rmtree(dir_path, ignore_errors=True)
    os.mkdir(dir_path)
    plot_dir_path = dir_path / "plot"
    env.PLOT_OUT_DIR = plot_dir_path
    os.mkdir(plot_dir_path)


def estimate_comparison():
    for conf in env:
        create_out_dir(conf)
        dataset = Dataset(
            env.DATASET_NAME,
            target=env.DATASET_TARGET,
            n_prevalences=env.DATASET_N_PREVS,
        )
        output_path = env.OUT_DIR / f"{dataset.name}.md"
        try:
            dr = comp.evaluate_comparison(dataset, estimators=env.COMP_ESTIMATORS)
            for m in env.METRICS:
                output_path = env.OUT_DIR / f"{conf}_{m}.md"
                with open(output_path, "w") as f:
                    f.write(dr.to_md(m))
        except Exception as e:
            print(f"Configuration {conf} failed. {e}")

    # print(df.to_latex(float_format="{:.4f}".format))
    # print(utils.avg_group_report(df).to_latex(float_format="{:.4f}".format))


def main():
    estimate_comparison()


if __name__ == "__main__":
    main()
