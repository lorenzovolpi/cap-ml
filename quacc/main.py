import quacc.evaluation.comp as comp
from quacc.dataset import Dataset
from quacc.environ import env


def estimate_comparison():
    dataset = Dataset(
        env.DATASET_NAME, target=env.DATASET_TARGET, n_prevalences=env.DATASET_N_PREVS
    )
    output_path = env.OUT_DIR / f"{dataset.name}.md"
    with open(output_path, "w") as f:
        dr = comp.evaluate_comparison(dataset, estimators=env.COMP_ESTIMATORS)
        f.write(dr.to_md("acc"))

    # print(df.to_latex(float_format="{:.4f}".format))
    # print(utils.avg_group_report(df).to_latex(float_format="{:.4f}".format))


def main():
    estimate_comparison()


if __name__ == "__main__":
    main()
