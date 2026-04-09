import itertools as IT
import os

import env


def local_path(domain, dataset_name, cls_name, method_name, acc_name, experiment=None, format="parquet"):
    base_dir = env.root_dir if experiment is None else os.path.join(env.root_dir, experiment)
    parent_dir = os.path.join(base_dir, domain, acc_name, dataset_name, method_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{cls_name}.{format}")


def all_results_exist(domain, dataset_name, cls_name, method_names, acc_names, experiment=None):
    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        path = local_path(domain, dataset_name, cls_name, method, acc, experiment=experiment)
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist
