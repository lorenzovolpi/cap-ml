# cap-ml

version: `0.1.1.14`

## Changelog

### v0.1.1.14

- `fetch_UCIMulticlassDataset` has a new parameter `min_class_support` (default `100`), passed to QuaPy's `fetch_UCIMulticlassDataset`: the classes with fewer instances than `min_class_support` in the full collection are dropped before the split, and the remaining classes are relabelled `0, …, K-1`. QuaPy also accepts a float, read as a share of the collection.
- With the default the data are unchanged (`100` was already QuaPy's default), so results obtained with earlier versions are unchanged.
- Why: on large collections QuaPy caps the training part at `max_train_instances=25000`, and `split_train` halves it into `T` and `V`, so a class kept by the threshold of `100` can have very few instances in `V` (on `poker_hand`, class 7 has 236 instances out of 1,024,985 and about 3 in `V`). A caller can now pass a larger threshold for such datasets.
- No other function changes.

### v0.1.1.13

- `RQBS` (SQS-H) has a new parameter `bootstrap` (default `True`), which selects how the prevalences of its `num_samples` validation bags are chosen:
  - `bootstrap=True`: the prevalence of the target is estimated on `num_samples` bootstrap resamples of the target, and one bag is drawn at each estimate (the variant implemented so far; SQS-H with a bootstrap of the prevalence);
  - `bootstrap=False`: the prevalence of the target is estimated once, on the whole target, and all the bags are drawn at that estimate (SQS-H as defined in the paper).
- With `bootstrap=True` the draws are those of v0.1.1.11, so results of `RQBS` obtained with v0.1.1.11 and v0.1.1.12 are unchanged. Both variants draw from NumPy's global state seeded by `random_state` at each call (as in v0.1.1.11).
- No other method changes.
- Added tests on both variants to `tests/test_rqbs.py` (the bags of `bootstrap=False` are all at the estimate on the whole target; the range of `bootstrap=True` equals the one of v0.1.1.11, rebuilt step by step).

### v0.1.1.12

- `PrediQuant` (SQS-S) has a new parameter `rho` (default `None`), the share of `V` that trains the quantifier (`Vq`); the rest (`Vb`) is used to draw the candidate bags. With `rho=None` the split is unchanged (`train_test_split(test_size=0.5, stratify=y)` seeded by `random_state`). With `rho` in `(0, 1)`, the validation indices of each class are permuted once, seeded by `random_state`, and `Vq` takes the first `round(rho * |V_j|)` of each class (half rounded up), so that, for the same seed, the splits at different values of `rho` are nested. A `rho` outside `(0, 1)`, or one that leaves a class empty in `Vq` or in `Vb`, raises `ValueError`.
- After `fit`, `PrediQuant` keeps the indices in `V` of the split (`Vq_idx`, `Vb_idx`), the positions in `Vb` of the instances of each candidate bag (`sigma_indices`) and the labels of `Vb` (`Vb_y`). After each `predict_ct_range` call (and so after `predict_ct`, `predict_range`, `predict`), it keeps the estimated prevalence of the target (`last_test_prev`) and the positions of the selected candidate bags (`last_selected_bags`). New method `class_support(selected_bags=None)`: for each class, the share of the instances of that class in `Vb` that appear in at least one selected bag (by default, the bags of the last prediction).
- Results of `PrediQuant` with `rho=None` are unchanged. No other method changes.
- Added tests on the split with `rho`, the stored quantities and `class_support` in `tests/test_prediquant.py`.

### v0.1.1.11

- `RQBS` (SQS-H, bootstrap variant) is now reproducible from `random_state` (default `qp.environ["_R_SEED"]`). Two sources of randomness ignored the seed:
  - the bootstrap of the target prevalence was computed by QuaPy's `AggregativeBootstrap.predict_conf`, which draws its resamples in joblib workers (`Parallel(n_jobs=-1)`) from their own global NumPy state, and received the constructor argument `random_state` (`None` by default) instead of the resolved seed;
  - the bags were drawn from `V` with `sampling_index` without a `random_state`, i.e. from NumPy's global state.
- `RQBS` no longer wraps its quantifier in `AggregativeBootstrap`: `fit` trains the quantifier on `V` (as `AggregativeBootstrap` did with one training sample), and `predict_ct_range` draws the `num_samples` bootstrap resamples of the target's classifier outputs, aggregates each of them, smooths the estimates and draws one bag per estimate, sequentially, from NumPy's global state seeded by `random_state` at each call and restored afterwards (new helper `_temp_np_seed`, which, unlike `quapy.util.temp_seed`, does not write `qp.environ["_R_SEED"]`). The range of a target therefore depends only on the target, on the fitted quantifier and on `random_state`, also inside joblib workers, and the global NumPy state is left untouched. `self.q` is now the quantifier itself.
- The method is unchanged in distribution (the same bootstrap of the target and the same bag sampling), but results of `RQBS` obtained with earlier versions cannot be reproduced draw by draw; they agree with the new ones up to the run-to-run variation. The randomness of the quantifier itself (e.g. an unseeded `MLPClassifier`) is up to the caller.
- No other method changes.
- Added `tests/test_rqbs.py`.

### v0.1.1.10

- `PrediQuant` (SQS-S) is now a contingency-table method: it derives from `CAPContingencyTable` and `CTCAPWithConfidence` instead of `CAPDirect` and `DirectCAPWithConfidence`. The selection of the bags is unchanged:
  - `predict_ct_range` returns the contingency tables (normalised, true classes on rows) of the `num_samples` candidate bags whose estimated prevalence is closest to the target's, i.e. the bags whose accuracies made up `predict_range` before;
  - `predict_ct` returns the mean of these tables;
  - `predict_range` returns the accuracy of each table of `predict_ct_range`, as before;
  - `predict` is inherited from `CAPContingencyTable`: the accuracy of the mean table. All the bags have the same size, so for the vanilla accuracy it equals the mean of `predict_range`, as before; for a non-linear measure (e.g. F1) it can differ from that mean;
  - `predict_with_confidence` builds the interval from the accuracies of the tables of the range, so its samples and its point estimate (their mean) are unchanged.
- The accuracy function is stored as `acc_fn` (the attribute of `CAPContingencyTable`); `acc` is kept as an alias, and `switch_and_fit` is kept.
- Results of `PrediQuant` obtained with v0.1.1.9 stay comparable: the accuracy ranges, the intervals and the mean of the range are unchanged; what is new is that a caller dispatching on `CAPContingencyTable` now gets the tables (`predict_ct_range`, `predict_ct`).
- Added tests to `tests/test_prediquant.py`.

### v0.1.1.9

- Fixed `PrediQuant` (SQS-S), so that it follows the method as defined: `V` is split 50/50 (stratified) into `V1` and `V2`; the quantifier is trained on `V1` and the candidate bags are drawn from `V2`. Two problems are fixed:
  - the quantifier was trained on all of `V`, so the quantifier-training subset and the bag-sampling subset overlapped; it is now trained on `V1` only;
  - the prevalences of the candidate bags and of the target sample were estimated by two different estimators: the bags by aggregating the posteriors of the classifier `h` under evaluation with the quantifier's aggregation (fitted on the posteriors of the quantifier's own classifier), the target with the quantifier's full pipeline (its own classifier, then its aggregation). The nearest-bag selection therefore compared estimates from different spaces. Both are now estimated by the quantifier's own pipeline: its classifier's posteriors on `V2` are computed once in `fit` and aggregated for every candidate bag, as the target is in `predict`; `h` is used only to build the contingency tables of the bags.
- Removed the `reuse_h` parameter of `PrediQuant`.
- Results of `PrediQuant` obtained with earlier versions are not comparable with the new ones.
- Added `tests/test_prediquant.py`.

### v0.1.1.8

- Added PrediQuant direct method
- Added RQBS direct method

### v0.1.1.7

- added smoothing function for errors based con contingency table 
- added K accuracy measure

### v0.1.1.6

- added balanced accuracy measure

### v0.1.1.5

- `cap.table` module removed

### v0.1.1.4

- refactored repository name from **QuAcc** to **cap-ml**
    - note that the package name is now `cap`, to import the package now you can do

    ```python
    import cap
    ```

- set default values for `cap.env` dictionary
    - the `.env` file will no longer be supported for the library
- refactored `OCE` and `PHD` CAP models to `S_LEAP` and `O_LEAP`, respectively
- `cap.plot.plotly` module removed
- refactored project folder structure, moved `cap` folder inside `src` folder
- removed _Hugging Face_ related methods in `cap.data`
- added direct import in the `cap.models` module for the CAP models
