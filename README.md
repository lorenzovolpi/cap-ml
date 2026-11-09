# cap-ml

version: `0.1.1.9`

## Changelog

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
