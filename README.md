# cap-ml

version: `0.1.1.6`

## Changelog

### v0.1.1.7

- 

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
