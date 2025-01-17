# FlexWrfInversion
Scripts to perform inversions in an Observing System Simulation Experiments (OSSEs)

## Usage
### `run_osse.py`
The script can be run with:
```
python -m flexwrfinversion.run_osse /path/to/config.yaml
```

The config has the following structure:
```
prior:
  prior_loader: ''              # Name of the prior loader class
  kwargs: {}                    # Arguments to pass to the prior loader
prior_covariance:
  prior_covariance_loader: ''   # Name of the prior covariance loader class
  kwargs: {}                    # Arguments to pass to the prior covariance loader
target:
  target_loader: ''             # Name of the target loader class
  kwargs: {}                    # Arguments to pass to the target loader
measurement:
  measurement_loader: ''        # Name of the measurement loader class
  kwargs: {}                    # Arguments to pass to the measurement loader
measurement_covariance:
  measurement_covariance_loader: '' # Name of the measurement covariance loader class
  kwargs: {}                    # Arguments to pass to the measurement covariance loader
footprint:
  footprint_loader: ''          # Name of the footprint loader class
  kwargs: {}                    # Arguments to pass to the footprint loader

n_permutations: #               # Number of permutations to run
n_stations: #                   # Number of stations to use
output_dir: ''                  # Directory to save output
output_name: ''                 # Name of the output file
(permutation_seed: #)           # Seed for the permutation (optional)
(start_index: #)                # Start index for the permutation (optional)
```

In general the idea is to have one script that uses `Loader` classes to prepare the desired data in the required format for the inversion. To use a specific `Loader` check its implementation in the respective file (in `flexwrfinversion/loaders`) to find its required keyword-arguments. Each `Loader` can be referenced by its name in the config. Additionally set the `kwargs` for the initialization of the given `Loader` if needed. To see how to implement your own `Loaders`, see chapter "Contribution".

For example for total CO2 you can use the following:
TODO

## Contribution
### Loader
To add a `Loader` class follow the steps:
* Go the the respective file
* Start your own class and inherit from the base class
* First add the arguments, that are also given in the base class and call the `__init__` of the base class in your `__init__`
* Implement all abstract methods and properties
* Add the import of your class to the imports in `run_osse.py`
