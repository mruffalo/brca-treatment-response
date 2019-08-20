This code is structured as several standalone scripts, each of which consume
and/or produce some data or plots.

Requirements:

* Python 3.6 or newer
* R, for survival analysis

Python packages:

* NumPy, (current) version unimportant
* SciPy, 1.0.0 or newer
* scikit-learn, 0.19 or newer
* Pandas 0.23.2 or newer
* NetworkX 2.0 or newer
* matplotlib 2.2.2 or newer
* tables, 3.4.3 or newer
* mygene, 3.0.0 or newer
* data-path-utils, 0.8.1 or newer

Usage:

The main results in the manuscript and supplement are produced by the
`tcga_train.py` script. This script reads features and labels produced by
`compute_drug_features_labels.py`, and output of that feature/label computation
is included with this code in a subdirectory of data/. `tcga_train.py` performs
cross-validation, both leave-one-out and 5-fold.

# vim: set tw=79:
