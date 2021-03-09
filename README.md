# Code for Coping with Mistreatment in Fair Algorithms

This repo includes code for the UAI paper for reproducibility. 


### Running the code:

Run Vanilla FERM model:

```bash
python ferm.py --dataset [adult, compas, drug, arrhythm]
```

Run MT-SVM: :


```bash
python ferm_mt.py --dataset [adult, compas, drug, arrhythm] -r [rho_value]
```


### About Included Notebooks

The notebooks contain experiments on running our algorithm on balanced datasets. The datasets are balanced by undersampling the majority class.


### Software Requirements

1. `scikit-learn`
2. `numpy`
3. `pandas`
