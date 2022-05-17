# Index_Rebalancing

### Package Requirments & Versions
`pip install x` where x is the below listed packages
* `python == 3.7+` Code was tested on versions`3.8.10`& `3.9.7`
* `numpy == 1.21.5`
* `pandas == 1.3.5+`
* `hvplot == 0.8.0`
* `openpyxl == '3.0.9'`
* `scipy == 1.7.3`

### Purpose of Use
* Fetch an already existing index constitutes assets in the means of rebalacing its weights semi-annually
* The code is two parts:
    1. Index Construction Rules:Fetch assets based on Z_Value ranking and apply certain buffer for a specific date
    2. Constituent Weighting Scheme: Run an optimization problem given:
    
        a. minimize $\sum_{i=1}^{n_stocks} [(Capped_Wt - Unapped_Wt) / Unapped_Wt]$

        b. Stock Cap = Min (5%, 20*FCap_Wt)

        c. Stock Floor = 0.05%

### Files Navigation
* `Index_Rebalancing.ipynb`: Code explanation, building, some data visulaization and divide by parts
* `Index_Rebalancer.py`: Executable script alternative for the jupyter notebook
    - In the command line to run (after going to directory location): Type `python Index_Rebalancer.py` then follow terminal prompts to select rebalancing date.
* ` functions.py`: Some helper functions needed for the executable script
* `testing.py`: Unit testing file for proof of concept on testing
    - In the command line to run (after going to directory location): Type `python -m unittest -v testing.py`
* `Tasks for Python Test (003).xlsx`: Input and destination output file for algo to run and save results in to. 

*Note*: Output tab in the  `Tasks for Python Test (003).xlsx` is already populated with results and the code already is resilient against overwriting existing output based on date already saved, thus delete all contenst in the output tab to see the full proccess of saving the results.

### Limitation
The optimization constraint `Sector Cap = 50%` was not met in the minimization process although passed to the minimization
function. That is a limitation is must be investigated, one reason might be numerical precision in the optimizatio process
or the constrain equations (for each sector) was not passed programatically correct to the minimization process.

After researching also found that the issue might be from the intial_weights/gusses do not satisfy the future constraints and adjusted those by the `intial_weights()` function but that also didnt solve the issue and the sector constraint was not met after the optimization process.
One thing to note, when the sector consrtaint type `ineq` is switched to `eq` in the `{'type': 'eq', 'fun': lambda x: np.array(0.5 - (np.sum(x[[indexes]])))}` all optimizations constrainst pass the check!

