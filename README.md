# NMR Clustering

NMR Clustering is a python tool to applying clustering algorithms for fluid characterization on NMR T1-T2 of shale.

## Author

The code is developed and maitained by Han Jiang (jianghan2013@gmail.com)
- [citation]: Jiang, H., Daigle, H., Tian, X., Pyrcz, M., Zhang, B., 2019. A Comparison of Clustering Algorithms applied to Fluid Characterization using NMR T1-T2 Maps of Shale. Computers & Geosciences. (accepted)
## Usage

First, download this repo.
- You need to have 'python' installed.
- You also need to install 'numpy', 'matplotplib', 'pandas', and 'sklearn'.
- You may also want to install jupyter notebook to run notebook file.

Second, import those libararies in your python environment
```sh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
Also import the Model_runner class 
```sh
from utilis.model_runner import Model_runner
```

Creat a instance
```sh
nmr_cluster = Model_runner()  
```

Load data
```sh
nmr_cluster.load_data()
```

Do preprocessing
```sh
nmr_cluster.preprocess()
```

Perform clustering
```sh
nmr_cluster.fit()
```

## More details and Test Case
For test case, please click 'demo.ipynb'

