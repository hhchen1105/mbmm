# mbmm

This repo contains the code for the multivariate beta mixture model (MBMM), a new probabilistic model for data clustering.  The flexible probability density function of the multivariate beta distribution suggests that the MBMM can adapt to diverse cluster shapes.  The MBMM can fit non-convex cluster shapes.

# Installation

Tested under Ubuntu and Macbook Air/Pro with Python 3.10.4.

```
pip install -r requirements.txt
```

# Usage

Use the following command to visualize the clustering result on the synthetic datasets.
```
python algo-cmp-synthetic-data.py
```

Use the following command to generate the clustering result on the breast cancer dataset.
```
python breast_cancer.py
```

Use the following command to generate the clustering result on the MNIST dataset.
```
python mnist.py
```
