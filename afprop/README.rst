# afprop

afprop is a testing
package for clustering using affinity propagation, a method based on real-valued message exchange between data points until clusters emerge. Unlike KMeans, it does not require pre-specifying the number of clusters.

## Installation

```
pip install -i https://test.pypi.org/simple/ afprop
from afprop import afprop_vec
```

## Usage

```
clusters, exemplars, num_clusters, final_iter = afprop_vec(data)
```

Input
: 2-dimensional or multi-dimensional array and numerical dataframe

Output
: cluster labels, exemplars (cluster centers) and final iterations

### Reference

Frey, Brendan J., and Delbert Dueck. “Clustering by Passing Messages Between Data Points.” Science 315, no. 5814 (February 16, 2007): 972. https://doi.org/10.1126/science.1136800.
