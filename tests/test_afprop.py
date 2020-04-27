import numpy as np
import pandas as pd
import pytest
from afprop import aprop_vec


# sample data with clusters for testing
C1 = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=30)
C2 = np.random.multivariate_normal(mean=[4, 4], cov=np.eye(2), size=30)
mydata = np.r_[C1, C2]
df = pd.DataFrame(mydata)


def test_n_cluster():
    # test n_cluster equals #unique cluster labels
    clusters, exemplars, cluster_plot, num_clusters, final_iter = aprop_vec(
        mydata=mydata
    )
    assert len(set(clusters)) == num_clusters
    #assert num_clusters == 2


# def test_valid_input():
#     # test input need to have 2 columns
#     with pytest.raises(ValueError):
#         aprop_vec(mydata=np.random.rand(8, 3))
#     # assert str(e.value) == "S must have two columns."


def test_cluster_pref():
    # test valid specification of cluster preference
    pytest.raises(ValueError):
        aprop_vec(mydata=np.random.rand(8, 2), num_cluster_pref=0)
    # assert (
    #     str(e.value) == "Enter valid indication (1 or 2) of cluster number preference."
    # )


def test_valid_damping():
    # test valid specification of cluster preference
    pytest.raises(ValueError):
        aprop_vec(mydata=np.random.rand(8, 2), damp_c=2)
    # assert str(e.value) == "Enter a valid damping constant."


# test pd df is valid input
