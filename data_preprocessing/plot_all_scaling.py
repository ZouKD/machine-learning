from __future__ import print_function

import numpy as np

import matplotlib as mpl
from matplotlib import  pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

from sklearn.datasets import fetch_california_housing

# print(__doc__)

dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target
# print(X_full)
# print(y_full)

# Take only 2 features to make visualization easier
# Feature of 0 has a long tail distribution.
# Feature 5 has a few but very large outliers.

X = X_full[:, [0,5]]

distributions = [
    ('Unscaled data', X),
    ('Data after standard scaling', StandardScaler().fit_transform(X)),
    ('Data after min-max scaling', MinMaxScaler().fit_transform(X)),
    ('Data after max-abs scaling', MaxAbsScaler().fit_transform(X)),
    ('Data after robust scaling', RobustScaler(quantile_range=(25,75)).fit_transform(X)),
    ('Data after quantile transformation (uniform pdf)', QuantileTransformer(output_distribution='uniform')
     .fit_transform(X)),
    ('Data after quantile transformation (gaussian pdf)', QuantileTransformer(output_distribution='normal')
     .fit_transform(X)),
    ('Data after sample-wise L2 normalizing', Normalizer().fit_transform(X))
]

# scale the output between 0 and 1 for the colorbar
y = minmax_scale(y_full)
# print(y.size)

def create_axes(title, figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbor
    left, width, = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return ((ax_scatter, ax_histy, ax_histx),
            (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
            ax_colorbar)


