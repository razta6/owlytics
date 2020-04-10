import numpy as np
import pandas as pd
import pywt

import matplotlib.pyplot as plt
import seaborn as sns

def get_samples_by_class(dataset, num_samples, class_colname, random_state=0):
    grouped_by_class = dataset.groupby(class_colname)
    samples = grouped_by_class.apply(lambda x: x.sample(num_samples, random_state=random_state))
    samples = samples.drop(["1"], axis=1)  # the op keeps the groupby col for some reason
    class_names = samples.index.get_level_values(0).unique()
    
    return samples, class_names

def visualize_dataset(dataset, num_samples, class_colname="1", random_state=0):
    samples, class_names = get_samples_by_class(dataset, num_samples, class_colname, random_state)
    print(f"Samples shape: {samples.shape}")

    fig, axes = plt.subplots(num_samples, len(class_names), figsize=(15, 10), sharex=True, sharey=True)

    for j, cls in enumerate(class_names):
        for i in range(num_samples):
            row = samples.loc[cls].iloc[i]
            sns.lineplot(x=range(len(row.index)), y=row.values, ax=axes[i][j])
            if i == 0:
                axes[i][j].set_title(f"Class: {cls}")

    plt.tight_layout()

    return samples


def plot_wavelet(time, signal, scales,
                 waveletname='cmor',
                 cmap=plt.cm.seismic,
                 title='Wavelet Transform (Power Spectrum) of signal',
                 ylabel='Period (years)',
                 xlabel='Time'):
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both', cmap=cmap)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()