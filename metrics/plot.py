import copy
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import os

from matplotlib.testing.jpl_units import Epoch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import itertools
import argparse
import glob
import sys
from pathlib import Path
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import DATASETS_COLORS, DATASETS_MARKERS, DATASETS_ACRONYM, DATASETS, APPROACH_MAP, DS_MAP

def read_metrics_files(data_dir, filename="metrics.csv"):
    """ find all files named as `metrics.csv` in the data_dir and read them into a single dataframe

    Parameters
    ----------
    data_dir

    Returns
    -------

    """
    all_metrics = []
    for dataset, acr in zip(DATASETS, DATASETS_ACRONYM):
        metrics_file = os.path.join(data_dir, dataset, filename)
        if os.path.exists(metrics_file):
            metrics = pd.read_csv(metrics_file)
            metrics["Dataset"] = acr
            all_metrics.append(metrics)
    return pd.concat(all_metrics)


def plot_metrics_with_heatmaps(df, metric_keys, plot_name):
    scGen = df[df["Approach"] == "scGen"]
    df = df[df["Approach"] == "FedscGen"]
    df.drop(columns=["Approach"], inplace=True)

    dataset_keys = df['Dataset'].unique().tolist()

    fig, axs = plt.subplots(len(metric_keys), len(dataset_keys), figsize=(len(dataset_keys) * 4, df.Round.max() * 2),
                            squeeze=False)
    cmap = cm.get_cmap('viridis')
    norm = colors.Normalize(vmin=df[metric_keys].min().min(), vmax=df[metric_keys].max().max())
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    for j, dataset in enumerate(dataset_keys):
        for i, metric in enumerate(metric_keys):
            ax = axs[i][j]
            data = df[df['Dataset'] == dataset]
            # pivot = data.pivot(index='Round', columns='Epoch', values=metric)
            pivot = data.pivot(index='Epoch', columns='Round', values=metric)
            sns.heatmap(pivot, ax=ax, cmap='viridis', cbar=False, vmin=0, vmax=1)

            scgen_color_value = scGen[scGen['Dataset'] == dataset][metric].values[0]
            scgen_color = mappable.to_rgba(scgen_color_value)

            # Create a legend for each subplot
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='scGen',
                                          markerfacecolor=scgen_color, markersize=15, markeredgewidth=1,
                                          markeredgecolor='black')]
            ax.legend(handles=legend_elements, loc=(0.6, 1.02))
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
            ax.grid(False)
            if j == 0:  # Optionally, label rows with dataset names
                ax.text(-0.5, 0.38, metric, transform=ax.transAxes, fontsize=24, va='bottom', ha='left', rotation=90)
                ax.set_ylabel("Epochs", fontsize=18)
            else:
                ax.set_ylabel('')
                # set off the y-axis
                ax.yaxis.set_visible(False)
            if i == 0:  # Optionally, label columns with metric names
                ax.set_title(dataset, fontsize=18, loc='left')
            if i < len(metric_keys) - 1:
                ax.xaxis.set_visible(False)
            else:
                ax.set_xlabel("Rounds", fontsize=18)

    # Adjust layout for colorbar and legend
    plt.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.805, 0.28, 0.01, 0.5])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)

    # Adjust layout to make space for legends
    plt.subplots_adjust(right=0.8)
    plt.savefig(plot_name, dpi=300)




def plot_bo_hitmap(df, plt_name, dpi, tick_size=14, cell_size=1):
    # Determine the figure size based on the number of columns and rows
    if "Batch" in df.columns:
        df.drop(columns=["Batch"], inplace=True)
    num_cols, num_rows = df.shape
    fig_width = cell_size * num_rows
    fig_height = cell_size * num_cols

    # Plotting the heatmap
    fig = plt.figure(figsize=(fig_width + 3, fig_height + 3))
    fig.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.2, wspace=0.01, hspace=0.01)
    plt.grid(False)
    # abs_max = max(abs(df.values.min()), abs(df.values.max()))
    abs_max = 0.5
    ax = sns.heatmap(df, annot=True, cmap='RdBu', vmin=-abs_max, vmax=abs_max, center=0, annot_kws={"size": 26},
                     square=True, cbar=False)
    plt.xticks(fontsize=tick_size)
    ax.yaxis.set_visible(False)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=30, rotation=45, ha='right')
    norm = colors.Normalize(vmin=-abs_max, vmax=abs_max)
    mappable = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    cbar_ax = fig.add_axes([0.81, 0.27, 0.02, 0.6])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=26)
    plt.savefig(plt_name, dpi=dpi)




def read_kbet_score(scores_path):
    df = pd.read_csv(scores_path)
    df["stats"] = df['Unnamed: 0']
    df['acceptance_rate'] = 1 - df['kBET.observed']
    df = df.groupby(["approach", "k_value"])["acceptance_rate"].median().reset_index()
    return df


def acceptance_plot(df, plot_name):
    # k_values = df['k_values'].unique()

    for approach, color in zip(["scGen", "FedscGen"], ['blue', 'red']):
        subset = df[df["approach"] == approach]
        plt.plot(subset['k_value'], subset['acceptance_rate'], label=approach, marker='o', color=color)

    plt.xlabel('k-value')
    plt.ylabel('Acceptance Rate')
    plt.xticks(df.k_value.unique())
    plt.legend()
    plt.title('kBET Acceptance Rates across k-values')
    plt.savefig(plot_name)
    plt.close()




def read_tuning_res(data_dir):
    metric_keys = ["ARI", "NMI", "EBM", "ASW_B", "ASW_C", "KNN Acc"]
    drop_columns = ["Approach", "Seed", "File", "Inclusion", "BatchOut", "Batch", "N_Clients", "ILF1", "GC"]

    df = pd.read_csv(os.path.join(data_dir,"fedscgen", "param-tuning", "benchmark_metrics.csv"))
    plot_name = os.path.join(data_dir, "tuning-diff.png")
    scGen = pd.read_csv(os.path.join(data_dir, "scgen", "benchmark_metrics.csv"))
    scGen = scGen[(scGen.Seed == 42) & (scGen.Inclusion == "all") & (scGen.BatchOut.isna())]
    scGen.drop(columns=drop_columns + ["Round", "Epoch"], inplace=True)
    df.drop(columns=drop_columns, inplace=True)
    df = df[~df.Epoch.isna()]
    df.Epoch = df.Epoch.astype(int)
    df.Round = df.Round.astype(int)
    df.Dataset = df.Dataset.apply(lambda x: DS_MAP[x])
    scGen.Dataset = scGen.Dataset.apply(lambda x: DS_MAP[x])
    dataset_keys = df['Dataset'].unique().tolist()
    find_best_round_epoch(dataset_keys, copy.deepcopy(df), metric_keys, scGen)
    best_csv_path = os.path.join(data_dir, "best_performance_summary.csv")
    save_best_epoch_round_performance(df, scGen, dataset_keys, metric_keys, best_csv_path)

    plot_tuning_heatmap(dataset_keys, df, metric_keys, plot_name, scGen)

def save_best_epoch_round_performance(df, scGen, dataset_keys, metric_keys, output_path):
    records = []
    for dataset in dataset_keys:
        data = df[df['Dataset'] == dataset]
        scgen_row = scGen[scGen["Dataset"] == dataset]
        for metric in metric_keys:
            # Calculate performance difference
            data_diff = data.copy()
            data_diff[metric] = data[metric] - scgen_row[metric].values[0]

            # Identify best performance (maximum difference)
            best_row = data_diff.loc[data_diff[metric].idxmax()]
            records.append({
                "Dataset": dataset,
                "Metric": metric,
                "Best_Round": int(best_row["Round"]),
                "Best_Epoch": int(best_row["Epoch"]),
                "Performance_Diff": round(best_row[metric], 4)
            })

    best_df = pd.DataFrame(records)
    best_df.to_csv(output_path, index=False)
    print(f"Saved best (round, epoch) performance differences to {output_path}")



def find_best_round_epoch(dataset_keys, df, metric_keys, scGen):
    dfs = []
    for dataset in dataset_keys:
        data = df[df['Dataset'] == dataset]
        for metric in metric_keys:
            scgen_ds_metric = scGen[scGen["Dataset"] == dataset][metric].values[0]
            data[metric] = data[metric].apply(lambda x: x - scgen_ds_metric)
        dfs.append(data)
    df = pd.concat(dfs)
    res_min = {}
    for r in df['Round'].unique():
        for e in df['Epoch'].unique():
            data = df[(df['Round'] == r) & (df['Epoch'] == e)]
            res_min[f"E{e}R{r}"] = data[metric_keys].min().min()
    res_max = sorted(res_min.items(), key=lambda x: x[1], reverse=True)
    print(res_max[:10])
    count_max = {}
    for k, v in res_max:
        e = int(k[1:].split("R")[0])
        r = int(k.split("R")[1])
        temp = df[(df['Round'] == r) & (df['Epoch'] == e)]
        count_max[k] = (temp[metric_keys] >= 0).sum().sum()
    count_max = sorted(count_max.items(), key=lambda x: x[1], reverse=True)
    print(count_max[:10])
    print(len(count_max))
    er = df[(df["Epoch"] == 2) & (df["Round"] == 8)]
    print((er[metric_keys] >= 0).sum())
    print((er[metric_keys] < 0).sum())

def plot_tuning_heatmap(dataset_keys, df, metric_keys, plot_name, scGen):
    fig, axs = plt.subplots(len(metric_keys), len(dataset_keys),
                            figsize=(len(dataset_keys) * 4, len(df['Epoch'].unique()) * 2),
                            squeeze=True)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)
    # Ensure symmetric vmin and vmax for color normalization around zero
    max_abs_value = max(abs(df[metric_keys].min().min()), abs(df[metric_keys].max().max()))
    max_abs_value = 1
    norm = colors.Normalize(vmin=-max_abs_value, vmax=max_abs_value)
    cmap = cm.get_cmap('RdBu')
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    for j, dataset in enumerate(dataset_keys):
        for i, metric in enumerate(metric_keys):
            ax = axs[i][j]
            data = df[df['Dataset'] == dataset]
            scgen_ds_metric = scGen[scGen["Dataset"] == dataset][metric].values[0]
            # Find the index of the maximum value
            max_index = data[metric].idxmax()
            best_epoch = data.loc[max_index, "Epoch"]
            best_round = data.loc[max_index, "Round"]
            data[metric] = data[metric].apply(lambda x: x - scgen_ds_metric)
            max_value_diff = data.loc[max_index, metric]

            pivot = data.pivot(index='Epoch', columns='Round', values=metric)
            pivot = pivot.sort_index()
            pivot = pivot.sort_index(axis=1)
            abs_max = 1
            cmap = 'RdBu'
            sns.heatmap(pivot, ax=ax, cmap=cmap, cbar=False, center=0, vmin=-abs_max, vmax=abs_max, square=True)
            ax.text(best_round - 0.5, best_epoch - 0.5, f"{max_value_diff:.2f}", ha='center', va='center', fontsize=12)

            ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
            ax.grid(False)
            if j == 0:
                ax.text(-0.5, 0.38, metric, transform=ax.transAxes, fontsize=24, va='bottom', ha='left',
                        rotation=90)
                ax.set_ylabel("Epochs", fontsize=18)
            else:
                ax.set_ylabel('')
                ax.yaxis.set_visible(False)
            if i == 0:
                ax.set_title(dataset, fontsize=18, loc='left')
            if i < len(metric_keys) - 1:
                ax.xaxis.set_visible(False)
            else:
                ax.set_xlabel("Rounds", fontsize=18)
    # Adjust layout for colorbar and legend
    plt.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.805, 0.28, 0.01, 0.5])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    plt.savefig(plot_name, dpi=300)


def acceptance_diff_plot(df, plot_name):
    unique_datasets = df["Dataset"].unique()

    # Group by dataset, approach, and k_value to compute the median acceptance rate
    grouped_df = df.groupby(["Dataset", "approach", "k_value"])["acceptance_rate"].median().reset_index()

    # Pivot the dataframe to get acceptance rates for each approach in separate columns
    pivoted_df = grouped_df.pivot_table(index=["Dataset", "k_value"], columns="approach",
                                        values="acceptance_rate").reset_index()

    # Calculate the difference in acceptance rates
    pivoted_df["difference"] = pivoted_df["FedscGen"] - pivoted_df["scGen"]
    df = pivoted_df

    # Create a Seaborn line plot
    plt.figure(figsize=(6, 4))
    for i, dataset in enumerate(unique_datasets):
        subset = df[df["Dataset"] == dataset]
        c_num = i + 1 if i < 3 else i + 1
        sns.scatterplot(data=subset, x='k_value', y='difference', label=dataset,
                        color=DATASETS_COLORS[dataset],
                        marker=DATASETS_MARKERS[dataset], s=100)

    # Adjust legend position to be outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20, prop={"size": 12})
    plt.axhline(y=0, color='gray', linestyle='-')
    plt.xlabel('k-value', fontsize=16)
    plt.ylabel('Acceptance Rate Difference', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks([0.05, 0.1, 0.15, 0.2, 0.25], fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_name, dpi=1000)
    plt.close()


def read_kbet_file(hp_kbet_file):
    kbet_df = pd.read_csv(hp_kbet_file)
    kbet_df.drop(columns=["Unnamed: 0", "filename"], inplace=True)
    kbet_df["approach"] = kbet_df["approach"].apply(lambda x: "scGen" if x == "scgen" else "FedscGen")
    kbet_df['acceptance_rate'] = 1 - kbet_df['kBET.observed']
    grouped_df = kbet_df.groupby(["approach", "k_value"])["acceptance_rate"].median().reset_index()
    pivoted_df = grouped_df.pivot_table(index=["k_value"], columns="approach",
                                        values="acceptance_rate").reset_index()
    pivoted_df["difference"] = pivoted_df["FedscGen"] - pivoted_df["scGen"]
    kbet_diff = pivoted_df["difference"].mean().round(2)
    return kbet_diff


def read_plot_batchout(data_dir):
    def read_metrics(data_dir, approach):
        drop_columns = ["Approach", "Dataset", "Seed", "File", "Inclusion", "N_Clients", "Epoch", "Round", "BatchOut", "index"]
        df = pd.read_csv(os.path.join(data_dir, approach, "benchmark_metrics.csv"))
        df = df[(df.Seed == 42) & (df.Inclusion == "all") & (df.Dataset == "HumanPancreas")].reset_index()
        if approach == 'fedscgen':
            df = df[~((df.File.str.endswith("fed_corrected.h5ad")) & (~df.Batch.isna()))]
        df.drop(columns=drop_columns, inplace=True)
        df["Batch"] = df["Batch"].apply(lambda x: int(x + 1) if pd.notna(x) else 0)
        df.set_index(["Batch"], inplace=True)
        return df
    scgen = read_metrics(data_dir, "scgen")
    fedscgen = read_metrics(data_dir, "fedscgen")
    diff_df = get_hp_metrics_kbet_diff(scgen, fedscgen, data_dir)
    bo_kbet_dir = f"{data_dir}/fedscgen/HumanPancreas/all/BO1-C4"
    diff_df = get_bo_kbet_diff(bo_kbet_dir, diff_df)
    diff_df = diff_df.astype('float')
    diff_df = diff_df.loc[diff_df.index[diff_df.index != 0].tolist() + [0]]
    diff_df.to_csv(f"{data_dir}/bo-metrics.csv", index=True)
    plot_bo_hitmap(diff_df, f"{data_dir}/bo-hitmap.png", dpi=300)


def get_bo_kbet_diff(bo_kbet_dir, diff_df):
    for bo in range(5):
        file_path = os.path.join(bo_kbet_dir, str(bo), 'kBET_summary_results.csv')
        diff_df.loc[diff_df.index == bo + 1, "kBET"] = read_kbet_file(file_path)
    return diff_df

def get_hp_metrics_kbet_diff(scgen, fedscgen, data_dir):
    diff = fedscgen.subtract(scgen).round(2)
    diff["kBET"] = None
    kbet_value_diff = read_kbet_file(f"{data_dir}/fedscgen/HumanPancreas/all/BO0-C5/kBET_summary_results.csv")
    diff.loc[diff.index == 0, "kBET"] = kbet_value_diff
    return diff

def read_kbet(data_dir):
    for inclusion in ["all", "combined", "dropped"]:
        print("Inclusion:", inclusion)
        all_kbets = []
        for dataset, acr in zip(DATASETS, DATASETS_ACRONYM):
            if dataset == "MouseBrain" or (inclusion != "all" and dataset in ["CellLine", "HumanDendriticCells"]):
                continue
            n_clients = 5 if dataset == "HumanPancreas" else 3 if dataset == "CellLine" else 2
            kbet_file = os.path.join(data_dir, dataset, inclusion, f"BO0-C{n_clients}", "kBET_summary_results.csv")
            if os.path.exists(kbet_file):
                kbet_df = pd.read_csv(kbet_file)
                kbet_df.drop(columns=["Unnamed: 0", "filename"], inplace=True)
                kbet_df["Dataset"] = acr
                kbet_df["approach"] = kbet_df["approach"].apply(lambda x: "scGen" if x == "scgen" else "FedscGen")
                all_kbets.append(kbet_df)
            else:
                raise FileNotFoundError(f"{kbet_file} is not exist")
        df = pd.concat(all_kbets)
        df.sort_values(by="Dataset", inplace=True)
        df['acceptance_rate'] = 1 - df['kBET.observed']

        acceptance_diff_plot(df, os.path.join(data_dir, f'kBET_Acceptance_Rates-{inclusion}.png'))


def read_scenarios_metrics(data_dir):
    all_scenarios = []
    for inclusion in ["all", "combined", "dropped"]:
        df = get_scenario_metrics_diff(data_dir, inclusion)
        all_scenarios.append(df)
    df = pd.concat(all_scenarios, ignore_index=True)
    df.kBET = df.kBET.astype(float)
    df.to_csv(os.path.join(data_dir, "datasets-metrics-scenarios.csv"))
    plot_scenarios_heatmap(df, data_dir)


def plot_scenarios_heatmap(df, data_dir):
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    df["inclusion"] = df["inclusion"].apply(lambda x: x.capitalize())

    # Subplot: 3 rows, 2 columns
    fig, axs = plt.subplots(3, 2, figsize=(30, 18), squeeze=True)
    fig.subplots_adjust(left=0.09, right=0.92, top=0.99, bottom=0.08, wspace=0.01, hspace=0.01)
    abs_max = 1  # fixed color scale range
    norm = colors.Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = cm.get_cmap('RdBu')
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    datasets = ["HP", "MB", "MR", "MCA", "PBMC", "MHSPC"]

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, 2)
        ax = axs[row][col]

        subset = df[df["Dataset"] == dataset].set_index("inclusion")
        subset.drop(columns=["Dataset"], inplace=True)
        sns.heatmap(
            subset,
            annot=True,
            cmap='RdBu',
            center=0,
            ax=ax,
            cbar=False,
            annot_kws={"size": 26},
            square=True,
            vmin=-abs_max,
            vmax=abs_max
        )

        ax.set_title(dataset, fontsize=34)
        ax.grid(False)

        # Y-axis label and ticks
        if col == 0:
            ax.set_ylabel("Inclusion", fontsize=32)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=28, rotation=0, ha='right')
        else:
            ax.set_ylabel('')
            ax.yaxis.set_visible(False)

        # X-axis label and ticks
        if row == 2:
            ax.set_xlabel("Metrics", fontsize=32)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=28, rotation=45, ha='right')
        else:
            ax.set_xlabel('')
            ax.xaxis.set_visible(False)

    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=22)

    # Save figure
    plt.savefig(os.path.join(data_dir, "datasets-metrics-scenarios.png"), dpi=300)
    plt.close()

def read_all_inc_plot_heatmap(data_dir):
    df = get_scenario_metrics_diff(data_dir, inclusion="all", skip_datasets=[])
    df.to_csv(os.path.join(data_dir, "datasets-metrics-all.csv"), index=False)
    df = pd.read_csv(os.path.join(data_dir, "datasets-metrics-all.csv"))
    df.set_index("Dataset", inplace=True)
    df.drop(columns=["inclusion"], inplace=True)
    fig, axs = plt.subplots(1, 1, figsize=(16, 14), squeeze=True)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)
    abs_max = 1
    sns.heatmap(df, annot=True, cmap='RdBu', center=0, ax=axs, cbar=False, annot_kws={"size": 28}, square=True,
                vmin=-abs_max, vmax=abs_max)
    axs.set_title("Performance difference: FedscGen - scGen", fontsize=36)
    axs.grid(False)
    axs.set_ylabel("Datasets", fontsize=30)
    axs.set_yticklabels(axs.get_yticklabels(), fontsize=30, rotation=0, ha='right')
    axs.set_xlabel("Metrics", fontsize=30)
    axs.set_xticklabels(axs.get_xticklabels(), fontsize=30, rotation=45, ha='right')
    norm = colors.Normalize(vmin=-abs_max, vmax=abs_max)
    mappable = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    cbar_ax = fig.add_axes([0.91, 0.25, 0.02, 0.5])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=24)
    plt.savefig(os.path.join(data_dir, "datasets-metrics-all.png"), dpi=300)


def get_scenario_metrics_diff(data_dir, inclusion, skip_datasets=["CL", "HDC"]):
    def read_approach_matrics(approach):
        df = pd.read_csv(os.path.join(data_dir, approach, "benchmark_metrics.csv"))
        df = df[(df.Seed == 42) & (df.Inclusion == inclusion) & (df.Batch.isna())]
        df.drop(columns=["Seed", "File", "Inclusion", "N_Clients", "Epoch", "Round", "Approach", "Batch", "BatchOut"], inplace=True)
        df.Dataset = df.Dataset.apply(lambda x: DS_MAP[x])
        df = df[~df.Dataset.isin(skip_datasets)]
        df.set_index("Dataset", inplace=True)
        return df

    fedscgen = read_approach_matrics("fedscgen")
    scgen = read_approach_matrics("scgen")
    metrics_df = fedscgen.subtract(scgen).round(2)
    metrics_df["kBET"] = None
    for dataset, acr in zip(DATASETS, DATASETS_ACRONYM):
        if acr in metrics_df.index:
            n_clients = 5 if dataset == "HumanPancreas" else 3 if dataset == "CellLine" else 2
            kbet_file = os.path.join(data_dir,"fedscgen", dataset, inclusion, f"BO0-C{n_clients}", "kBET_summary_results.csv")
            metrics_df.loc[metrics_df.index == acr, "kBET"] = read_kbet_file(kbet_file)

    metrics_df["inclusion"] = inclusion
    metrics_df.reset_index(inplace=True)
    return metrics_df


def get_classification_accuracy(data_dir):
    dfs = []
    for filename in glob.glob(data_dir + "/classification_acc_*.csv"):
        df = pd.read_csv(filename, index_col=None, header=0)
        mean_df = df.groupby('Epoch').mean().reset_index()
        dfs.append(mean_df)
    if len(dfs) == 0:
        return None
    result = pd.concat(dfs, axis=0, ignore_index=True)
    dir_mean_acc_by_epoch = result.groupby('Epoch')['Accuracy'].mean()
    dir_mean_auc_by_epoch = result.groupby('Epoch')['AUC'].mean()

    # Get highest accuracy and its corresponding epoch
    max_acc_epoch = dir_mean_acc_by_epoch.idxmax()
    max_acc = dir_mean_acc_by_epoch.max()

    # Get highest AUC and its corresponding epoch
    max_auc_epoch = dir_mean_auc_by_epoch.idxmax()
    max_auc = dir_mean_auc_by_epoch.max()
    return max_acc_epoch, max_acc, max_auc_epoch, max_auc


def read_classification(data_dir):
    columns = ['Dataset', 'Inclusion', 'Model', "Batch Out", 'Approach', 'Max Accuracy', 'Max Accuracy Epoch',
               'Max AUC', 'Max AUC Epoch']
    results_df = pd.DataFrame(columns=columns)

    for dataset, acr in zip(DATASETS, DATASETS_ACRONYM):
        for model in ["mlp-norm", "knn"]:
            for approach in ["scGen", "FedscGen"]:
                if approach == "scGen":
                    r_dir = os.path.join(data_dir, "centralized", dataset, "all", "classification", model)
                else:
                    n_clients = 5 if dataset == "HumanPancreas" else 3 if dataset == "CellLine" else 2
                    r_dir = os.path.join(data_dir, "federated", dataset, "all", f"BO0-C{n_clients}", "classification",
                                         model)

                metrics = get_classification_accuracy(r_dir)
                if metrics:
                    max_acc_epoch, max_acc, max_auc_epoch, max_auc = metrics
                else:
                    print(f"Metrics file not found in {r_dir}")
                    continue

                row = pd.DataFrame([{
                    'Dataset': acr,
                    'Inclusion': "all",
                    "Model": "MLP" if model == "mlp-norm" else model.upper(),
                    'Approach': approach,
                    'Max Accuracy': max_acc,
                    'Max Accuracy Epoch': max_acc_epoch,
                    'Max AUC': max_auc,
                    'Max AUC Epoch': max_auc_epoch
                }]
                )
                results_df = pd.concat([results_df, row], ignore_index=True)
    print(results_df.head(len(results_df)))
    results_df.to_csv(os.path.join(data_dir, "latent_acc_diff.csv"))

    # Calculate the difference in accuracy between FedscGen and scGen
    diff_df = results_df.pivot_table(index=['Dataset', 'Model'], columns='Approach',
                                     values='Max Accuracy').reset_index()

    diff_df['Accuracy Difference'] = diff_df['FedscGen'] - diff_df['scGen']
    print(diff_df)
    # Plotting
    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(6, 6))

    markers = {'MLP': 'o', 'KNN': '*'}
    # use palette='viridis' to color the markers
    palette = sns.color_palette('viridis', len(diff_df['Dataset'].unique()))

    for i, model in enumerate(diff_df["Model"].unique()):
        model_df = diff_df[diff_df["Model"] == model]
        ax.scatter(model_df["Dataset"], model_df["Accuracy Difference"], marker=markers[model], label=model,
                   facecolors='none', edgecolors=palette[i])

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy Difference')
    ax.axhline(0, color='gray', linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    # Format y-axis numbers to two decimal places
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.set_ylim(-0.15, 0.15)
    # set ytick labels with step of 0.05
    ax.set_yticks(np.arange(-0.15, 0.16, 0.05))

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "latent_accuracy_comparison.png"), dpi=300)


def set_fontsize(ax, y_label, font_size, tick_fontsize):
    ax.set_xlabel('', fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize)


def collect_lisi_results(data_dir):
    for inclusion in ["all", "dropped", "combined"]:
        aggregated_data = []
        for dataset in DATASETS:
            if inclusion != "all" and dataset in ["CellLine", "HumanDendriticCells"]:
                continue
            n_clients = 5 if dataset == "HumanPancreas" else 3 if dataset == "CellLine" else 2
            csv_path = os.path.join(data_dir, dataset, inclusion, f"BO0-C{n_clients}", 'lisi_results.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df.rename(columns={"file_name": "Approach"}, inplace=True)
                df["Approach"] = df["Approach"].apply(lambda x: APPROACH_MAP[x])
                df["Dataset"] = dataset
                aggregated_data.append(df)
            else:
                raise FileNotFoundError(f"{csv_path} is not exist")
        # Concatenate all dataframes to get a single dataframe for the current target.
        aggregated_df = pd.concat(aggregated_data, ignore_index=True)
        categories = ["Raw", "scGen", "FedscGen"]
        if inclusion == "all":
            categories.append("FedscGen-SMPC")
        aggregated_df['Approach'] = pd.Categorical(aggregated_df['Approach'], categories=categories, ordered=True)
        aggregated_df.Dataset = aggregated_df.Dataset.apply(lambda x: DS_MAP[x])
        aggregated_df.to_csv(f"{data_dir}/lisi_{inclusion}.csv", index=False)


def plot_lisi(data_dir, inclusions=['all', "dropped", "combined"]):
    color_palette = sns.color_palette("viridis", 4)
    for inclusion in inclusions:
        aggregated_df = pd.read_csv(f"{data_dir}/lisi_{inclusion}.csv")
        hatch_patterns = ['/', '\\', '|']
        plt.figure(figsize=(18, 6))
        approaches = aggregated_df["Approach"].unique()
        file_name_to_color = {file_name: color for file_name, color in zip(approaches, color_palette)}
        plt.subplot(1, 2, 1)
        ax1 = sns.boxplot(x='Dataset', y='batch', hue='Approach', data=aggregated_df,
                          palette=file_name_to_color)
        ax1.legend(fontsize=20)

        # Boxplot for 'cell_type'
        plt.subplot(1, 2, 2)
        ax2 = sns.boxplot(x='Dataset', y='cell_type', hue='Approach', data=aggregated_df,
                          palette=file_name_to_color)
        ax2.legend_.remove()
        set_fontsize(ax1, 'iLISI', font_size=22, tick_fontsize=20)
        set_fontsize(ax2, 'cLISI', font_size=22, tick_fontsize=20)
        # Add hatch patterns for models (file_name)
        for i, patches in enumerate(zip(ax1.artists, ax2.artists)):
            hatch = hatch_patterns[i % len(hatch_patterns)]
            patches[0].set_hatch(hatch)
            patches[1].set_hatch(hatch)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{data_dir}/lisi_{inclusion}.png', dpi=1000)
        plt.close()

def plot_accuracy_diff(df, plot_dir):
    df['Dataset'] = df['Dataset'].map(dict(zip(DATASETS, DATASETS_ACRONYM)))
    df_long = df.melt(
        id_vars=['Dataset', 'Fold'],
        value_vars=['FedscGen - scGen', 'FedscGen-SMPC - scGen'],
        var_name='Combination',
        value_name='Accuracy Difference'
    )
    plt.figure(figsize=(7, 6))
    ax = sns.boxplot(
        x='Dataset',
        y='Accuracy Difference',
        hue='Combination',
        data=df_long,
        palette='Set2',
        width=0.6,  # Make boxes narrower
        linewidth=0.8, # Reduce error bar width
        dodge=True
    )
    set_fontsize(ax, 'Accuracy Difference', 22, 18)
    # Formatting
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line at 0
    plt.title('')
    plt.legend(
        title='',
        loc='upper left',
        fontsize=18,
        frameon=False,
        bbox_to_anchor=(-.2, 1.1),
        ncol=2,
        columnspacing=0.4,
        borderaxespad=0.0,
        handletextpad=0.2
    )

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/classification_accuracy_difference.png', dpi=300)


def get_classification_stats(data_dir):
    """Reads classification accuracy files, aligns runs, and computes statistics.

    Returns a DataFrame with columns: Dataset, Model, Run, Approach, Accuracy, AUC
    """
    results = []
    n_clients_map = {"HumanPancreas": 5, "CellLine": 3}

    for approach in ["scGen", "FedscGen", "FedscGen-SMPC"]:
        for dataset in DATASETS:
            if dataset == "MouseBrain":
                continue
            files_dir = os.path.join(data_dir, approach.lower(), dataset)

            if approach == "scGen":
                files_path = f"{files_dir}/all/classification/classification_acc_*.csv"
                files = glob.glob(files_path)
            elif approach == "FedscGen":
                n_clients = n_clients_map.get(dataset, 2)
                files_path = f"{files_dir}/all/BO0-C{n_clients}/classification/classification_acc_*.csv"
                files = glob.glob(files_path)
            else:  # FedscGen-SMPC
                files_path = f"{files_dir}/classification/classification_acc_*.csv"
                files = glob.glob(files_path)

            if not files:
                raise FileNotFoundError(f"No classification accuracy files found for in {files_path}")

            for file in files:
                try:
                    fold = int(file.split("_")[-1].split(".csv")[0])  # Extract run number
                    df = pd.read_csv(file)

                    # Get best epoch based on Accuracy
                    best_epoch = df.loc[df["Accuracy"].idxmax()]
                    max_acc = best_epoch["Accuracy"]
                    max_auc = best_epoch["AUC"]

                    results.append({
                        "Dataset": dataset,
                        "Fold": fold,
                        "Approach": approach,
                        "Accuracy": max_acc,
                        "AUC": max_auc
                    })

                except Exception as e:
                    print(f"[ERROR] Failed to process {file}: {e}")

    return pd.DataFrame(results)


def compute_accuracy_differences(df):
    """Aligns classification stats and computes pairwise accuracy differences from DataFrame."""
    aligned_data = []

    grouped = df.groupby(['Dataset', 'Fold'])

    for (dataset, fold), group in grouped:
        acc_map = {row['Approach']: row['Accuracy'] for _, row in group.iterrows()}

        if all(approach in acc_map for approach in ['scGen', 'FedscGen', 'FedscGen-SMPC']):
            aligned_data.append({
                "Dataset": dataset,
                "Fold": fold,
                "FedscGen - scGen": acc_map['FedscGen'] - acc_map['scGen'],
                "FedscGen-SMPC - scGen": acc_map['FedscGen-SMPC'] - acc_map['scGen']
            })

    return pd.DataFrame(aligned_data)


def classification_error_bar_plot(data_dir):
    df_stats = get_classification_stats(data_dir)
    accuracy_diff_df = compute_accuracy_differences(df_stats)
    accuracy_diff_df.to_csv(os.path.join(data_dir, "accuracy_diff.csv"), index=False)
    plot_accuracy_diff(accuracy_diff_df, data_dir)


def plot_smpc_wmw_heatmap(data_dir, plot_name="smpc-wmw.png", alpha=0.05):
    df = read_benchmarks(data_dir)
    df.to_csv(os.path.join(data_dir, "smpc_wmw_test.csv"), index=False)
    avg_df, p_df, adj_p_dict, datasets, metrics = manwhitney_test(df, alpha)
    wmw_heatmap(avg_df, p_df, alpha, plot_name)

    # Report significant results
    significant = (p_df < alpha).sum().sum()
    print(f"Number of significant metric-dataset pairs: {significant}")
    for dataset in datasets:
        for metric in metrics:
            adj_p = adj_p_dict[dataset][metric]
            if not np.isnan(adj_p) and adj_p < alpha:
                print(f"{dataset} - {metric}: Adjusted p-value = {adj_p:.4f}")


def manwhitney_test(df, alpha=0.05):
    metrics = ['NMI', 'GC', 'ILF1', 'ARI', 'EBM', 'KNN Acc', 'ASW_B', 'ASW_C']
    df.Dataset = df.Dataset.apply(lambda x: DS_MAP[x])
    datasets = df['Dataset'].unique()
    avg_diff = {}
    p_values = {}
    for dataset in datasets:
        p_values[dataset] = {}
        avg_diff[dataset] = {}
        df_dataset = df[df['Dataset'] == dataset]
        for metric in metrics:
            scgen_data = df_dataset[df_dataset['Approach'] == 'scGen'][metric].values
            fedscgen_data = df_dataset[df_dataset['Approach'] == 'FedscGen-SMPC'][metric].values
            if len(scgen_data) > 0 and len(fedscgen_data) > 0:
                stat, p = mannwhitneyu(scgen_data, fedscgen_data, alternative='two-sided')
                p_values[dataset][metric] = p
                avg_diff[dataset][metric] = np.mean(fedscgen_data) - np.mean(scgen_data)
            else:
                raise ValueError(f"Missing data for {dataset} - {metric}. Ensure both approaches have data.")

    flat_p_values = [p_values[d][m] for d in datasets for m in metrics if not np.isnan(p_values[d][m])]
    rejected, adj_p_values, _, _ = multipletests(flat_p_values, alpha=alpha, method='fdr_bh')
    adj_p_values = iter(adj_p_values)
    adj_p_dict = {}
    for dataset in datasets:
        adj_p_dict[dataset] = {}
        for metric in metrics:
            if not np.isnan(p_values[dataset][metric]):
                adj_p_dict[dataset][metric] = next(adj_p_values)
            else:
                raise ValueError(f"Missing data for {dataset} - {metric}")


    p_df = pd.DataFrame(adj_p_dict).T
    p_df.index.name = 'Dataset'
    p_df.columns.name = 'Metric'
    avg_df = pd.DataFrame(avg_diff).T
    avg_df.index.name = 'Dataset'
    avg_df.columns.name = 'Metric'
    return avg_df, p_df, adj_p_dict, datasets, metrics

def wmw_heatmap(avg_df, p_df, alpha, file_path):
    datasets = avg_df.index
    metrics = avg_df.columns
    annotations = avg_df.copy().astype(str)

    for i in range(len(datasets)):
        for j in range(len(metrics)):
            if not pd.isna(avg_df.iloc[i, j]):
                annotations.iloc[i, j] = f"{avg_df.iloc[i, j]:.2f}"
            else:
                raise ValueError("NaN value found in avg_df")

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(avg_df, annot=annotations, cmap='RdBu', center=0, ax=ax, cbar=False,
                annot_kws={"size": 32}, square=True, fmt="", vmin=-1, vmax=1,)

    ax.set_yticklabels(datasets, fontsize=30, rotation=0, ha='right')
    ax.set_xticklabels(metrics, fontsize=30, rotation=45, ha='right')

    cbar_ax = fig.add_axes([0.81, 0.25, 0.02, 0.5])
    norm = colors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    plt.colorbar(sm, cax=cbar_ax).ax.tick_params(labelsize=24)

    for i in range(len(datasets)):
        for j in range(len(metrics)):
            p = p_df.iloc[i, j]
            stars = ""
            if not pd.isna(p):
                if p < alpha:
                    stars = "*"
            if stars:
                ax.text(j + 0.5, i + 0.3, stars, fontsize=28, ha='center', va='bottom', color='black',
                        fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(file_path, dpi=300)
    plt.close()

def read_benchmarks(data_dir, filename="benchmark_metrics.csv"):
    """
    Parameters
    ----------
    data_dir: str
        Assuming data_dir has two subfolders: "scgen" and "fedscgen-smpc"
    Returns
    -------
    """
    def read_approach_benchmarks(approach):
        file_path = os.path.join(data_dir, approach, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            raise FileNotFoundError(file_path)
        df = df[df.Batch.isna() & df.BatchOut.isna() & (df.Inclusion == "all") & (df.Dataset != "MouseBrain")]
        df.drop(columns=["Inclusion", "Epoch", "Round", "Batch", "BatchOut", "N_Clients", "File"], inplace=True)
        return df
    fedscgen_smpc_df = read_approach_benchmarks("fedscgen-smpc")
    scgen_df = read_approach_benchmarks("scgen")
    def sanity_check(df, name):
        print(f"\n--- Sanity check for {name} ---")
        dataset_group_sizes = df.groupby(["Dataset", "Seed"]).size().unstack(fill_value=0)
        valid_seeds = dataset_group_sizes.columns[
            (dataset_group_sizes != 0).all(axis=0)
        ].tolist()
        print(f"Dropped seeds: {set(dataset_group_sizes.columns) - set(valid_seeds)}")
        # Filter dataframe to keep only rows with shared seeds
        df_filtered = df[df.Seed.isin(valid_seeds)].copy()
        return df_filtered
    fedscgen_smpc_df = sanity_check(fedscgen_smpc_df, "fedscgen-smpc")
    scgen_df = sanity_check(scgen_df, "scgen")
    sort_cols = ["Dataset", "Seed"]
    fedscgen_smpc_df = fedscgen_smpc_df.sort_values(by=sort_cols).reset_index(drop=True)
    scgen_df = scgen_df.sort_values(by=sort_cols).reset_index(drop=True)
    assert fedscgen_smpc_df["Dataset"].equals(scgen_df["Dataset"]), "❌ Dataset values do not match!"
    assert fedscgen_smpc_df["Seed"].equals(scgen_df["Seed"]), "❌ Seed values do not match!"
    combined_df = pd.concat([fedscgen_smpc_df, scgen_df], ignore_index=True)
    return combined_df

def validate_symmetry(df, ds_map, plot_dir):
    df_pivot = df.pivot(index=['Seed', 'Dataset'], columns='Approach')
    metrics = [c for c in df.columns if c not in ['Seed', 'Dataset', 'Approach']]
    df_diff = df_pivot.xs('FedscGen-SMPC', axis=1, level=1)[metrics] - df_pivot.xs('scGen', axis=1, level=1)[metrics]
    df_diff.reset_index(inplace=True)
    df = df_diff.melt(id_vars=['Seed', 'Dataset'], var_name='Metric', value_name='Difference')
    df = df[df.Metric != "ILF1"]
    df = df[df.Metric != "GC"]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df.Dataset = df.Dataset.apply(lambda x: ds_map[x])
    datasets = df.Dataset.unique()
    metrics =  df.Metric.unique()
    seeds = df.Seed.unique()
    diff_df = df # compute_differences(df, datasets, metrics, seeds)
    plot_all_distributions(diff_df, save_path=f"{plot_dir}/symmetry_grid.png")
    symmetry_df = calculate_symmetry_metrics(diff_df, datasets, metrics)
    symmetry_df.to_csv(f"{plot_dir}/symmetry_metrics.csv", index=False)
    # check_symmetry_holds(symmetry_df, skewness_threshold=0.5, bowley_threshold=0.3)
    check_symmetry_by_skewness(symmetry_df, skewness_threshold=1)
    return diff_df

def plot_all_distributions(diff_df, save_path="diff_grid_metrics_rows_with_inset.png"):
    metrics = sorted(diff_df["Metric"].unique())
    datasets = sorted(diff_df["Dataset"].unique())
    n_rows = len(metrics)
    n_cols = len(datasets)

    # Single subplot per dataset-metric pair (QQ with inset histogram)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))

    # Handle case where there's only one row or column
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            diffs = diff_df[
                (diff_df["Dataset"] == dataset) & (diff_df["Metric"] == metric)
                ]["Difference"].values

            # QQ plot
            ax = axes[i][j]
            stats.probplot(diffs, dist="norm", plot=ax)

            # Style the QQ plot
            ax.get_lines()[0].set_marker('o')
            ax.get_lines()[0].set_color('blue')
            ax.get_lines()[1].set_color('red')
            ax.get_lines()[1].set_linewidth(2)

            # Add inset for histogram + KDE (upper left, like your example)
            ax_inset = inset_axes(ax, width="40%", height="40%", loc="upper left")
            sns.histplot(diffs, kde=True, bins=5, ax=ax_inset, color="lightblue", edgecolor="gray")
            ax_inset.set_title("", fontsize=8)
            ax_inset.set_xlabel("", fontsize=6)
            ax_inset.set_ylabel("", fontsize=6)
            ax_inset.tick_params(labelsize=6)
            xlim = ax_inset.get_xlim()
            x_ticks = [xlim[0], xlim[1]]
            ax_inset.set_xticks(x_ticks)
            ax_inset.set_xticklabels([f"{x:.2f}" for x in x_ticks], fontsize=8)

            ylim = ax_inset.get_ylim()
            y_ticks = [1, int(np.floor(max(ylim)))]
            ax_inset.set_yticks(y_ticks)
            ax_inset.set_yticklabels(y_ticks, fontsize=12)
            ax_inset.yaxis.tick_right()
            ax_inset.spines["left"].set_visible(False)
            ax_inset.spines["top"].set_visible(False)
            ax_inset.tick_params(axis="y", which="both", left=False)
            ax_inset.tick_params(axis="x", which="both", top=False)
            ax.set_title("")
            # Labels
            # set off the y-axis labels
            if i == 0:
                ax.set_title(f"{dataset}", fontsize=22)
            if j == 0:
                ax.set_ylabel(f"{metric}", fontsize=22)
                ylim = ax.get_ylim()
                y_ticks = np.linspace(ylim[0], ylim[1], 4)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([f"{y:.2f}" for y in y_ticks], fontsize=14)
            else:
                ax.set_ylabel("")
                ax.get_yaxis().set_visible(False)

            ax.set_xlabel("")
            if i == n_rows - 1:
                xlim = ax.get_xlim()
                xmin, xmax = xlim
                x_ticks = [round(xmin, 2), 0, round(xmax, 2)]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([f"{x:.2f}" for x in x_ticks], fontsize=14)
            else:
                ax.get_xaxis().set_visible(False)


    plt.tight_layout(pad=.1, h_pad=0.1, w_pad=0.1)
    plt.savefig(save_path, dpi=300)
    plt.close()


# Function to calculate skewness and Bowley’s skewness
def calculate_symmetry_metrics(diff_df, datasets, metrics):
    symmetry_results = []
    for dataset in datasets:
        for metric in metrics:
            diffs = diff_df[(diff_df["Dataset"] == dataset) & (diff_df["Metric"] == metric)]["Difference"].values
            # skewness = stats.skew(diffs)
            skewness = median_skewness(diffs)
            q1, q2, q3 = np.percentile(diffs, [25, 50, 75])
            bowley = (q3 + q1 - 2 * q2) / (q3 - q1) if (q3 - q1) != 0 else np.nan
            symmetry_results.append({
                "Dataset": dataset,
                "Metric": metric,
                "Skewness": skewness,
                "Bowley_Skewness": bowley
            })
    return pd.DataFrame(symmetry_results)


def check_symmetry_holds(symmetry_df, skewness_threshold, bowley_threshold):
    # Filter tests where both skewness and Bowley's are within bounds
    symmetric_tests = symmetry_df[
        (symmetry_df["Skewness"].abs() <= skewness_threshold) &
        (symmetry_df["Bowley_Skewness"].abs() <= bowley_threshold)
        ]

    # Total tests
    total_tests = len(symmetry_df)
    symmetric_count = len(symmetric_tests)

    # Print summary
    print(f"Total tests: {total_tests}")
    print(
        f"Tests satisfying symmetry assumption (|Skewness| <= {skewness_threshold}, |Bowley| <= {bowley_threshold}): {symmetric_count}")

    # Print symmetric test names in Metric-Dataset format
    print("\nSymmetric Tests (Metric-Dataset):")
    for index, row in symmetric_tests.iterrows():
        test_name = f"{row['Metric']}-{row['Dataset']}"
        print(f"{test_name} (Skewness: {row['Skewness']:.3f}, Bowley: {row['Bowley_Skewness']:.3f})")

    # Print failing tests (if any)
    asymmetric_tests = symmetry_df[
        ~((symmetry_df["Skewness"].abs() <= skewness_threshold) &
          (symmetry_df["Bowley_Skewness"].abs() <= bowley_threshold))
    ]
    if not asymmetric_tests.empty:
        print("\nTests failing symmetry assumption (Metric-Dataset):")
        for index, row in asymmetric_tests.iterrows():
            test_name = f"{row['Metric']}-{row['Dataset']}"
            print(f"{test_name} (Skewness: {row['Skewness']:.3f}, Bowley: {row['Bowley_Skewness']:.3f})")
    symmetric_tests.to_csv("symmetric_tests.csv", index=False)
    asymmetric_tests.to_csv("asymmetric_tests.csv", index=False)

def median_skewness(diffs):
    median = np.median(diffs)
    n = len(diffs)
    mean_dev = diffs - median
    skewness_median = (np.mean(mean_dev**3)) / (np.mean(mean_dev**2)**1.5)
    return skewness_median

def check_symmetry_by_skewness(symmetry_df, skewness_threshold):
    # Filter tests where skewness is within bounds
    symmetric_tests = symmetry_df[
        (symmetry_df["Skewness"].abs() <= skewness_threshold)
    ]

    # Total tests
    total_tests = len(symmetry_df)
    symmetric_count = len(symmetric_tests)

    # Print summary
    print(f"Total tests: {total_tests}")
    print(f"Tests satisfying symmetry assumption (|Skewness| <= {skewness_threshold}): {symmetric_count}")

    # Print symmetric test names in Metric-Dataset format
    print("\nSymmetric Tests (Metric-Dataset):")
    for index, row in symmetric_tests.iterrows():
        test_name = f"{row['Metric']}-{row['Dataset']}"
        print(f"{test_name} (Skewness: {row['Skewness']:.3f}, Bowley: {row['Bowley_Skewness']:.3f})")

    # Print failing tests
    asymmetric_tests = symmetry_df[
        (symmetry_df["Skewness"].abs() > skewness_threshold)
    ]
    if not asymmetric_tests.empty:
        print("\nTests failing symmetry assumption (Metric-Dataset):")
        for index, row in asymmetric_tests.iterrows():
            test_name = f"{row['Metric']}-{row['Dataset']}"
            print(f"{test_name} (Skewness: {row['Skewness']:.3f}, Bowley: {row['Bowley_Skewness']:.3f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plot lineplot for the metrics")
    parser.add_argument('--scenario', type=str, choices=['datasets', 'tuning', 'batchout', 'kbet-diff', "wmw",
                                                         "scenarios", "classification", "lisi", "classification_error_bar"],
                        default='datasets')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--bo_metrics_file', type=str,
                        help="Path to the file containing the metrics for all batch out of HP dataset.")
    parser.add_argument('--bo_kbet_dir', type=str,
                        help="Path to the main directory containing the batch out kBET results.")

    parser.add_argument("--all_ds_metrics_file", type=str,
                        help="Path to the file containing the metrics for all datasets.")

    parser.add_argument("--hp_kbet_file", type=str, help="Path to the file containing the kBET results for HP dataset.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args()
    if args.scenario == "tuning":
        read_tuning_res(args.data_dir)
    elif args.scenario == "wmw":
        plot_smpc_wmw_heatmap(args.data_dir)
    elif args.scenario == "batchout":
        read_plot_batchout(args.data_dir)
    elif args.scenario == "scenarios":
        read_scenarios_metrics(args.data_dir)
    elif args.scenario == "datasets":
        read_all_inc_plot_heatmap(args.data_dir)
    elif args.scenario == "kbet-diff":
        read_kbet(args.data_dir)
    elif args.scenario == "lisi":
        collect_lisi_results(args.data_dir)
        plot_lisi(args.data_dir)
    elif args.scenario == "classification":
        classification_error_bar_plot(args.data_dir)