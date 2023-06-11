import logging
from typing import Literal

from scvi.criticism import PosteriorPredictiveCheck as PPC
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from math import ceil

METRIC_CV_CELL = "cv_cell"
METRIC_CV_GENE = "cv_gene"
METRIC_DIFF_EXP = "diff_exp"

logger = logging.getLogger(__name__)

# from https://stackoverflow.com/a/28216751
def _add_identity(axes, *line_args, no_legend: bool = False, **line_kwargs):
    (identity,) = axes.plot([], [], *line_args, **line_kwargs, label="identity")

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect("xlim_changed", callback)
    axes.callbacks.connect("ylim_changed", callback)
    if not no_legend:
        axes.legend()
    return axes


class PPCPlot:
    """
    Plotting utilities for posterior predictive checks

    Parameters
    ----------
    ppc
        An instance of the :class:`~scvi.criticism.PosteriorPredictiveCheck` class containing the computed metrics
    """

    def __init__(
        self,
        ppc: PPC,
    ):
        self._ppc = ppc

    def plot_cv(self, model_name: str, cell_wise: bool = True, plt_type: Literal["scatter", "hist2d"] = "hist2d"):
        """
        Plot coefficient of variation metrics results.

        See our tutorials for a demonstration of the generated plot along with detailed explanations.

        Parameters
        ----------
        model_name
            Name of the model
        cell_wise
            Whether to plot the cell-wise or gene-wise metric
        plt_type
            The type of plot to generate.
        """
        metric = METRIC_CV_CELL if cell_wise is True else METRIC_CV_GENE
        model_metric = self._ppc.metrics[metric][model_name].values
        raw_metric = self._ppc.metrics[metric]["Raw"].values
        title = f"model={model_name} | metric={metric} | n_cells={self._ppc.raw_counts.shape[0]}"

        # log mae, pearson corr, spearman corr, R^2
        logger.info(
            f"{title}:\n"
            f"Mean Absolute Error={mae(model_metric, raw_metric):.2f},\n"
            f"Pearson correlation={pearsonr(model_metric, raw_metric)[0]:.2f}\n"
            f"Spearman correlation={spearmanr(model_metric, raw_metric)[0]:.2f}\n"
            f"r^2={r2_score(raw_metric, model_metric):.2f}\n"
        )

        # plot visual correlation (scatter plot or 2D histogram)
        if plt_type == "scatter":
            plt.scatter(model_metric, raw_metric)
        elif plt_type == "hist2d":
            h, _, _, _ = plt.hist2d(model_metric, raw_metric, bins=300)
            plt.close()  # don't show it yet
            a = h.flatten()
            cmin = np.min(a[a > 0])  # the smallest value > 0
            h = plt.hist2d(model_metric, raw_metric, bins=300, cmin=cmin, rasterized=True)
        else:
            raise ValueError(f"Invalid plt_type={plt_type}")
        ax = plt.gca()
        _add_identity(ax, color="r", ls="--", alpha=0.5)
        # add line of best fit
        a, b = np.polyfit(model_metric, raw_metric, 1)
        plt.plot(model_metric, a * model_metric + b, color="blue", ls="--", alpha=0.8, label="line of best fit")
        ax.legend()
        # add labels and titles
        plt.xlabel("model")
        plt.ylabel("raw")
        plt.title(title)

    def plot_diff_exp(
        self,
        model_name: str,
        plot_kind: str,
        figure_size=None,
    ):
        """
        Plot differential expression results.

        Parameters
        ----------
        model_name
            Name of the model
        var_gene_names_col
            Column name in the `adata.var` attribute containing the gene names, if different from `adata.var_names`
        figure_size
            Size of the figure to plot. If None, we will use a heuristic to determine the figure size.
        """
        de_metrics = self._ppc.metrics[METRIC_DIFF_EXP]["summary"]
        model_de_metrics = de_metrics[de_metrics["model"] == model_name]
        del model_de_metrics["lfc_mae"]  # not on the same scale as the other ones

        bar_colors = {
            "gene_overlap_f1": "#AEC9E7",
            "lfc_pearson": "#AA8FC4",
            "lfc_spearman": "#A4DE87",
            "pr_auc": "#D9A5CC",
            "roc_auc": "#769FCC",
        }

        if plot_kind == "lfc_scatterplots":
            group_de_metrics = self._ppc.metrics[METRIC_DIFF_EXP]["lfc_per_model_per_group"][model_name]
            # define subplot grid
            # https://engineeringfordatascience.com/posts/matplotlib_subplots/
            ncols = 4
            nrows = ceil(len(group_de_metrics.keys()) / ncols)
            figsize = figure_size if figure_size is not None else (20, 5 * nrows)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained")
            title = f"LFC 1-vs-all across groups, dots are genes" # x=raw de, y=approx de
            fig.suptitle(title, fontsize=18)
            axs_lst = axs.ravel()
            # https://stackoverflow.com/a/66799199
            for ax in axs_lst:
                ax.set_axis_off()
            # plot all
            i = 0
            for group in group_de_metrics.keys():
                ax = axs_lst[i]
                i += 1
                raw = group_de_metrics[group]["raw"]
                approx = group_de_metrics[group]["approx"]
                # ax.scatter(raw.to_list(), approx.to_list())
                nbin = 200
                h, _, _ = np.histogram2d(approx, raw, bins=nbin)
                a = h.flatten()
                cmin = np.min(a[a > 0])  # the smallest value > 0
                h = ax.hist2d(approx, raw, bins=nbin, cmin=cmin, rasterized=True, cmap=plt.cm.viridis)
                _add_identity(ax, no_legend=True, color="r", ls="--", alpha=0.5)
                # add title
                ax.set_title(
                    f"{group} \n pearson={pearsonr(raw, approx)[0]:.2f} - spearman={spearmanr(raw, approx)[0]:.2f} - mae={np.mean(np.abs(raw - approx)):.2f}"
                )
                ax.set_axis_on()
        elif plot_kind == "summary_violin":
            col_names = {
                "gene_overlap_f1": "Gene Overlap\nF1",
                "lfc_pearson": "LFC\nPearson",
                "lfc_spearman": "LFC\nSpearman",
                "pr_auc": "auPRC",
            }
            del model_de_metrics["roc_auc"]  # skip this for now, TODO fix/remove
            model_de_metrics.rename(columns=col_names, inplace=True)
            sns.set(rc={"figure.figsize": (6, 6)})
            sns.set_theme(style="white")
            sns.violinplot(model_de_metrics, palette=sns.color_palette("pastel"))
            plt.grid()
        elif plot_kind == "summary_box_with_obs":
            col_names = {
                "gene_overlap_f1": "Gene Overlap\nF1",
                "lfc_pearson": "LFC\nPearson",
                "lfc_spearman": "LFC\nSpearman",
                "pr_auc": "auPRC",
            }
            del model_de_metrics["roc_auc"]  # skip this for now, TODO fix/remove
            model_de_metrics.rename(columns=col_names, inplace=True)
            sns.set(rc={"figure.figsize": (6, 6)})
            sns.set_theme(style="white")
            sns.boxplot(model_de_metrics, showfliers = False, width=.6, palette=sns.color_palette("pastel"))
            sns.stripplot(model_de_metrics, size=4, color=".3", linewidth=1, palette=sns.color_palette("pastel"))
            plt.grid()
        elif plot_kind == "summary_barv":
            figsize = figure_size if figure_size is not None else (0.8 * len(model_de_metrics), 3)
            model_de_metrics.index.name = None
            model_de_metrics.plot.bar(figsize=figsize, color=bar_colors, edgecolor="black")
            plt.legend(loc="lower right")
            # add a grid and hide it behind plot objects.
            # from https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
            ax = plt.gca()
            ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
            ax.set(axisbelow=True)
        elif plot_kind == "summary_barh":
            figsize = figure_size if figure_size is not None else (3, 0.8 * len(model_de_metrics))
            model_de_metrics.index.name = None
            model_de_metrics.plot.barh(figsize=figsize, color=bar_colors, edgecolor="black")
            plt.legend(loc="upper right")
            # add a grid and hide it behind plot objects.
            # from https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
            ax = plt.gca()
            ax.xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
            ax.set(axisbelow=True)
        else:
            raise ValueError("Unknown plot_kind: {plot_kind}")
