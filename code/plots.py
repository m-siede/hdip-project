# ############################################################################################################################
# IMPORTS
# ############################################################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GROUP_PALETTE = {
    "control": "#0072B2",
    "schizophrenia": "#D55E00",
    "Control": "#0072B2",
    "Schizophrenia": "#D55E00",
    0: "#0072B2",
    1: "#D55E00",
}

FEATURE_TYPE_PALETTE = {
    "manual": "#0072B2",
    "automatic": "#009E73",
    "baseline": "#999999",
}

FEATURE_SET_ORDER = [
    "standard",
    "activity_rhythm",
    "hourly",
    "bins",
    "all",
    "full",
    "reduced",
    "rfecv",
    "baseline",
]

GROUP_ORDER = ["control", "schizophrenia"]

GROUP_LABELS = {
    "control": "Control",
    "schizophrenia": "Schizophrenia"
}

GROUP_PALETTE = {
    "control": "#0072B2",        # blue
    "schizophrenia": "#D55E00"   # vermillion
}

# Utils

def require_columns(data, required_cols, func_name):
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(
            f"{func_name} requires columns {missing}, "
            f"but dataset only has: {list(data.columns)}"
        )

def set_thesis_plot_theme():
    """
    Unified thesis plotting style.
    Uses high contrast, colour-blind friendly colours, readable labels,
    and clean grid lines.
    """
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=1.05,
        rc={
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.edgecolor": "0.25",
            "axes.labelcolor": "0.1",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": "0.1",
            "ytick.color": "0.1",
            "legend.frameon": True,
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
        }
    )

def make_plot_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_plot(path, filename):
    path = make_plot_dir(path)
    plt.tight_layout()
    plt.savefig(path / filename, dpi=300, bbox_inches="tight")
    plt.close()

def clean_axis(ax, title=None, xlabel=None, ylabel=None, rotate_x=0):
    if title:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)

    if rotate_x:
        ax.tick_params(axis="x", rotation=rotate_x)

    sns.despine(ax=ax)

def get_group_order(data, group_col="group"):
    preferred = ["control", "schizophrenia", "Control", "Schizophrenia", 0, 1]
    existing = list(data[group_col].dropna().unique())

    ordered = [g for g in preferred if g in existing]
    ordered += [g for g in existing if g not in ordered]

    return ordered

def apply_ratio_axis_format(ax, values):
    """
    Formats a y or x axis as percentage when the values appear to be 0-1 ratios.
    """
    values = pd.Series(values).dropna()

    if len(values) > 0 and values.max() <= 1.0:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

############################################################################################################################
# PLOTS
############################################################################################################################

def plot_inactive_ratio_summary(data, path):
    """
    Thesis figure:
    Compares inactive ratio by group at daily-sample and subject-mean levels.
    """

    require_columns(
        data,
        ["group", "inactive_ratio", "person_id"],
        "plot_inactive_ratio_summary"
    )

    out_path = make_plot_dir(Path(path) / "zero_analysis")
    group_order = get_group_order(data)

    subject_df = (
        data
        .groupby(["person_id", "group"], as_index=False)["inactive_ratio"]
        .mean()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # Daily samples
    sns.boxplot(
        data=data,
        x="group",
        y="inactive_ratio",
        hue="group",
        order=group_order,
        palette=GROUP_PALETTE,
        showfliers=False,
        legend=False,
        ax=axes[0]
    )

    sns.stripplot(
        data=data,
        x="group",
        y="inactive_ratio",
        order=group_order,
        color="black",
        alpha=0.25,
        jitter=0.2,
        size=2.5,
        ax=axes[0]
    )

    clean_axis(
        axes[0],
        title="Daily Samples",
        xlabel="",
        ylabel="Inactive ratio",
        rotate_x=15
    )

    # Subject means
    sns.boxplot(
        data=subject_df,
        x="group",
        y="inactive_ratio",
        hue="group",
        order=group_order,
        palette=GROUP_PALETTE,
        showfliers=False,
        legend=False,
        ax=axes[1]
    )

    sns.stripplot(
        data=subject_df,
        x="group",
        y="inactive_ratio",
        order=group_order,
        color="black",
        alpha=0.55,
        jitter=0.18,
        size=4,
        ax=axes[1]
    )

    clean_axis(
        axes[1],
        title="Subject Means",
        xlabel="",
        ylabel="",
        rotate_x=15
    )

    apply_ratio_axis_format(axes[0], data["inactive_ratio"])

    fig.suptitle(
        "Inactive Ratio by Group",
        fontsize=14,
        fontweight="bold",
        y=1.03
    )

    save_plot(out_path, "inactive_ratio_summary.png")

def plot_daily_pattern(data, path):
    """
    Thesis figure:
    Average 24-hour activity pattern by group.
    Uses colour plus line style/markers for accessibility.
    """

    require_columns(data, ["hour", "activity", "group"], "plot_daily_pattern")

    out_path = make_plot_dir(Path(path) / "data_visualisations")

    avg = (
        data
        .groupby(["group", "hour"], as_index=False)["activity"]
        .mean()
    )

    group_order = get_group_order(avg)

    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=avg,
        x="hour",
        y="activity",
        hue="group",
        style="group",
        hue_order=group_order,
        style_order=group_order,
        palette=GROUP_PALETTE,
        markers=True,
        dashes=True,
        linewidth=2
    )

    ax.set_xticks(range(0, 24, 2))

    clean_axis(
        ax,
        title="Average 24-Hour Activity Pattern",
        xlabel="Hour of day",
        ylabel="Mean activity"
    )

    ax.legend(title="Group")

    save_plot(out_path, "daily_activity_pattern.png")

def plot_pca_projection(data, feature_cols, path, split_func):
    """
    Thesis figure:
    PCA projection of extracted features.
    This is exploratory only, not used for training.
    """

    require_columns(data, ["group"] + list(feature_cols), "plot_pca_projection")

    out_path = make_plot_dir(Path(path) / "eda_pca")

    X, _ = split_func(data, feature_cols)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2))
    ])

    X_pca = pipe.fit_transform(X)
    pca_model = pipe.named_steps["pca"]
    explained = pca_model.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "group": data["group"].values
    })

    group_order = get_group_order(pca_df)

    plt.figure(figsize=(7, 5.5))
    ax = sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="group",
        style="group",
        hue_order=group_order,
        style_order=group_order,
        palette=GROUP_PALETTE,
        alpha=0.75,
        s=55,
        edgecolor="black",
        linewidth=0.4
    )

    clean_axis(
        ax,
        title=(
            "PCA Projection of Activity Features\n"
            f"PC1={explained[0]:.1%}, PC2={explained[1]:.1%}"
        ),
        xlabel="Principal Component 1",
        ylabel="Principal Component 2"
    )

    ax.legend(title="Group")

    save_plot(out_path, "pca_projection.png")

    return pca_model, pca_df

def plot_pca_loadings(pca, feature_cols, path):
    """
    Thesis/supporting figure:
    Shows which features contribute most to PC1 and PC2.
    """

    out_path = make_plot_dir(Path(path) / "eda_pca")

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=feature_cols
    )

    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    for feature in loadings.index:
        x = loadings.loc[feature, "PC1"]
        y = loadings.loc[feature, "PC2"]

        ax.arrow(
            0,
            0,
            x,
            y,
            head_width=0.025,
            length_includes_head=True,
            alpha=0.75,
            color="#0072B2"
        )

        ax.text(
            x * 1.08,
            y * 1.08,
            feature,
            fontsize=8
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    clean_axis(
        ax,
        title="PCA Loadings: Feature Contributions",
        xlabel="PC1 loading",
        ylabel="PC2 loading"
    )

    save_plot(out_path, "pca_loadings.png")

    return loadings

def plot_zero_dominance(data, path):
    """
    Thesis figure:
    Shows which features are dominated by zero values.
    """

    require_columns(data, ["feature", "zero_pct"], "plot_zero_dominance")

    out_path = make_plot_dir(Path(path) / "zero_analysis")

    plot_df = data.sort_values("zero_pct", ascending=True)

    plt.figure(figsize=(7.5, max(4, 0.35 * len(plot_df))))
    ax = sns.barplot(
        data=plot_df,
        x="zero_pct",
        y="feature",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.5
    )

    ax.axvline(
        0.8,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="80% threshold"
    )

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    clean_axis(
        ax,
        title="Zero Percentage per Feature",
        xlabel="Percentage of zero values",
        ylabel="Feature"
    )

    ax.legend()

    save_plot(out_path, "zero_dominance.png")

def plot_feature_correlation_clustermap(data, feature_cols, path):
    """
    Thesis figure:
    Clustered feature correlation matrix.
    Useful for showing redundancy between engineered features.
    """

    require_columns(data, feature_cols, "plot_feature_correlation_clustermap")

    out_path = make_plot_dir(Path(path) / "feature_analysis")

    X = (
        data[feature_cols]
        .copy()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    corr = X.corr().fillna(0)

    g = sns.clustermap(
        corr,
        method="average",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        figsize=(9, 9),
        cbar_kws={"label": "Correlation"}
    )

    g.figure.suptitle(
        "Clustered Feature Correlation Matrix",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )

    g.figure.savefig(out_path / "feature_correlation_clustermap.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)

def plot_feature_importance(importance_df, path):
    """
    Thesis figure:
    Permutation feature importance ranked by mean decrease in balanced accuracy.
    """

    require_columns(
        importance_df,
        ["feature", "importance"],
        "plot_feature_importance"
    )

    out_path = make_plot_dir(Path(path) / "feature_selection")

    plot_df = importance_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(7.5, max(4, 0.35 * len(plot_df))))
    ax = sns.barplot(
        data=plot_df,
        x="importance",
        y="feature",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.5
    )

    ax.axvline(0, color="black", linewidth=1)

    clean_axis(
        ax,
        title="Permutation Feature Importance",
        xlabel="Mean decrease in balanced accuracy",
        ylabel=""
    )

    save_plot(out_path, "permutation_feature_importance.png")

def plot_rfecv_curve(rfecv_curve_df, path):
    """
    Plot RFECV cross-validation score against number of selected features.
    """

    require_columns(
        rfecv_curve_df,
        ["n_features", "mean_test_score", "std_test_score"],
        "plot_rfecv_curve"
    )

    out_path = make_plot_dir(Path(path) / "feature_selection")

    best_idx = rfecv_curve_df["mean_test_score"].idxmax()
    best_row = rfecv_curve_df.loc[best_idx]

    plt.figure(figsize=(7, 4.5))
    ax = plt.gca()

    ax.plot(
        rfecv_curve_df["n_features"],
        rfecv_curve_df["mean_test_score"],
        marker="o",
        linewidth=2,
        color="#0072B2"
    )

    ax.fill_between(
        rfecv_curve_df["n_features"],
        rfecv_curve_df["mean_test_score"] - rfecv_curve_df["std_test_score"],
        rfecv_curve_df["mean_test_score"] + rfecv_curve_df["std_test_score"],
        alpha=0.2,
        color="#0072B2"
    )

    ax.axvline(
        best_row["n_features"],
        linestyle="--",
        color="black",
        linewidth=1.2,
        label=f"Best: {int(best_row['n_features'])} features"
    )

    clean_axis(
        ax,
        title="RFECV Feature Selection Curve",
        xlabel="Number of selected features",
        ylabel="Mean cross-validated balanced accuracy"
    )

    ax.set_ylim(0, 1)
    ax.legend()

    save_plot(out_path, "rfecv_curve.png")

def plot_performance_vs_complexity(final_summary, path, metric="balanced_accuracy"):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    required = [
        "feature_set",
        "feature_set_type",
        "n_features",
        mean_col,
        std_col
    ]

    require_columns(final_summary, required, "plot_performance_vs_complexity")

    out_path = make_plot_dir(Path(path) / "model_comparison")

    plot_df = final_summary.copy()

    type_palette = {
        "manual": "#0072B2",
        "automatic": "#009E73",
        "baseline": "#999999"
    }

    label_map = {
        "standard": "Standard",
        "activity_rhythm": "Act. Rhythm",
        "bins": "Bins",
        "all": "All",
        "full": "Full",
        "reduced": "Reduced",
        "rfecv": "RFECV",
        "select_from_model": "SFM",
        "baseline": "Dummy"
    }

    plt.figure(figsize=(10, 6.5))
    ax = sns.scatterplot(
        data=plot_df,
        x="n_features",
        y=mean_col,
        hue="feature_set_type",
        style="feature_set_type",
        palette=type_palette,
        s=100,
        edgecolor="black",
        linewidth=0.6
    )

    offsets = [
        (0.08, 0.015),
        (0.08, -0.015),
        (0.08, 0.025),
        (0.08, -0.025),
        (0.12, 0.01),
        (0.12, -0.01),
    ]

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.errorbar(
            row["n_features"],
            row[mean_col],
            yerr=row[std_col],
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=3,
            alpha=0.75
        )

        dx, dy = offsets[i % len(offsets)]
        label = label_map.get(row["feature_set"], row["feature_set"])

        ax.annotate(
            label,
            xy=(row["n_features"], row[mean_col]),
            xytext=(row["n_features"] + dx, row[mean_col] + dy),
            textcoords="data",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.5)
        )

    pretty_metric = metric.replace("_", " ").title()

    ax.set_title("Performance vs Feature-Set Size", fontweight="bold")
    ax.set_xlabel("Number of features")
    ax.set_ylabel(f"Mean {pretty_metric}")
    ax.set_ylim(0, 1)
    ax.set_xlim(plot_df["n_features"].min() - 0.5, plot_df["n_features"].max() + 1.2)

    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    ax.legend(title="Feature-set type")

    save_plot(out_path, "performance_vs_feature_count.png")

def plot_model_comparison(final_summary, path):
    """
    Final thesis plot comparing manual feature sets,
    automatic feature sets, and dummy baseline.
    """

    required = [
        "feature_set",
        "feature_set_type",
        "n_features",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_mean",
        "f1_std"
    ]

    require_columns(final_summary, required, "plot_model_comparison")

    out_path = make_plot_dir(Path(path) / "model_comparison")

    feature_order = [
        "standard",
        "activity_rhythm",
        "bins",
        "all",
        "full",
        "reduced",
        "rfecv",
        "baseline"
    ]

    plot_df = final_summary.copy()

    plot_df["feature_set"] = pd.Categorical(
        plot_df["feature_set"],
        categories=[f for f in feature_order if f in plot_df["feature_set"].values],
        ordered=True
    )

    plot_df = plot_df.sort_values("feature_set")

    type_palette = {
        "manual": "#0072B2",
        "automatic": "#009E73",
        "baseline": "#999999"
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    metric_specs = [
        ("balanced_accuracy_mean", "balanced_accuracy_std", "Balanced Accuracy"),
        ("f1_mean", "f1_std", "F1 Score")
    ]

    for ax, (mean_col, std_col, title) in zip(axes, metric_specs):
        x = np.arange(len(plot_df))
        y = plot_df[mean_col].values
        yerr = plot_df[std_col].values

        colors = [
            type_palette.get(t, "#999999")
            for t in plot_df["feature_set_type"]
        ]

        ax.bar(
            x,
            y,
            yerr=yerr,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.7
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            plot_df["feature_set"].astype(str),
            rotation=25,
            ha="right"
        )

        ax.set_ylim(0, 1)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Cross-validated score")

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=colour,
            ec="black",
            label=label.title()
        )
        for label, colour in type_palette.items()
        if label in plot_df["feature_set_type"].values
    ]

    axes[1].legend(
        handles=legend_handles,
        title="Feature-set type",
        loc="lower right"
    )

    fig.suptitle(
        "Model Performance Across Feature Sets",
        fontsize=15,
        fontweight="bold",
        y=1.03
    )

    save_plot(out_path, "model_comparison.png")

def plot_best_manual_vs_automatic(best_df, path):
    """
    Direct comparison of the best manual and best automatic feature sets.
    Uses points rather than bars because the scores are close together.
    """

    required = [
        "feature_set",
        "feature_set_type",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "n_features"
    ]

    require_columns(best_df, required, "plot_best_manual_vs_automatic")

    out_path = make_plot_dir(Path(path) / "model_comparison")

    plot_df = best_df.copy()

    plot_df["label"] = (
        plot_df["feature_set_type"].str.title()
        + "\n"
        + plot_df["feature_set"].astype(str)
        + "\n"
        + plot_df["n_features"].astype(int).astype(str)
        + " features"
    )

    type_palette = {
        "manual": "#0072B2",
        "automatic": "#009E73"
    }

    colors = [
        type_palette.get(t, "#999999")
        for t in plot_df["feature_set_type"]
    ]

    x = np.arange(len(plot_df))
    y = plot_df["balanced_accuracy_mean"].values
    yerr = plot_df["balanced_accuracy_std"].values

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        markersize=10,
        capsize=5,
        color="black",
        ecolor="black",
        linewidth=1.2
    )

    for i, colour in enumerate(colors):
        ax.scatter(
            x[i],
            y[i],
            s=160,
            color=colour,
            edgecolor="black",
            zorder=3
        )

        ax.text(
            x[i],
            y[i] + yerr[i] + 0.015,
            f"{y[i]:.3f}",
            ha="center",
            fontsize=10
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"])

    ymin = max(0, min(y - yerr) - 0.05)
    ymax = min(1, max(y + yerr) + 0.05)
    ax.set_ylim(ymin, ymax)

    ax.set_title(
        "Best Manual vs Best Automatic Feature Set",
        fontweight="bold"
    )
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_xlabel("")

    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    save_plot(out_path, "best_manual_vs_automatic.png")

def patient_plots(patient_statistics, patient_num, group_name, plots_path):
    # plot_dir = Path(f"plots/{group_name}/{patient_num}")
    show_plots = False
    y = "activity"
    print(patient_statistics)
    create_plots(patient_statistics, x="timestamp", y=y, show=show_plots, plots_path=plots_path / "line_timestamp_vs_activity.png")
    create_plots(patient_statistics, x="timestamp", y=y, kind="scatter", plots_path=plots_path / "scatter_timestamp_vs_activity.png")
    create_plots(patient_statistics, x="hour", y=y, kind="box", plots_path=plots_path / "box_hour_vs_activity.png")
    create_plots(patient_statistics, x="hour", y=y, kind="box", plots_path=plots_path / "box_hour_vs_activity_short.png")
    create_plots(patient_statistics, x="hour", y=y, kind="box", yscale="log", plots_path=plots_path / "box_hour_vs_log_activity.png")

def combined_plots(combined_groups_statistics, plots_path):
    for col in ['mean', 'max', 'sd']:
        create_plots(
            combined_groups_statistics, x='group', y=col, kind='box', show=False, legend=False,
            plots_path=plots_path / "summary/box_ALL_{col}.png",
            figsize=(10, 8),
            title=f"Comparison of {col} activity between groups")

def create_plots(
    df, x, y, kind="line", hue=None, figsize=(12, 6), show=False,
    plots_path=None, yscale="linear", ylimits=None, legend=False, title=None
):
    """Create plots."""
    if not show and plots_path and Path(plots_path).exists():
        return

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", context="talk", palette="colorblind", rc=custom_params)

    # df = df.copy()
    fig, ax = plt.subplots(figsize=figsize)

    if kind == "box":
        sns.boxplot(
            data=df, x=x, y=y, ax=ax, hue=x, width=0.6, showfliers=False,
            showmeans=True, meanprops={
                                        "marker": "o",
                                        "markerfacecolor": "black",
                                        "markeredgecolor": "white",
                                        "markersize": 6
                                        },
            legend=legend
        )
        # ax.legend(handles=[], labels=[])

    elif kind == "line":
        sns.lineplot(data=df, x=x, y=y, hue=x, marker="o", linewidth=2.5, ax=ax, legend=legend)

    elif kind == "scatter":
        sns.scatterplot(data=df, x=x, y=y, hue=x, alpha=0.7, s=60, ax=ax, legend=legend)

    elif kind == "bar":
        sns.barplot(data=df, x=x, y=y, hue=x, ax=ax, errorbar="sd", legend=legend)

    elif kind == "area":
        # seaborn doesn't have area — simulate via line + fill
        if hue:
            for key, subdf in df.groupby(hue):
                sns.lineplot(data=subdf, x=x, y=y, ax=ax, label=key)
                ax.fill_between(subdf[x], subdf[y], alpha=0.3)
        else:
            sns.lineplot(data=df, x=x, y=y, ax=ax)
            ax.fill_between(df[x], df[y], alpha=0.3)

    # group x axis by date
    ax.set_xticks(ax.get_xticks())

    if ylimits:
        ax.set_ylim(ylimits)

    plt.title(title)
    # transform y to different kind e.g. linear or log scale
    plt.yscale(yscale)

    # eliminate whitespace
    # plt.tight_layout()

    # save graph to file
    if plots_path:
        plots_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(plots_path)
    if show:
        plt.show()
    plt.close()
