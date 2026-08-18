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
from matplotlib.patches import Patch

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
    missing_columns = [col for col in required_cols if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"{func_name} requires columns {missing_columns}, "
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

def make_plot_dir(plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir

def save_plot(fig, plots_dir, filename):
    plots_dir = make_plot_dir(plots_dir)
    
    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

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

# def get_group_order(data, group_col="group"):
#     preferred_order = ["control", "schizophrenia"]
#     existing_groups = (data[group_col].dropna().unique().tolist())

#     # ordered = [g for g in preferred_order if g in existing_groups]
#     # ordered += [g for g in existing_groups if g not in ordered]

#     return [group
#             for group in preferred_order
#             if group in existing_groups]

def get_group_order(data, group_col="group"):
    return [
        group
        for group in GROUP_ORDER
        if group in data[group_col].dropna().unique()
    ]
    
def apply_ratio_axis_format(ax, values):
    """
    Formats a y- axis as percentages when the values are within [0,1]
    """
    values = pd.Series(values).dropna()

    if ( not values.empty and values.min() >= 0 and values.max() <= 1.0):
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

############################################################################################################################
# PLOTS
############################################################################################################################

def plot_inactive_ratio_summary(data, plots_dir):
    """
    Thesis figure:
    Compares inactive ratio by group at daily-sample and subject-mean levels.
    """

    require_columns(
        data,
        ["group", "inactive_ratio", "person_id"],
        "plot_inactive_ratio_summary"
    )

    plots_dir = make_plot_dir(plots_dir)
    group_order = get_group_order(data)

    participant_mean_df = (
        data
        .groupby(["person_id", "group"], as_index=False)["inactive_ratio"]
        .mean()
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

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
        ax=ax[0]
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
        ax=ax[0]
    )

    clean_axis(
        ax[0],
        title="Daily Samples",
        xlabel="",
        ylabel="Inactive ratio",
        rotate_x=15
    )

    # Subject means
    sns.boxplot(
        data=participant_mean_df,
        x="group",
        y="inactive_ratio",
        hue="group",
        order=group_order,
        palette=GROUP_PALETTE,
        showfliers=False,
        legend=False,
        ax=ax[1]
    )

    
    sns.stripplot(
        data=participant_mean_df,
        x="group",
        y="inactive_ratio",
        order=group_order,
        color="black",
        alpha=0.55,
        jitter=0.18,
        size=4,
        ax=ax[1]
    )
    
    clean_axis(
        ax[1],
        title="Participant Means",
        xlabel="",
        ylabel="",
        rotate_x=15
    )

    apply_ratio_axis_format(ax[0], data["inactive_ratio"])

    fig.suptitle(
        "Inactive Ratio by Group",
        fontsize=14,
        fontweight="bold",
        y=1.03
    )

    save_plot(fig, plots_dir, "inactive_ratio_summary.png")

def plot_participant_inactive_ratio(data, plots_dir):
    require_columns(
        data,
        ["group", "inactive_ratio"],
        "plot_participant_inactive_ratio"
    )

    plots_dir = make_plot_dir(plots_dir)
    group_order = get_group_order(data)

    fig, ax = fig.subplots(figsize=(6, 4.5))

    sns.boxplot(
        data=data,
        x="group",
        y="inactive_ratio",
        hue="group",
        order=group_order,
        palette=GROUP_PALETTE,
        showfliers=False,
        legend=False,
        ax=ax
    )

    sns.stripplot(
        data=data,
        x="group",
        y="inactive_ratio",
        order=group_order,
        color="black",
        alpha=0.55,
        jitter=0.18,
        size=4,
        ax=ax
    )

    clean_axis(
        ax,
        title="Inactive Ratio by Group",
        xlabel="",
        ylabel="Inactive ratio"
    )

    apply_ratio_axis_format(ax, data["inactive_ratio"])

    save_plot(fig, plots_dir, "inactive_ratio_by_group.png")

def plot_daily_pattern(data, plots_dir):
    """
    Thesis figure:
    Average 24-hour activity pattern by group.
    Uses colour plus line style/markers for accessibility.
    """

    require_columns(data,["person_ID","hour", "activity", "group"], "plot_daily_pattern")

    plots_dir = make_plot_dir(plots_dir)

    # Mean activity for each participant at each hour
    participant_hourly_mean = (
        data
        .groupby(
            ["person_id", "group", "hour"],
            as_index=False
        )["activity"]
        .mean()
    )

    # Mean across participants within each group
    group_hourly_mean = (
        participant_hourly_mean
        .groupby(
            ["group", "hour"],
            as_index=False
        )["activity"]
        .mean()
    )

    group_order = get_group_order(group_hourly_mean)
    
    fig, axes = plt.subplots(figsize=(8,5))
    
    ax = sns.lineplot(
        data=group_hourly_mean,
        x="hour",
        y="activity",
        hue="group",
        style="group",
        hue_order=group_order,
        style_order=group_order,
        palette=GROUP_PALETTE,
        markers=True,
        dashes=True,
        linewidth=2,
        ax=ax
    )

    ax.set_xticks(range(0, 24, 2))

    clean_axis(
        ax,
        title="Average 24-Hour Activity Pattern",
        xlabel="Hour of day",
        ylabel="Mean activity"
    )

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles,
        [GROUP_LABELS.get(label, label) for label in labels],
        title="Group"
    )

    save_plot(fig, plots_dir, "daily_activity_pattern.png")

def plot_pca_projection(
    data,
    feature_cols,
    plots_dir
):
    """
    Plot the first two principal components of the extracted activity features.

    Features are standardised before PCA. This analysis is exploratory
    and is not used for classifier training.
    """

    require_columns(
        data,
        ["group"] + list(feature_cols),
        "plot_pca_projection"
    )

    plots_dir = make_plot_dir(plots_dir)

    # Prepare feature matrix
    feature_matrix = (
        data[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Standardise features before PCA
    pca_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2))
    ])

    pca_coordinates = pca_pipeline.fit_transform(
        feature_matrix
    )

    pca_model = pca_pipeline.named_steps["pca"]

    explained_variance_ratio = (
        pca_model.explained_variance_ratio_
    )

    pca_projection_df = pd.DataFrame({
        "PC1": pca_coordinates[:, 0],
        "PC2": pca_coordinates[:, 1],
        "group": data["group"].to_numpy()
    })

    group_order = get_group_order(
        pca_projection_df
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.scatterplot(
        data=pca_projection_df,
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
        linewidth=0.4,
        ax=ax
    )

    clean_axis(
        ax,
        title=(
            "PCA Projection of Activity Features\n"
            f"PC1 = {explained_variance_ratio[0]:.1%}, "
            f"PC2 = {explained_variance_ratio[1]:.1%}"
        ),
        xlabel="Principal Component 1",
        ylabel="Principal Component 2"
    )

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles,
        [GROUP_LABELS.get(label, label) for label in labels],
        title="Group"
    )

    save_plot(fig,plots_dir,"pca_projection.png")

    return pca_model, pca_projection_df

def plot_pca_loadings(pca, feature_cols, plots_dir):
    """
    Thesis/supporting figure:
    Shows which features contribute most to PC1 and PC2.
    """

    out_path = make_plot_dir(plots_dir)

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

def plot_feature_zero_proportions(
    data,
    plots_dir,
    zero_proportion_threshold=0.80
):
    """
    Plot the proportion of zero values for each feature.

    The dashed line indicates the zero-proportion threshold used
    when identifying features potentially related to inactivity.
    """

    require_columns(data,["feature","zero_proportion","is_inactivity_related"],"plot_inactivity_related_features")

    plots_dir = make_plot_dir(plots_dir)

    feature_inactivity_summary = (data.sort_values("zero_proportion",ascending=True))

    fig, ax = plt.subplots(figsize=(7.5,max(4,0.35 * len(feature_inactivity_summary))))

    sns.barplot(
        data=feature_inactivity_summary,
        x="zero_proportion",
        y="feature",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    ax.axvline(
        zero_proportion_threshold,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label=f"{zero_proportion_threshold:.0%} zero threshold"
    )

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    clean_axis(ax,
        title="Zero Proportion by Feature",
        xlabel="Proportion of zero values",
        ylabel="Feature"
    )

    ax.legend()

    save_plot(fig,plots_dir,"feature_zero_proportions.png")

def plot_feature_correlation_clustermap(
    data,
    feature_cols,
    plots_dir
):
    """
    Plot a clustered correlation matrix of engineered features
    to visualise groups of highly related or redundant features.
    """

    require_columns(
        data,
        feature_cols,
        "plot_feature_correlation_clustermap"
    )

    plots_dir = make_plot_dir(plots_dir)

    feature_data = (
        data[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Remove constant features because their correlations are undefined
    non_constant_features = [
        column
        for column in feature_data.columns
        if feature_data[column].nunique() > 1
    ]

    feature_data = feature_data[
        non_constant_features
    ]

    if feature_data.shape[1] < 2:
        raise ValueError(
            "plot_feature_correlation_clustermap requires "
            "at least two non-constant features."
        )

    correlation_matrix = feature_data.corr()

    cluster_grid = sns.clustermap(
        correlation_matrix,
        method="average",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        figsize=(9, 9),
        cbar_kws={"label": "Correlation"}
    )

    cluster_grid.figure.suptitle(
        "Clustered Feature Correlation Matrix",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )

    cluster_grid.savefig(
        plots_dir / "feature_correlation_clustermap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(cluster_grid.figure)

def plot_rfecv_curve(
    rfecv_curve_df,
    selected_feature_count,
    plots_dir
):
    """
    Plot mean cross-validation balanced accuracy against the
    number of features retained by RFECV.
    """

    require_columns(
        rfecv_curve_df,
        ["n_features","mean_test_score","std_test_score"],
        "plot_rfecv_curve"
    )

    plots_dir = make_plot_dir(plots_dir)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        rfecv_curve_df["n_features"],
        rfecv_curve_df["mean_test_score"],
        marker="o",
        linewidth=2,
        color="#0072B2"
    )

    ax.fill_between(
        rfecv_curve_df["n_features"],
        (
            rfecv_curve_df["mean_test_score"]
            - rfecv_curve_df["std_test_score"]
        ),
        (
            rfecv_curve_df["mean_test_score"]
            + rfecv_curve_df["std_test_score"]
        ),
        alpha=0.2,
        color="#0072B2"
    )

    ax.axvline(
        selected_feature_count,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label=f"Selected: {selected_feature_count} features"
    )

    clean_axis(
        ax,
        title="RFECV Feature Selection Curve",
        xlabel="Number of selected features",
        ylabel="Mean cross-validated balanced accuracy"
    )

    ax.set_ylim(0, 1)
    ax.legend()

    save_plot(fig,plots_dir,"rfecv_curve.png")

def plot_permutation_importance(
    permutation_importance_df,
    plots_dir
):
    """
    Plot permutation feature importance ranked by the mean decrease
    in balanced accuracy when each feature is permuted.
    """

    require_columns(
        permutation_importance_df,
        ["feature", "importance"],
        "plot_permutation_importance"
    )

    plots_dir = make_plot_dir(plots_dir)

    # Order features from lowest to highest importance so that the
    # most important features appear at the top of the horizontal chart
    plot_data = (
        permutation_importance_df
        .sort_values(
            "importance",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            max(4, 0.35 * len(plot_data))
        )
    )

    sns.barplot(
        data=plot_data,
        x="importance",
        y="feature",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    # Mark zero importance so positive and negative values are easy to distinguish
    ax.axvline(
        0,
        color="black",
        linewidth=1
    )

    clean_axis(
        ax,
        title="Permutation Feature Importance",
        xlabel="Mean decrease in balanced accuracy",
        ylabel="Feature"
    )

    save_plot(
        fig,
        plots_dir,
        "permutation_feature_importance.png"
    )

def plot_performance_vs_complexity(
    performance_summary,
    plots_dir,
    scoring_metric="balanced_accuracy"
):
    """
    Plot cross-validation performance against the number of features
    used by each manual and automatic feature set.

    Error bars show ±1 standard deviation across cross-validation folds.
    """

    # build col names for selected metric
    mean_score_column = f"{scoring_metric}_mean"
    std_score_column = f"{scoring_metric}_std"

    required_columns = [
        "feature_set",
        "feature_set_type",
        "n_features",
        mean_score_column,
        std_score_column
    ]

    require_columns(performance_summary, required_columns, "plot_performance_vs_complexity")

    plots_dir = make_plot_dir(plots_dir)

    # DummyClassifier does not meaningfully use the supplied features,
    # so it is from the feature-count comparison
    plot_data = (performance_summary.loc[performance_summary["feature_set_type"] != "baseline"].copy())

    if plot_data.empty:
        raise ValueError(
            "plot_performance_vs_complexity has no manual or "
            "automatic feature-set results to plot."
        )

    # Use distinct colours for different selection feature sets
    feature_set_type_palette = {
        "manual": "#0072B2",
        "automatic": "#009E73"
    }

    feature_set_labels = {
        "base": "Base",
        "activity_rhythm": "Activity Rhythm",
        "hourly_activity": "Hourly Activity",
        "six_hour_bins": "6-Hour Bins",
        "all_candidates": "All Candidates",
        "positive_permutation": "Positive Permutation",
        "rfecv_selected": "RFECV",
        "select_from_model": "SFM"
    }

    annotation_offsets = [
        (6, 8),
        (6, -12),
        (6, 14),
        (6, -18),
        (8, 6),
        (8, -8)
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    sns.scatterplot(
        data=plot_data,
        x="n_features",
        y=mean_score_column,
        hue="feature_set_type",
        style="feature_set_type",
        palette=feature_set_type_palette,
        s=100,
        edgecolor="black",
        linewidth=0.6,
        ax=ax
    )

    for index, (_, feature_set_result) in enumerate(
        plot_data.iterrows()
    ):
        ax.errorbar(
            feature_set_result["n_features"],
            feature_set_result[mean_score_column],
            yerr=feature_set_result[std_score_column],
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=3,
            alpha=0.75
        )

        x_offset, y_offset = annotation_offsets[index % len(annotation_offsets)]

        feature_set_label = feature_set_labels.get(
            feature_set_result["feature_set"],
            feature_set_result["feature_set"]
        )

        ax.annotate(
            feature_set_label,
            xy=(
                feature_set_result["n_features"],
                feature_set_result[mean_score_column]
            ),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=8,
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.5,
                "alpha": 0.5
            }
        )

    formatted_metric_name = (scoring_metric.replace("_", " ").title())

    clean_axis(
        ax,
        title="Performance vs Feature-Set Size",
        xlabel="Number of features",
        ylabel=f"Mean {formatted_metric_name}"
    )

    ax.set_ylim(0, 1)

    ax.set_xlim(
        max(0, plot_data["n_features"].min() - 1),
        plot_data["n_features"].max() + 1
    )

    handles, labels = ax.get_legend_handles_labels()
    feature_set_type_labels = {"manual": "Manual","automatic": "Automatic"}

    ax.legend(handles,
        [feature_set_type_labels.get(label, label)
        for label in labels],
        title="Feature-set type"
    )

    save_plot(fig,plots_dir,"performance_vs_feature_count.png")

def plot_feature_set_performance_comparison(
    performance_summary,
    plots_dir
):
    """
    Compare cross-validation performance across manual feature sets,
    automatically selected feature sets, and the dummy baseline.

    Error bars show ±1 standard deviation across cross-validation folds.
    """

    required_columns = [
        "feature_set",
        "feature_set_type",
        "n_features",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_mean",
        "f1_std"
    ]

    require_columns(
        performance_summary,
        required_columns,
        "plot_feature_set_performance_comparison"
    )

    plots_dir = make_plot_dir(plots_dir)

    # Define a consistent order for manual, automatic, and baseline results
    feature_set_order = [
        "base",
        "activity_rhythm",
        "hourly_activity",
        "six_hour_bins",
        "full",
        "all_candidates",
        "positive_permutation",
        "rfecv_selected",
        "select_from_model",
        "baseline"
    ]

    # Convert internal feature-set names to readable plot labels
    feature_set_labels = {
        "base": "Base",
        "activity_rhythm": "Activity Rhythm",
        "hourly_activity": "Hourly Activity",
        "six_hour_bins": "6-Hour Bins",
        "full": "All Manual",
        "all_candidates": "All Candidates",
        "positive_permutation": "Positive Permutation",
        "rfecv_selected": "RFECV",
        "select_from_model": "SFM",
        "baseline": "Dummy"
    }

    # Use consistent colours for each feature-set type
    feature_set_type_palette = {
        "manual": "#0072B2",
        "automatic": "#009E73",
        "baseline": "#999999"
    }

    plot_data = performance_summary.copy()

    # Preserve the predefined order while allowing unexpected feature sets
    # to remain visible rather than being converted to missing values
    available_feature_sets = plot_data["feature_set"].tolist()

    # Fix the NaNs in the plot
    ordered_feature_sets = [
        feature_set for feature_set in feature_set_order
        if feature_set in available_feature_sets
    ]

    ordered_feature_sets += [
        feature_set for feature_set in available_feature_sets
        if feature_set not in ordered_feature_sets
    ]

    plot_data["feature_set"] = pd.Categorical(
        plot_data["feature_set"],
        categories=ordered_feature_sets,
        ordered=True
    )

    plot_data = (plot_data.sort_values("feature_set").reset_index(drop=True))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Define the performance measures shown in each panel
    metric_columns = [
        (
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "Balanced Accuracy"
        ),
        (
            "f1_mean",
            "f1_std",
            "F1 Score"
        )
    ]

    x_positions = np.arange(len(plot_data))

    bar_colours = [
        feature_set_type_palette.get(
            feature_set_type,
            "#999999"
        )
        for feature_set_type
        in plot_data["feature_set_type"]
    ]

    x_axis_labels = [
        feature_set_labels.get(
            str(feature_set),
            str(feature_set)
        )
        for feature_set
        in plot_data["feature_set"]
    ]

    for ax, (
        mean_score_column,
        std_score_column,
        metric_title
    ) in zip(axes, metric_columns):

        # Plot mean CV performance with ±1 SD error bars
        ax.bar(
            x_positions,
            plot_data[mean_score_column],
            yerr=plot_data[std_score_column],
            capsize=4,
            color=bar_colours,
            edgecolor="black",
            linewidth=0.7
        )

        ax.set_xticks(x_positions)

        ax.set_xticklabels(
            x_axis_labels,
            rotation=25,
            ha="right"
        )

        ax.set_ylim(0, 1)

        clean_axis(
            ax,
            title=metric_title,
            xlabel="",
            ylabel=""
        )

    axes[0].set_ylabel("Cross-validated score")

    # Create a legend showing manual, automatic, and baseline categories
    legend_handles = [
        Patch(
            facecolor=colour,
            edgecolor="black",
            label=feature_set_type.title()
        )
        for feature_set_type, colour
        in feature_set_type_palette.items()
        if feature_set_type
        in plot_data["feature_set_type"].values
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

    save_plot(fig, plots_dir, "feature_set_performance_comparison.png")

def plot_best_manual_vs_automatic(
    best_feature_sets_summary,
    plots_dir
):
    """
    Compare the best-performing manual and automatic feature sets.

    Points show mean cross-validation balanced accuracy and error bars
    show ±1 standard deviation across cross-validation folds.
    """

    required_columns = [
        "feature_set",
        "feature_set_type",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "n_features"
    ]

    require_columns(
        best_feature_sets_summary,
        required_columns,
        "plot_best_manual_vs_automatic"
    )

    plots_dir = make_plot_dir(plots_dir)

    plot_data = best_feature_sets_summary.copy()

    # Convert internal feature-set names to readable plot labels
    feature_set_labels = {
        "base": "Base",
        "activity_rhythm": "Activity Rhythm",
        "hourly_activity": "Hourly Activity",
        "six_hour_bins": "6-Hour Bins",
        "all_candidates": "All Candidates",
        "positive_permutation": "Positive Permutation",
        "rfecv_selected": "RFECV",
        "select_from_model": "SFM"
    }

    # Create a multi-line label showing feature-set type, name, and size
    plot_data["display_label"] = (
        plot_data["feature_set_type"]
        .str.title()
        + "\n"
        + plot_data["feature_set"].map(
            lambda feature_set:
                feature_set_labels.get(feature_set, feature_set)
        )
        + "\n"
        + plot_data["n_features"]
        .astype(int)
        .astype(str)
        + " features"
    )

    # Use consistent colours for manual and automatic feature sets
    feature_set_type_palette = {
        "manual": "#0072B2",
        "automatic": "#009E73"
    }

    marker_colours = [
        feature_set_type_palette.get(
            feature_set_type,
            "#999999"
        )
        for feature_set_type
        in plot_data["feature_set_type"]
    ]

    x_positions = np.arange(len(plot_data))

    mean_scores = (
        plot_data["balanced_accuracy_mean"]
        .to_numpy()
    )

    score_std = (
        plot_data["balanced_accuracy_std"]
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    # Add ±1 SD error bars without drawing an additional marker
    ax.errorbar(
        x_positions,
        mean_scores,
        yerr=score_std,
        fmt="none",
        capsize=5,
        ecolor="black",
        elinewidth=1.2
    )

    # Plot each feature set using the colour for its selection type
    for index, marker_colour in enumerate(marker_colours):
        ax.scatter(
            x_positions[index],
            mean_scores[index],
            s=160,
            color=marker_colour,
            edgecolor="black",
            zorder=3
        )

        # Display the mean balanced accuracy above each error bar
        ax.text(
            x_positions[index],
            mean_scores[index] + score_std[index] + 0.015,
            f"{mean_scores[index]:.3f}",
            ha="center",
            fontsize=10
        )

    ax.set_xticks(x_positions)

    ax.set_xticklabels(
        plot_data["display_label"]
    )

    # Focus the y-axis on the observed score range while remaining
    # within the valid 0-1 range for balanced accuracy
    y_min = max(
        0,
        np.min(mean_scores - score_std) - 0.05
    )

    y_max = min(
        1,
        np.max(mean_scores + score_std) + 0.05
    )

    ax.set_ylim(y_min, y_max)

    clean_axis(
        ax,
        title="Best Manual vs Best Automatic Feature Set",
        xlabel="",
        ylabel="Mean balanced accuracy"
    )

    save_plot(
        fig,
        plots_dir,
        "best_manual_vs_automatic.png"
    )

def create_plot(
    data,
    x,
    y,
    kind="line",
    hue=None,
    figsize=(12, 6),
    show=False,
    output_path=None,
    yscale="linear",
    ylimits=None,
    legend=False,
    title=None
):
    """
    Create and optionally save a basic exploratory plot.
    """

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "box":
        sns.boxplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            width=0.6,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "white",
                "markersize": 6
            },
            legend=legend,
            ax=ax
        )

    elif kind == "line":
        sns.lineplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            marker="o",
            linewidth=2.5,
            legend=legend,
            ax=ax
        )

    elif kind == "scatter":
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            alpha=0.7,
            s=60,
            legend=legend,
            ax=ax
        )

    elif kind == "bar":
        sns.barplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            errorbar="sd",
            legend=legend,
            ax=ax
        )

    elif kind == "area":
        if hue:
            for group_name, group_data in data.groupby(hue):
                group_data = group_data.sort_values(x)

                sns.lineplot(
                    data=group_data,
                    x=x,
                    y=y,
                    ax=ax,
                    label=group_name
                )

                ax.fill_between(
                    group_data[x],
                    group_data[y],
                    alpha=0.3
                )
        else:
            sorted_data = data.sort_values(x)

            sns.lineplot(
                data=sorted_data,
                x=x,
                y=y,
                ax=ax
            )

            ax.fill_between(
                sorted_data[x],
                sorted_data[y],
                alpha=0.3
            )

    else:
        raise ValueError(
            f"Unknown plot kind: '{kind}'"
        )

    if ylimits is not None:
        ax.set_ylim(ylimits)

    ax.set_yscale(yscale)

    if title:
        ax.set_title(title)

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fig.tight_layout()

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    plt.close(fig)
    
def create_combined_group_plots(
    combined_group_statistics,
    plots_dir
):
    for feature_name in ["mean", "max", "sd"]:
        create_plot(
            data=combined_group_statistics,
            x="group",
            y=feature_name,
            kind="box",
            show=False,
            output_path=(
                plots_dir
                / "summary"
                / f"box_ALL_{feature_name}.png"
            ),
            figsize=(10, 8),
            title=(
                f"Comparison of {feature_name} "
                f"activity between groups"
            )
        )
def create_participant_plots(
    participant_activity,
    plots_dir
):
    create_plot(
        data=participant_activity,
        x="timestamp",
        y="activity",
        output_path=(
            plots_dir
            / "line_timestamp_vs_activity.png"
        )
    )

    create_plot(
        data=participant_activity,
        x="timestamp",
        y="activity",
        kind="scatter",
        output_path=(
            plots_dir
            / "scatter_timestamp_vs_activity.png"
        )
    )

    create_plot(
        data=participant_activity,
        x="hour",
        y="activity",
        kind="box",
        output_path=(
            plots_dir
            / "box_hour_vs_activity.png"
        )
    )

    create_plot(
        data=participant_activity,
        x="hour",
        y="activity",
        kind="box",
        yscale="log",
        output_path=(
            plots_dir
            / "box_hour_vs_log_activity.png"
        )
    )