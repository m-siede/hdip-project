############################################################################################################################
# IMPORTS
############################################################################################################################
from pathlib import Path

# Core libraries for data handling, utilities and plotting
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn: model selection, preprocessing, evaluation
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import (
    cross_validate, cross_val_score, cross_val_predict,
    StratifiedKFold, StratifiedGroupKFold,
    KFold, GroupKFold, train_test_split
)

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import confusion_matrix

# Feature selection & interpretability
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.inspection import permutation_importance

# Clustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Plot functions
from plots import (
    set_thesis_plot_theme,
    plot_daily_pattern,
    create_plot,
    create_combined_group_plots,
    create_participant_plots,
    plot_inactive_ratio_summary,
    plot_participant_inactive_ratio,
    plot_pca_projection,
    plot_pca_loadings,
    
    plot_permutation_importance,
    plot_feature_correlation_clustermap,
    plot_feature_zero_proportions,
    plot_rfecv_curve,
    
    plot_feature_set_performance_comparison,
    plot_performance_vs_complexity,
    plot_best_manual_vs_automatic,
    
    
)

############################################################################################################################
# MAIN
############################################################################################################################

def main():
    """
    Main function for the full ML pipeline.

    Runs:
    1. Daily-level analysis (each day = one sample)
    2. Person-level analysis (each person = one sample)

    Controls:
    - Feature set
    - Classifier choice
    - Validation strategy
    - Evaluation metrics
    """
    # Evaluation metrics used in cross-validation
    scoring_metrics = [
        "accuracy", "balanced_accuracy",
        "average_precision", "f1", "precision", "recall",
    ]

    # Reproducibility
    random_state = 42
    np.random.seed(random_state)

    # Cross-validation strategies
    group_k_fold = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )
    StratifiedK = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )
    
    set_thesis_plot_theme()
    
    preprocessing_by_participant, preprocessing_by_group = summarise_preprocessing_removals(
        save_path=Path("data/outputs/preprocessing_removal_summary.csv")
    )

    # Train models
    kwargs = {
        "model_names": ["Random_Forest", "Logistic_Regression", "Decision_Tree"],
        "train_function": samples_analysis,
        "create_plots": True,
        "random_state": random_state,
        "scoring_metrics": scoring_metrics,
    }
    run_models(
        output_dir="data/outputs/daily",
        sample_type="daily",
        cv_strategy=group_k_fold,
        **kwargs,
    )
    run_models(
        output_dir="data/outputs/person",
        sample_type="person",
        cv_strategy=StratifiedK,
        **kwargs,
    )

def init_model(model_name: str, random_state: int = 1):
    """Initialise a model by name"""
    if model_name == 'Random_Forest':
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state
        )
    elif model_name == 'Logistic_Regression':
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state
        )
    elif model_name == 'Decision_Tree':
        model = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state
        )
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    return model

def init_feature_select(random_state: int = 1):
    feature_select = SelectFromModel(
        estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        threshold="median"
    )
    return feature_select

def run_models(
    model_names,
    train_function,
    output_dir,
    sample_type,
    cv_strategy,
    create_plots,
    random_state,
    scoring_metrics
):
    """Run ML models."""
    all_results = {}

    # Train each model
    for model_name in model_names:
        name_ = model_name.replace("_", " ")
        all_results[name_] = train_function(
            mode=sample_type,
            classifier=init_model(model_name, random_state),
            classifier_name=model_name,
            cv_strategy=cv_strategy,
            feature_select=init_feature_select(random_state),
            create_plots=create_plots,
            random_state=random_state,
            scoring_metrics=scoring_metrics
        )
    
    output_dir = Path(output_dir)
    # Save with unformatted text
    combine_and_save_classifier_results(
        classifier_results=all_results,
        sample_type=sample_type,
        output_dir=output_dir / "combined_results_across_classifiers",
        round=3,
        format_text=False
    )
    # Save again with formatted text
    combine_and_save_classifier_results(
        classifier_results=all_results,
        sample_type=sample_type,
        output_dir=output_dir / "reformatted_combined_results_across_classifiers",
        round=3,
        format_text=True
    )

def samples_analysis(
    mode, classifier, classifier_name, cv_strategy,
    feature_select, create_plots, random_state, scoring_metrics
):

    if mode not in ["daily", "person"]:
        raise ValueError(f"Expected 'mode' to be 'daily' or 'person'. Got: {mode}")

    # -----------------------------
    # SETUP
    # -----------------------------
    # Set output paths
    input_path = Path(f"data/outputs/{mode}/{classifier_name}/combined.csv")
    output_subdir = Path(f"data/outputs/{mode}/{classifier_name}/")
    plots_dir = Path(f"plots/{mode}/{classifier_name}/")
    
    # Read dataframe if possible
    if input_path.exists():
        df_samples = pd.read_csv(input_path)
        feature_cols = get_model_feature_columns(df_samples)
    else:
        # Otherwise, compile data into dataframe
        if mode == "daily":  # Build the dataset from raw time-series files
            df_samples, feature_cols = build_daily_level_dataset(plots_dir=plots_dir)
        else:
            df_samples, feature_cols = build_participant_level_dataset(input_path, classifier_name, create_plots, plots_dir)
    
    # Set groups and data variables (for daily data)
    cv_groups = None
    if mode == "daily":
        df_samples["subject_id"] = (
            df_samples["group"].astype(str)
            + "_"
            + df_samples["person_id"].astype(str)
        )
        cv_groups = df_samples["subject_id"]
        feature_cols = [col for col in feature_cols if col != "IS"]
        save_dataframe_to_csv(df_samples, output_subdir / "combined.csv")

    data = df_samples.copy()
    automatic_feature_candidates = get_candidate_features_for_automatic_feature_selection(feature_cols)
    
    # -----------------------------
    # DATA QUALITY CHECKS
    # ------------------------------
    # Get class distribution to check imbalance
    class_distribution = data["group"].value_counts(normalize=True)
    
    # print("\n --- Class distribution: --- \n")
    # print(class_distribution)

    # Check all generated features for zero values, NaNs etc
    features_quality_report = audit_feature_quality(data=data.copy())
    
    # Feature-based plots
    if create_plots:
        if mode == "daily":
            plot_inactive_ratio_summary(data=data, plots_dir=plots_dir / "EDA")
        else:
            plot_inactive_ratio_summary(data=data, plots_dir=plots_dir / "EDA")
            if {"hour", "activity", "group"}.issubset(data.columns):
                plot_daily_pattern(
                    data=data, plots_dir=plots_dir / "EDA"
                )
    
    # -----------------------------
    # PCA VISUALISATION
    # -----------------------------
    pca_model, pca_df = plot_pca_projection(
        data=data,
        feature_cols=feature_cols,
        plots_dir=plots_dir / "EDA"
    )

    # pca_loadings = plot_pca_loadings(
    #     pca=pca_model,
    #     feature_cols=feature_cols,
    #     path=plots_dir / "EDA"
    # )

    # -----------------------------
    # MANUAL FEATURE SETS
    # -----------------------------
    # build the manual feature set
    manual_feature_set_definitions = build_manual_feature_sets(feature_cols)
    
    # optional print
    # for feature_set_name, feature_names in manual_feature_set_definitions.items():
    #     print(feature_set_name, len(feature_names), feature_names)

    manual_feature_set_cv_results = evaluate_feature_sets_with_cv(
        data=data,
        feature_sets=manual_feature_set_definitions,
        classifier=classifier,
        cv_strategy=cv_strategy,
        cv_groups=cv_groups,
        scoring_metrics=scoring_metrics,
        create_plots=create_plots,
        plots_dir=plots_dir / "feature_selection" / "manual"
    )

    # add label
    for manual_cv_result in manual_feature_set_cv_results:
        manual_cv_result["metadata"]["feature_set_type"] = "manual"

    # get performance summary and rank the results by balanced accuracy mean
    manual_feature_set_cv_performance_summary = summarise_cv_performance(manual_feature_set_cv_results)
    manual_feature_set_cv_performance_summary = rank_feature_sets_results_by_metric(
        manual_feature_set_cv_performance_summary,
        primary_metric="balanced_accuracy_mean"
    )
    # compare results across cv folds
    manual_feature_set_cv_fold_comparison = compare_feature_sets_across_cv_folds(
        manual_feature_set_cv_results,
        scoring_metric="balanced_accuracy"
    )

    # -----------------------------
    # AUTOMATIC FEATURE SETS
    # -----------------------------
    
    # FEATURE PERMUTATION IMPORTANCE
    # Determine which features are the most important for the model through 
    # permutation importance, measure the performance drop through shuffeling.
    # Plot the features and their associated clusters
    feature_permutation_importance_df, permutation_positive_importance_features = compute_permutation_importance(
        data=data,
        classifier=classifier,
        feature_cols=automatic_feature_candidates,
        create_plots=create_plots,
        cv_groups=cv_groups,
        cv_strategy=cv_strategy,
        random_state=random_state,
    )
    
    # Extract the found feature clusters
    correlated_features_assigned_clusters = cluster_correlated_features(
        data=data,
        feature_cols=automatic_feature_candidates,
        n_clusters=4,
        create_plots=create_plots,
        plots_dir=plots_dir / "feature_selection" / "automatic"
    )

    # Combine the feature clusters, determine the feature importance and plot
    clusters_importance_summary = merge_feature_clusters_and_importance(
        correlated_features_assigned_clusters,
        feature_permutation_importance_df,
        create_plots=create_plots,
        plots_dir=plots_dir / "feature_selection" / "automatic",
    )
    
    # RFECV FEATURE SELECTION
    # Find the smallest necessary feature set through Recursive Feature Elimination with Cross-Validation
    rfecv_selected_features, rfecv_feature_rankings, rfecv_selection_metadata, rfecv_performance_curve_df = run_rfecv_feature_selection(
        data=data.copy(),
        classifier=classifier,
        cv_strategy=cv_strategy,
        automatic_feature_candidates=automatic_feature_candidates.copy(),
        create_plots=create_plots,
        plots_dir= plots_dir /"feature_selection" / "automatic",
        cv_groups=cv_groups
    )
    
    # Combine all automatic feature sets
    automatic_feature_set_definitions = build_automatic_selection_feature_sets(
        automatic_feature_candidates=automatic_feature_candidates,
        permutation_importance=feature_permutation_importance_df,
        rfecv_features=rfecv_selected_features
    )
    # -----------------------------
    # AUTOMATIC FEATURE SET EVALUATION
    # -----------------------------
    # Evaluate pre-defined automatic feature sets
    automatic_feature_set_cv_results = evaluate_feature_sets_with_cv(
            data=data,
            feature_sets=automatic_feature_set_definitions,
            classifier=classifier,
            cv_strategy=cv_strategy,
            cv_groups=cv_groups,
            scoring_metrics=scoring_metrics,
            create_plots=create_plots,
            plots_dir=plots_dir / "feature_selection" / "automatic"
        )
    
    # Run select from model
    select_from_model_cv_result = evaluate_select_from_model(
            data=data,
            feature_cols=automatic_feature_candidates.copy(),
            classifier=classifier,
            cv_strategy=cv_strategy,
            cv_groups=cv_groups,
            scoring_metrics=scoring_metrics,
            plots_dir=plots_dir / "feature_selection" / "automatic",
            create_plots=create_plots,
            feature_select=feature_select)
        
    # Combine automatic feature sets
    automatic_feature_set_cv_results.append(select_from_model_cv_result)
    
    # Label automatic results
    for automatic_cv_result in automatic_feature_set_cv_results:
            automatic_cv_result["metadata"]["feature_set_type"] = "automatic"
    
    # Summarise and rank the results
    automatic_feature_set_performance_summary = summarise_cv_performance(automatic_feature_set_cv_results)
    automatic_feature_set_performance_summary = rank_feature_sets_results_by_metric(
        automatic_feature_set_performance_summary,
        primary_metric="balanced_accuracy_mean"
    )

    # Compare automatic feature sets across folds
    automatic_feature_sets_comparison_by_fold = compare_feature_sets_across_cv_folds(
        automatic_feature_set_cv_results,
        scoring_metric="balanced_accuracy"
    )
    
    # -----------------------------
    # FINAL FEATURE SETS EVALUATION
    # -----------------------------
    
    # Combine manual and automatic CV results
    all_feature_set_cv_results = manual_feature_set_cv_results + automatic_feature_set_cv_results
    
    # Summarise performance across CV folds
    all_feature_set_performance_summary = summarise_cv_performance(all_feature_set_cv_results)

    # Add efficiency/stability measures
    all_feature_set_performance_summary = add_performance_efficiency_metrics(
        all_feature_set_performance_summary,
        scoring_metric="balanced_accuracy"
    )
    
    # Rank by raw balanced accuracy
    all_feature_set_performance_summary["overall_rank"] = (
    all_feature_set_performance_summary["balanced_accuracy_mean"]
    .rank(ascending=False,method="min").astype(int)
)

    # Rank by performance adjusted for CV variability (stability rank)
    all_feature_set_performance_summary["stability_rank"] = (
        all_feature_set_performance_summary["stability_adjusted_score"]
        .rank(ascending=False,method="min")
        .astype(int)
    )

    # Compare feature sets across individual CV folds
    all_feature_set_fold_comparisons = (compare_feature_sets_across_cv_folds(
            all_feature_set_cv_results,
            scoring_metric="balanced_accuracy"
        )
    )

    # Sort for final presentation
    all_feature_set_performance_summary = (all_feature_set_performance_summary
        .sort_values("stability_rank")
    .reset_index(drop=True)
    )
    # print("\n=== MANUAL AND AUTOMATIC FEATURE SETS SUMMARY ===")
    # print(combined_summary)

    # print("\n=== FEATURE SET FOLD COMPARISON ===")
    # print(automatic_feature_sets_comparison_by_fold)
    
    # -----------------------------
    # INACTIVITY-SENSITIVITY ANALYSIS
    # -----------------------------
    # Check which features are dominated or highly correlated with no activity/inactivity
    # of subjects and Testing if classifiers overly rely on inactive ratio
    # and model relies on zero dominated features to make predictions
    
    # create feature sets to test
    inactivity_analysis_feature_sets = {
        **{
            f"manual_{feature_set_name}": feature_names
            for feature_set_name, feature_names in manual_feature_set_definitions.items()
        },
        **{
            f"automatic_{name}": features
            for name, features in automatic_feature_set_definitions.items()
        }
    }

    # run sensitivity analysis
    (inactivity_all_feature_sets_cv_results,
    inactivity_related_features,
    inactivity_ablation_cv_results,
    inactivity_shuffled_cv_results,
    inactivity_baseline_full_datase_cv_results) = run_inactivity_sensitivity_analysis(
    data=data,
    cv_groups=cv_groups,
    classifier=classifier,
    cv_strategy=cv_strategy,
    feature_sets=inactivity_analysis_feature_sets,
    create_plots=create_plots,
    plots_dir=plots_dir / "inactivity_analysis",
    scoring_metrics=scoring_metrics,
    random_state=random_state,
    )  # type: ignore
    
    # summarise results
    inactivity_all_feature_sets_cv_results_summary = summarise_cv_performance(inactivity_all_feature_sets_cv_results)
    inactivity_ablation_cv_results_summary = summarise_cv_performance(inactivity_ablation_cv_results)

    if create_plots:
        plot_feature_zero_proportions(data=inactivity_related_features, plots_dir=plots_dir / "inactivity_analysis")

    # -----------------------------
    # FEATURE SELECTION ROBUSTNESS ANALYSIS
    # -----------------------------

    permutation_importance_stability_df = evaluate_permutation_importance_stability(
        data=data,
        feature_cols=automatic_feature_candidates,
        classifier=classifier,
        cv_strategy=cv_strategy,
        cv_groups=cv_groups,
        random_state=random_state
    )

    # Compare feature selection methods
    feature_selection_comparison = compare_feature_selection_methods(
        feature_permutation_importance_df,
        rfecv_selected_features,
        inactivity_related_features
    )
    feature_selection_comparison_df = pd.DataFrame([feature_selection_comparison.copy()])
    
    # -----------------------------
    # BASELINE MODEL
    # ------------------------------
    ''' Running a dummy to provide a consistent baseline'''
    
    dummy_baseline_cv_result = evaluate_dummy_baseline(
        data=data,
        feature_cols=feature_cols,
        cv_strategy=cv_strategy,
        cv_groups=cv_groups,
        scoring_metrics=scoring_metrics,
        create_plots=create_plots,
        random_state=random_state,
        plots_dir=plots_dir / "baseline"
    )

    dummy_baseline_cv_result["metadata"]["feature_set_type"] = "baseline"
    
    # dummy_summary = summarise_results(dummy_result)

    # -----------------------------
    # FINAL FEATURE SET PERFORMANCE COMPARISON
    # -----------------------------
    
    final_comparison_cv_results = manual_feature_set_cv_results + automatic_feature_set_cv_results + [dummy_baseline_cv_result]

    final_performance_summary = summarise_cv_performance(final_comparison_cv_results)
    
    best_manual_vs_automatic_sets_summary = get_best_manual_and_automatic_sets(
        final_performance_summary,
        scoring_metric="balanced_accuracy_mean"
    )

    if create_plots:
        plot_feature_set_performance_comparison(performance_summary=final_performance_summary,plots_dir=plots_dir/"final")
        plot_performance_vs_complexity(performance_summary=final_performance_summary,scoring_metric="balanced_accuracy", plots_dir=plots_dir/"final")
        plot_best_manual_vs_automatic(best_feature_sets_summary=best_manual_vs_automatic_sets_summary, plots_dir=plots_dir/"final")


    
    save_dataframe_to_csv(class_distribution, output_subdir / "eda" / "class_distribution.csv")
    save_dataframe_to_csv(features_quality_report, output_subdir / "eda" / "feature_quality_report.csv")
    # save_dataframe_to_csv(pca_loadings, output_subdir / "eda/pca_loadings.csv")
    save_dataframe_to_csv(correlated_features_assigned_clusters, output_subdir / "feature_selection" / "feature_clusters.csv")
    save_dataframe_to_csv(feature_permutation_importance_df, output_subdir / "feature_selection" / "permutation_importance.csv")
    save_dataframe_to_csv(rfecv_feature_rankings, output_subdir / "feature_selection" / "rfecv_feature_ranking.csv")
    save_dataframe_to_csv(rfecv_performance_curve_df, output_subdir / "feature_selection" / "rfecv_performance_curve.csv")
    save_dataframe_to_csv(inactivity_all_feature_sets_cv_results_summary, output_subdir / "inactivity_analysis" / "sensitivity_summary.csv")
    save_dataframe_to_csv(inactivity_related_features, output_subdir / "inactivity_analysis" / "inactivity_dominated_features.csv")
    save_dataframe_to_csv(inactivity_ablation_cv_results_summary, output_subdir / "inactivity_analysis" / "ablation_summary.csv")
    save_dataframe_to_csv(all_feature_set_fold_comparisons, output_subdir / "final_comparison" / "fold_comparisons.csv")
    save_dataframe_to_csv(final_performance_summary, output_subdir / "final_comparison"/ "performance_summary.csv")
    save_dataframe_to_csv(permutation_importance_stability_df, output_subdir / "feature_selection" / "permutation_importance_stability.csv")
    save_dataframe_to_csv(feature_selection_comparison_df, output_subdir / "feature_selection" / "feature_selection_comparison.csv")

    return {
        # DATA
        "data": data,
        "feature_cols": feature_cols,
        # EDA
        "class_distribution": class_distribution,
        "pca_projection": pca_df,
        # "pca_loadings": pca_loadings,
        "feature_quality_report" : features_quality_report,
        # MANUAL FEATURE SETS
        "manual__feature_set_definitions": manual_feature_set_definitions,
        "manual_feature_set_cv_results": manual_feature_set_cv_results,
        "manual_feature_set_performance_summary": manual_feature_set_cv_performance_summary,
        "manual_feature_set_fold_comparison": manual_feature_set_cv_fold_comparison,
        # PERMUTATION IMPORTANCE AND CLUSTERING
        "permutation_importance": feature_permutation_importance_df,
        "positive_importance_features": permutation_positive_importance_features,
        "feature_cluster_assignments": correlated_features_assigned_clusters,
        "clusters_importance_summary": clusters_importance_summary,
        # RFECV
        "rfecv_selected_features": rfecv_selected_features,
        "rfecv_feature_rankings": rfecv_feature_rankings,
        "rfecv_selection_metadata": rfecv_selection_metadata,
        "rfecv_performance_curve_df": rfecv_performance_curve_df,
        # AUTOMATIC FEATURE SETS
        "automatic_feature_set_definitions": automatic_feature_set_definitions,
        "automatic_feature_set_cv_results": automatic_feature_set_cv_results,
        "automatic_feature_set_performance_summary": automatic_feature_set_performance_summary,
        "automatic_feature_sets_comparison_by_fold": automatic_feature_sets_comparison_by_fold,
        # INACTIVITY ANALYSIS
        "inactivity_all_feature_sets_cv_results": inactivity_all_feature_sets_cv_results,
        "identified_inactivity_related_features": inactivity_related_features,
        "inactivity_ablation_cv_results": inactivity_ablation_cv_results,
        "inactivity_shuffled_cv_results": inactivity_shuffled_cv_results,
        "inactivity_baseline_full_datase_cv_results": inactivity_baseline_full_datase_cv_results,
        "inactivity_performance_summary": inactivity_all_feature_sets_cv_results_summary,
        "inactivity_ablation_performance_summary": inactivity_ablation_cv_results_summary,
        # COMBINED FEATURE SET ANALYSIS
        "all_feature_set_cv_results": all_feature_set_cv_results,
        "all_feature_set_performance_summary": all_feature_set_performance_summary,
        "all_feature_set_fold_comparisons": all_feature_set_fold_comparisons,
        # FINAL COMPARISON
        "final_comparison_cv_results": final_comparison_cv_results,
        "final_performance_summary": final_performance_summary,
        "best_manual_vs_automatic_sets_summary": best_manual_vs_automatic_sets_summary,
        # FEATURE SELECTION ROBUSTNESS
        "permutation_importance_stability_df": permutation_importance_stability_df,
        "feature_selection_comparison_df": feature_selection_comparison_df
    }


############################################################################################################################
# BUILD DATASETS
############################################################################################################################
# Sample per Person Dataset
############################################################################################################################
def build_participant_level_dataset(path, classifier_name, create_plots, plots_dir):
    schizophrenia_standard_activity_statistics, _ = build_group_dataset_participant_level("schizophrenia", 22, classifier_name, create_plots, plots_dir)
    control_standard_activity_statistics, _ = build_group_dataset_participant_level("control", 32, classifier_name, create_plots, plots_dir)

    combined_groups_standard_activity_statistics = pd.concat(
        [schizophrenia_standard_activity_statistics, control_standard_activity_statistics],
        ignore_index=True
    )
    combined_groups_standard_activity_statistics = save_dataframe_to_csv(combined_groups_standard_activity_statistics, path / "combined_groups_standard_activity_statistics.csv")

    feature_cols = get_model_feature_columns(combined_groups_standard_activity_statistics)

    if create_plots:
        create_combined_group_plots(combined_groups_standard_activity_statistics, plots_dir)

    return combined_groups_standard_activity_statistics, feature_cols

def build_group_dataset_participant_level(group_name, n_participants, classifier_name, create_plots, plots_dir):
    """Read CSV files for a specified group (patients/control/etc.) and evaluate."""
    statistics_csv = Path(f"data/outputs/person/{classifier_name}/{group_name}_statistics.csv")
    all_data_csv = Path(f"data/outputs/person/{classifier_name}/{group_name}_data.csv")

    # Read CSV file if it exists, else import patient data and statistics and combine into one file
    if statistics_csv.exists() and all_data_csv.exists():
        combined_participant_standard_activity_statistics = pd.read_csv(statistics_csv)
        combined_participant_data = pd.read_csv(all_data_csv)
    else:
        all_participants_standard_activity_statistics = []
        all_participant_data = []
        
        for person_id in range(1, n_participants + 1):
            print('person_id:', person_id)
            create_plots = person_id < 11
            participant_standard_activity_statistics, patient_data = load_and_clean_participant_level_data(person_id, group_name, create_plots, plots_dir)
            
            engineered_features = extract_engineered_features(patient_data, daily=False)
            participant_standard_activity_statistics.update(engineered_features)

            participant_standard_activity_statistics["person_id"] = person_id
            participant_standard_activity_statistics["group"] = group_name

            patient_data["person_id"] = person_id
            patient_data["group"] = group_name

            all_participants_standard_activity_statistics.append(participant_standard_activity_statistics)
            all_participant_data.append(patient_data)
    
        # Save as CSV
        combined_participant_standard_activity_statistics = save_dataframe_to_csv(all_participants_standard_activity_statistics, statistics_csv)
        
        all_participants_activity_data = pd.concat(all_participant_data, ignore_index=True)
        combined_participant_data = save_dataframe_to_csv(all_participants_activity_data, all_data_csv)
        
        if create_plots:
            all_participants_standard_activity_statistics['person_no'] = all_participants_standard_activity_statistics.index
            create_plot(
                all_participants_standard_activity_statistics, x='person_no', y='mean', kind='bar', show=False, legend=False,
                title=f"Bar chart of mean activity per person in the {group_name} group",
                plots_dir=plots_dir / f"bar_{group_name}_mean_activity.png"
            )

            # Box plot of activity per person in group
            create_plot(
                all_participants_activity_data, x='person_no', y="activity", kind='box', legend=False,
                show=False, title=f"Boxplot of activity per person in the {group_name} group",
                plots_dir=plots_dir / f"box_{group_name}_activity.png"
            )

    return combined_participant_standard_activity_statistics, combined_participant_data

def load_and_clean_participant_level_data(participant_id, group_name, create_plots, plots_dir):
    """
    Read data for each patient, remove duplicate timestamps,
    remove incomplete days, remove days with mean activity of zero,
    then calculate statistics.
    """
    participant_data = pd.read_csv(
        f"data/{group_name}/{group_name}_{participant_id}.csv"
    )

    # Create timestamp, date and hour variables
    participant_data["timestamp"] = pd.to_datetime(participant_data["timestamp"])
    participant_data = participant_data.drop_duplicates(subset="timestamp")

    participant_data["date"] = participant_data["timestamp"].dt.date
    participant_data["hour"] = participant_data["timestamp"].dt.hour.astype(int)

    # Remove incomplete days
    complete_days = participant_data.groupby("date").size()
    complete_days = complete_days[complete_days >= 1440].index

    participant_data = participant_data[
        participant_data["date"].isin(complete_days)
    ].copy()

    # Remove days where mean activity is zero
    participant_data = remove_zero_mean_days(participant_data)

    participant_standard_activity_statistics = extract_standard_activity_statistics(participant_data)

    if create_plots:
        create_participant_plots(
            participant_data,
            plots_dir
        )

    return participant_standard_activity_statistics, participant_data


############################################################################################################################
# Sample per Day Dataset
############################################################################################################################
def build_daily_level_dataset(plots_dir):
    '''Build a dataset that calculates features for each day that will be used as samples'''
    schizophrenia = build_daily_samples_for_group("schizophrenia", 22)
    control = build_daily_samples_for_group("control", 32)

    df = pd.concat([schizophrenia, control], ignore_index=True)
    create_combined_group_plots(df, plots_dir)
    feature_cols = get_model_feature_columns(df)
    
    return df, feature_cols

def build_daily_samples_for_group(group_name, n_participants):
    
    rows = []
    
    for person_id in range(1, n_participants + 1):
        df = load_and_clean_participant_daily_level(group_name, person_id)

        # group by day
        for date, day_df in df.groupby("date"):
            if len(day_df) < 1440:
                continue
            
            
            standard_activity_statistics = extract_standard_activity_statistics(day_df)
            engineered_features = extract_engineered_features(day_df, daily=True)

            row = {**standard_activity_statistics, **engineered_features}
            row["person_id"] = person_id
            row["date"] = date
            row["group"] = group_name

            rows.append(row)

    return pd.DataFrame(rows)

def load_and_clean_participant_daily_level(group_name, person_id, min_samples_per_day=1440):
    df = pd.read_csv(
        f"data/{group_name}/{group_name}_{person_id}.csv"
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp")

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour.astype(int)

    # Remove incomplete days
    complete_days = df.groupby("date").size()
    complete_days = complete_days[complete_days >= min_samples_per_day].index

    df = df[df["date"].isin(complete_days)].copy()

    # Remove days where mean activity is zero
    df = remove_zero_mean_days(df)

    return df


############################################################################################################################
# EXPLORATORY
############################################################################################################################
# Extract features and statistics
############################################################################################################################
def extract_standard_activity_statistics(data):
    mean_hourly_activity = (
        data
        .groupby("hour")["activity"]
        .mean()
        .to_dict()
    )

    standard_statistics = {
        "min": data["activity"].min(),
        "max": data["activity"].max(),
        "mean": data["activity"].mean(),
        "median": data["activity"].median(),
        "sd": data["activity"].std(),
    }

    for hour in range(24):
        standard_statistics[f"mean_activity_{hour}"] = mean_hourly_activity.get(hour, 0)
        
    return standard_statistics

def extract_engineered_features(data, daily = True):
    
    df = data.sort_values("timestamp").copy()
    
    # ------------------------------------------------------------------
    # Overall activity characteristics
    # ------------------------------------------------------------------
    activity_variance = df["activity"].var()
    is_constant_activity = (pd.notna(activity_variance) and activity_variance == 0)
    
    inactive_ratio = (df["activity"] == 0).mean()
    
    # ------------------------------------------------------------------
    # Day and night activity
    # ------------------------------------------------------------------
    # Day and night mean activity
    # create day and night
    day_time = df[(df["hour"] >= 8) & (df["hour"] < 20)]
    night_time = df[(df["hour"] < 8) | (df["hour"] >= 20)]
    
    day_time_mean_activity = day_time["activity"].mean()
    night_time_mean_activity = night_time["activity"].mean()
    
    day_night_ratio = ensure_safe_divide(day_time_mean_activity, night_time_mean_activity)


    # ------------------------------------------------------------------
    # Inactivity streak
    # ------------------------------------------------------------------
    # Calculate inactivity streaks separately within each day so that
    # removed or missing days cannot create artificial continuous streaks
    
    
    daily_longest_inactivity = []

    for _, daily_data in df.groupby("date"):
        
        is_inactive = (daily_data["activity"] == 0).astype(int)
        streak_groups = (is_inactive != is_inactive.shift()).cumsum()
        inactivity_streaks = (is_inactive.groupby(streak_groups).cumsum())
        daily_longest_inactivity.append(inactivity_streaks.max())

    longest_inactivity = (
        max(daily_longest_inactivity)
        if daily_longest_inactivity
        else 0
    )
    
    # ------------------------------------------------------------------
    # Successive activity change
    # ------------------------------------------------------------------

    # Calculate minute-to-minute changes within days only
    mean_absolute_change_activity = (df.groupby("date")["activity"].diff().abs().mean())

    # ------------------------------------------------------------------
    # Circadian / activity-rhythm features
    # ------------------------------------------------------------------

    IV = compute_IV(df)
    RA = compute_RA(df)

    # ------------------------------------------------------------------
    # Features available at both daily and participant levels
    # ------------------------------------------------------------------

    features = {
        "inactive_ratio": inactive_ratio,
        "longest_inactivity": longest_inactivity,
        "is_constant_activity": int(is_constant_activity),
        "12_hour_day_activity": day_time_mean_activity,
        "12_hour_night_activity": night_time_mean_activity,
        "day_night_ratio": day_night_ratio,
        "mean_absolute_change_activity": mean_absolute_change_activity,
        "IV": IV,
        "RA": RA,
    }

    # ------------------------------------------------------------------
    # Participant-level features requiring multiple days
    # ------------------------------------------------------------------

    if not daily:
        
        daily_mean_activity = (df.groupby("date")["activity"].mean())

        daily_mean_activity_variance = (
            daily_mean_activity.var()
            if len(daily_mean_activity) > 1
            else np.nan
        )

        features["daily_mean_activity_variance"] = (daily_mean_activity_variance)

        # IS measures regularity of the activity pattern across days,
        # so it is only calculated for participant-level data
        features["IS"] = compute_IS(df)

    # ------------------------------------------------------------------
    # Six-hour activity bins
    # ------------------------------------------------------------------

    time_bins = ["night", "morning", "afternoon","evening"]
    df["time_bin"] = pd.cut(
        df["hour"], bins=[0, 6, 12, 18, 24],
        labels=time_bins,
        right=False
    )


    six_hour_bin_activity_means = (
        df.groupby(
            "time_bin",
            observed=True
        )["activity"]
        .mean()
        .to_dict()
    )

    for time_bin in time_bins:
        features[f"activity_{time_bin}"] = (six_hour_bin_activity_means.get(time_bin, 0))

    # ------------------------------------------------------------------
    # Six-hour inactivity bins
    # ------------------------------------------------------------------

    six_hour_bin_inactivity = (
        df.groupby("time_bin", observed=True)["activity"]
        .apply(lambda activity:
                (activity == 0).mean()).to_dict())

    for time_bin in time_bins:
        features[f"inactive_{time_bin}"] = (
            six_hour_bin_inactivity.get(time_bin,0))

    # ------------------------------------------------------------------
    # Diagnostic warning
    # ------------------------------------------------------------------

    if is_constant_activity:
        participant_id = (
            df["person_id"].iloc[0]
            if "person_id" in df.columns
            else "unknown"
        )

        print(
            "Warning: constant activity detected "
            f"for participant {participant_id}"
        )

    return features

def compute_IS(data):
    
    if data["date"].nunique() < 2:
        return np.nan
    
    hourly = (
        data.set_index("timestamp")["activity"]
        .resample("1h")
        .mean()
        .dropna()
    )
    
    if len(hourly) < 24:
        return np.nan
    
    hourly_profile = hourly.groupby(hourly.index.hour).mean()
    
    # Make sure all 24 hours are present
    if len(hourly_profile) < 24:
        return np.nan
    
    n = len(hourly)
    p = 24
    
    overall_mean = hourly.mean()
    
    
    num = (n * ((hourly_profile - overall_mean) ** 2).sum())
    den = (p * ((hourly - overall_mean) ** 2).sum())

    return ensure_safe_divide(num, den, default=np.nan)

def compute_IV(data):

    """
    Compute Intradaily Variability (IV).

    Higher values indicate greater fragmentation of the
    rest-activity rhythm.
    """
    hourly = (data.resample("1h", on="timestamp")["activity"].mean())
    
        # Calculate differences before dropping missing hours so that
        # gaps in the recording are not treated as consecutive observations
    successive_differences = (hourly.diff().dropna())
    valid_hourly = hourly.dropna()
    
    # make sure there is a valid consequtive pair of observations
    if len(valid_hourly) < 2 or len(successive_differences) == 0:
        return np.nan
    
    mean_squared_successive_difference = (successive_differences.pow(2).mean())
    
    activity_variance = (valid_hourly.var(ddof=0))

    return ensure_safe_divide(mean_squared_successive_difference, activity_variance, default=np.nan)

def compute_RA(data):
    """
    Computes Relative Amplitude using consecutive hourly windows.

    M10 = mean activity during the most active consecutive 10-hour period
    L5 = mean activity during the least active consecutive 5-hour period
    RA = (M10 - L5) / (M10 + L5)
    """
    hourly = (
        data
        .groupby("hour")["activity"]
        .mean()
        .reindex(range(24))
    )
    
    # Require complete 24 hour profile
    if hourly.isna().any():
        return np.nan

    values = hourly.to_numpy()

    # Duplicate the 24-hour profile to handle windows crossing midnight
    circular_values = np.concatenate([values, values])

    m10_values = [
        circular_values[start:start + 10].mean()
        for start in range(24)
    ]

    l5_values = [
        circular_values[start:start + 5].mean()
        for start in range(24)
    ]

    M10 = max(m10_values)
    L5 = min(l5_values)

    return ensure_safe_divide(M10 - L5, M10 + L5, default=np.nan)


############################################################################################################################
# Audit created features
############################################################################################################################
def audit_feature_quality(data, feature_cols=None, verbose=True):

    if feature_cols is None:
        X = data.select_dtypes(include=["number"]).copy()
    else:
        X = data[feature_cols].copy()

    report = []

    for col in X.columns:
        series = X[col]

        n_total = len(series)
        n_nan = series.isna().sum()
        n_inf = np.isinf(series).sum()
        n_inactive = (series == 0).sum()

        variance = series.var()
        std = series.std()

        report.append({
            "feature": col,
            "nan_count": n_nan,
            "nan_pct": n_nan / n_total,
            "inf_count": n_inf,
            "inactive_count": n_inactive,
            "zero_proportion": n_inactive / n_total,
            "variance": variance,
            "std_dev": std,
            "is_constant": variance == 0 or np.isnan(variance),
        })

    report_df = pd.DataFrame(report)

    # Flag problematic features
    issues = []

    for _, row in report_df.iterrows():
        problems = []
        if row["nan_count"] > 0:
            problems.append("NaNs")
        if row["inf_count"] > 0:
            problems.append("Infs")
        if row["is_constant"]:
            problems.append("Constant")
        if row["zero_proportion"] > 0.95:
            problems.append("Mostly zero")
        if row["nan_pct"] > 0.2:
            problems.append("High NaNs")
        issues.append(", ".join(problems) if problems else "OK")

    report_df["issues"] = issues

    if verbose:
        print("\n=== FEATURE AUDIT SUMMARY ===")
        print(report_df.sort_values(by="issues", ascending=False))
        print("\n=== PROBLEMATIC FEATURES ===")
        print(report_df[report_df["issues"] != "OK"])
    
    return report_df

def build_manual_feature_sets(feature_cols):
    """
    Build manually defined feature sets for comparison.

    These feature sets compare:
    - base summary features
    - activity-rhythm features
    - hourly activity features
    - 6-hour bin features
    - the full set of available modelling features
    """

    feature_cols = list(feature_cols)

    hourly_activity_features = [
        col for col in feature_cols
        if col.startswith("mean_activity_")
    ]

    bin_labels = ["night", "morning", "afternoon", "evening"]
    
    six_hour_bin_features = [
        col for col in feature_cols
        if col in [f"activity_{b}" for b in bin_labels]
        or col in [f"inactive_{b}" for b in bin_labels]
    ]

    activity_rhythm_features = [
        col for col in ["IS", "IV", "RA"]
        if col in feature_cols
    ]

    base_features = [
        col for col in [
            "mean",
            "sd",
            "inactive_ratio",
            "day_activity",
            "night_activity",
            "day_night_ratio",
            "longest_inactivity",
            "mean_absolute_change_activity",
            "daily_mean_activity_variance"
        ]
        if col in feature_cols
    ]

    feature_sets = {
        "base": base_features,
        "activity_rhythm": activity_rhythm_features,
        "hourly_activity": hourly_activity_features,
        "six_hour_bins": six_hour_bin_features,
        "full": feature_cols
    }

    # Remove empty feature sets
    feature_sets = {
        name: features
        for name, features in feature_sets.items()
        if len(features) > 0
    }
    return feature_sets

def identify_inactivity_related_features(
    data,
    feature_cols,
    feature_set_name="unknown",
    zero_threshold=0.80,
    corr_threshold=0.70
):
    """
    Detect features that are either:
    1. dominated by zero values, or
    2. highly correlated with inactive_ratio.

    """

    # Keep only valid features from the current feature set
    valid_features = filter_existing_features(
        data=data,
        feature_cols=feature_cols,
        name=f"inactivity dominance: {feature_set_name}"
    )

    X = data[valid_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # set the reference measure of overall inactivity from the full dataset
    inactivity_reference = data["inactive_ratio"]

    results = []
    
    for col in X.columns:
        series = X[col]

        # Proportion of observations that is 0
        zero_proportion = (series == 0).mean()

        # Get the correlation with overall inactivity
        if (
            col != "inactive_ratio"     # ignore inactive ratio feature and constant variables
            and series.nunique(dropna=True) > 1
            and inactivity_reference.nunique(dropna=True) > 1
        ):
            corr = series.corr(inactivity_reference)
        else:
            corr = np.nan

        # determine if feature is flagged for high 0 proportion or correlation
        is_high_inactive = zero_proportion >= zero_threshold

        is_high_corr = (
            not np.isnan(corr)
            and abs(corr) >= corr_threshold
        )

        results.append({
            "feature_set": feature_set_name,
            "feature": col,
            "zero_proportion": zero_proportion,
            "corr_with_inactive_ratio": corr,
            "is_high_inactive": is_high_inactive,
            "is_high_corr_with_inactivity": is_high_corr,
            "is_inactivity_related": is_high_inactive or is_high_corr
        })

    # Convert to datframe and sort by strongest results first
    df_results = (
        pd.DataFrame(results)
        .sort_values(by="zero_proportion", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n=== INACTIVITY-DOMINATED FEATURE ANALYSIS: {feature_set_name} ===")
    print(df_results)

    print(f"\n=== FLAGGED FEATURES: {feature_set_name} ===")
    print(df_results[df_results["is_inactivity_related"]])

    return df_results

def evaluate_inactivity_dependence_in_feature_sets(
    data,
    classifier,
    cv_strategy,
    feature_set_name,
    feature_cols,
    create_plots,
    plots_dir,
    cv_groups,
    scoring_metrics,
    random_state
):
    """
    Runs inactivity-dominance sensitivity analysis for ONE feature set.
    """

    feature_cols = filter_existing_features(
        data=data,
        feature_cols=feature_cols,
        name=f"Inactivity dominance: {feature_set_name}"
    )

    plots_base_path = Path(plots_dir) / feature_set_name
    
    all_feature_sets_cv_results = []
    all_ablation_results = []

    # -----------------------------
    # Detect zero-dominated features
    # -----------------------------
    inactivity_related_features = identify_inactivity_related_features(
        data=data,
        feature_cols=feature_cols,
        feature_set_name=feature_set_name,
        corr_threshold=0.70
    )

    inactivity_flagged_features = (
        inactivity_related_features.loc[inactivity_related_features["is_inactivity_related"], "feature"]
        .tolist()
    )

    inactivity_flagged_features = [
        f for f in inactivity_flagged_features
        if f in feature_cols
    ]

    if ("inactive_ratio" in feature_cols
        and "inactive_ratio" not in inactivity_flagged_features):
        inactivity_flagged_features.append("inactive_ratio")

    print(f"\nRemoved features for {feature_set_name}: {inactivity_flagged_features}")

    # -----------------------------
    # Baseline for this feature set
    # -----------------------------
    baseline_full_dataset_scores = evaluate_classifier_with_cv(
        classifier=classifier,
        data=data,
        cv_strategy=cv_strategy,
        scoring_metrics=scoring_metrics,
        cv_groups=cv_groups,
        create_plots=create_plots,
        feature_cols=feature_cols,
        plots_dir=plots_base_path / "baseline",
        classifier_evaluated=f"{classifier.__class__.__name__}_{feature_set_name}_baseline",
        metadata={
            "analysis": "inactivity_dominance",
            "feature_set": feature_set_name,
            "feature_set_type": "inactivity_baseline",
            "n_features": len(feature_cols),
            "features": list(feature_cols)
        }
    )
    all_feature_sets_cv_results.append(baseline_full_dataset_scores)

    # -----------------------------
    # Shuffled-y sanity check
    # -----------------------------
    shuffled_Y_validation_check_results = evaluate_classifier_with_cv(
        classifier=classifier,
        data=data,
        cv_strategy=cv_strategy,
        scoring_metrics=scoring_metrics,
        cv_groups=cv_groups,
        feature_cols=feature_cols,
        create_plots=create_plots,
        plots_dir= plots_base_path / "shuffled_Y",
        shuffled=True,
        classifier_evaluated=f"{classifier.__class__.__name__}_{feature_set_name}_shuffled_y",
        metadata={
            "analysis": "validation_check",
            "feature_set": feature_set_name,
            "feature_set_type": "shuffled_y",
            "n_features": len(feature_cols),
            "features": list(feature_cols),
            "shuffled_y": True
        }
    )
    all_feature_sets_cv_results.append(shuffled_Y_validation_check_results)

    # -----------------------------
    # Individual feature ablations
    # -----------------------------
    for feature in inactivity_flagged_features:

        ablated_feature_cols = [
            f for f in feature_cols
            if f != feature
        ]

        if len(ablated_feature_cols) == 0:
            print(f"Skipping ablation for {feature}: no features left.")
            continue

        ablation_scores = evaluate_classifier_with_cv(
            classifier=classifier,
            data=data,
            cv_strategy=cv_strategy,
            scoring_metrics=scoring_metrics,
            cv_groups=cv_groups,
            create_plots=create_plots,
            feature_cols=ablated_feature_cols,
            plots_dir= plots_base_path / "ablation",
            classifier_evaluated=f"{classifier.__class__.__name__}_{feature_set_name}_remove_{feature}",
            metadata={
                "analysis": "feature_ablation",
                "feature_set": feature_set_name,
                "feature_set_type": "ablation",
                "feature_removed": feature,
                "n_features": len(ablated_feature_cols),
                "features": list(ablated_feature_cols)
            }
        )

        all_ablation_results.append(ablation_scores)

    # -----------------------------
    # Remove all flagged features
    # -----------------------------
    inactivity_reduced_feature_cols = [
        f for f in feature_cols
        if f not in inactivity_flagged_features
    ]

    if len(inactivity_reduced_feature_cols) > 0:

        full_inactivity_reduced_feature_set_result = evaluate_classifier_with_cv(
            classifier=classifier,
            data=data,
            cv_strategy=cv_strategy,
            scoring_metrics=scoring_metrics,
            cv_groups=cv_groups,
            feature_cols=inactivity_reduced_feature_cols,
            create_plots=create_plots,
            plots_dir= plots_base_path / "full_inactivity_reduced",
            classifier_evaluated=f"{classifier.__class__.__name__}_{feature_set_name}_inactivity_reduced",
            metadata={
                "analysis": "inactivity_dominance",
                "feature_set": feature_set_name,
                "feature_set_type": "inactivity_reduced",
                "removed_features": list(inactivity_flagged_features),
                "n_features": len(inactivity_reduced_feature_cols),
                "features": list(inactivity_reduced_feature_cols)
            }
        )

        all_feature_sets_cv_results.append(full_inactivity_reduced_feature_set_result)

    else:
        full_inactivity_reduced_feature_set_result = None
        print(f"Skipping inactivity-reduced model for {feature_set_name}: no features left.")

    if not all(isinstance(r, dict) and "scores" in r for r in all_feature_sets_cv_results):
        raise ValueError("Invalid results format detected in inactivity-dominance analysis.")


    return {
        "feature_set": feature_set_name,
        "all_feature_sets_cv_results": all_feature_sets_cv_results,
        "inactivity_related_features": inactivity_related_features,
        "inactivity_flagged_features": inactivity_flagged_features,
        "ablation_classifier_scores": all_ablation_results,
        "shuffled_y_classifier_scores": shuffled_Y_validation_check_results,
        "baseline_full_dataset_classifier_scores": baseline_full_dataset_scores,
        "full_inactivity_reduced_classifier_scores": full_inactivity_reduced_feature_set_result,
        "inactivity_reduced_features": inactivity_reduced_feature_cols
    }

def run_inactivity_sensitivity_analysis(
    data,
    classifier,
    cv_strategy,
    feature_sets,
    create_plots,
    plots_dir,
    cv_groups,
    scoring_metrics,
    random_state
):
    """
    Runs inactivity-dominance sensitivity analysis across the multiple feature sets.

    """

    if not isinstance(feature_sets, dict):
        raise TypeError(
            "inactivity_dominance_sensitivity_analysis expects feature_sets, "
            "not feature_cols."
        )

    all_feature_sets_cv_results = []
    all_inactivity_related_features = []
    all_ablation_cv_results = []
    all_shuffled_cv_results = []
    all_baseline_full_datase_cv_results = []

    for feature_set_name, feature_cols in feature_sets.items():

        print("\n" + "=" * 80)
        print(f"INACTIVITY-DOMINANCE ANALYSIS FOR FEATURE SET: {feature_set_name}")
        print("=" * 80)

        result = evaluate_inactivity_dependence_in_feature_sets(
            data=data,
            classifier=classifier,
            cv_strategy=cv_strategy,
            feature_set_name=feature_set_name,
            feature_cols=feature_cols,
            create_plots=create_plots,
            plots_dir=plots_dir,
            cv_groups=cv_groups,
            scoring_metrics=scoring_metrics,
            random_state=random_state
        )

        all_feature_sets_cv_results.extend(result["all_feature_sets_cv_results"])
        all_inactivity_related_features.append(result["inactivity_related_features"])
        all_ablation_cv_results.extend(result["ablation_classifier_scores"])
        all_shuffled_cv_results.append(result["shuffled_y_classifier_scores"])
        all_baseline_full_datase_cv_results.append(result["baseline_full_dataset_classifier_scores"])

    inactivity_related_features = pd.concat(
        all_inactivity_related_features,
        ignore_index=True
    )

    return (
        all_feature_sets_cv_results,
        inactivity_related_features,
        all_ablation_cv_results,
        all_shuffled_cv_results,
        all_baseline_full_datase_cv_results
    )

############################################################################################################################
# MODELS
############################################################################################################################

def evaluate_dummy_baseline(
    data,
    feature_cols,
    cv_strategy,
    cv_groups,
    scoring_metrics,
    create_plots,
    plots_dir,
    strategy="most_frequent",
    random_state=42,
):
    dummy = DummyClassifier(strategy=strategy, random_state=random_state)

    return evaluate_classifier_with_cv(
        data=data,
        feature_cols=feature_cols,
        classifier=dummy,
        cv_strategy=cv_strategy,
        cv_groups=cv_groups,
        scoring_metrics=scoring_metrics,
        create_plots=create_plots,
        plots_dir=plots_dir,
        classifier_evaluated=f"DummyClassifier_{strategy}",
        metadata={
            "model": "DummyClassifier",
            "feature_set": "baseline",
            "feature_set_type": "baseline",
            "n_features": len(feature_cols),
            "features": list(feature_cols),
            "strategy": strategy
        }
    )

def evaluate_classifier_with_cv(
    data,
    feature_cols,
    classifier,
    cv_strategy,
    cv_groups,
    plots_dir,
    scoring_metrics,
    feature_select=None,
    classifier_evaluated=None,
    metadata=None,
    scale=True,
    shuffled=False,
    return_train_score=True,
    create_plots=False,
    
):
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(data["group"])
    
    if shuffled:
        y = shuffle(y, random_state=42)
    
    valid_features = filter_existing_features(
        data=data,
        feature_cols=feature_cols,
        name=classifier_evaluated
    )

    X = data[valid_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Sanity check to ensure no leakage
    # check_leakage = X.copy()
    # save_dataframe_to_csv(check_leakage, path / "check_leakage.csv")
    
    pipeline_steps = []
    
    if scale:
        pipeline_steps.append(("scaler", StandardScaler()))

    if feature_select:
        pipeline_steps.append(("feature_select", clone(feature_select)))

    pipeline_steps.append(("model", clone(classifier)))

    pipeline = Pipeline(pipeline_steps)
    
    # cv = cv_strategy
    
    cross_validate_output = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_strategy,
        return_train_score=return_train_score,
        scoring=scoring_metrics,
        groups=cv_groups,
        error_score="raise"
    )
    
    cv_result = format_cv_result(result_name=classifier_evaluated, cross_validate_output=cross_validate_output, scoring_metrics=scoring_metrics, metadata=metadata)
    
    # Predictions for confusion matrix
    # y_pred = cross_val_predict(pipeline, X, y, cv=cv_strategy, groups=cv_groups)

    # if create_plots and plots_dir is not None:
    #     path = Path(plots_dir)
    #     path.mkdir(parents=True, exist_ok=True)

    #     safe_name = str(classifier_evaluated).replace(" ", "_").replace("/", "_")

    #     cm = confusion_matrix(y, y_pred)

    #     plt.figure(figsize=(5, 4))
    #     ax = sns.heatmap(
    #         cm,
    #         annot=True,
    #         fmt="d",
    #         cmap="Blues",
    #         xticklabels=le.classes_,
    #         yticklabels=le.classes_,
    #         square=True,
    #         cbar=False
    #     )

    #     ax.set_xlabel("Predicted")
    #     ax.set_ylabel("Actual")
    #     ax.set_title(f"Confusion Matrix: {classifier_evaluated}")

    #     plt.tight_layout()
    #     plt.savefig(
    #         path / f"{safe_name}_confusion_matrix.png",
    #         dpi=300,
    #         bbox_inches="tight"
    #     )
    #     plt.close()
    
    assert isinstance(cv_result, dict), "run_classifier must return dict"
    assert "scores" in cv_result, "missing scores"
    
    return cv_result

def evaluate_select_from_model(
    data,
    feature_cols,
    classifier,
    cv_strategy,
    cv_groups,
    scoring_metrics,
    plots_dir,
    create_plots,
    feature_select,
):
    sfm_result = evaluate_classifier_with_cv(
        data=data,
        feature_cols=feature_cols,
        classifier=classifier,
        cv_strategy=cv_strategy,
        cv_groups=cv_groups,
        scoring_metrics=scoring_metrics,
        plots_dir=plots_dir,
        create_plots=create_plots,
        feature_select=feature_select,
        classifier_evaluated=f"{classifier.__class__.__name__}_select_from_model",
        metadata={
            "feature_set": "select_from_model",
            "feature_set_type": "automatic",
            "selection_method": "SelectFromModel",
            "threshold": "median",
            # Using this so summary table/plots do not break
            "n_features": len(feature_cols),
            "features": "selected_inside_cv",
            "note": "SelectFromModel was fitted inside each CV fold"
        },
    )
    return sfm_result

############################################################################################################################
# FEATURES THROUGH CLUSTERS
############################################################################################################################
def compute_permutation_importance(
    data,
    classifier,
    feature_cols,
    cv_strategy,
    random_state,
    cv_groups,
    create_plots=False,
):
    """
    Compute cross-validated permutation feature importance.

    Importance is measured as the decrease in balanced accuracy
    when a feature is randomly permuted.

    The reported mean and standard deviation summarise permutation
    importance across cross-validation folds.
    """

    X, y = standardise_model_inputs(data,feature_cols)

    fold_permutation_importances = []

    if cv_groups is None:
        split_iterator = cv_strategy.split(X,y)
    else:
        split_iterator = cv_strategy.split(X,y,cv_groups)

    for train_idx, test_idx in split_iterator:

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", clone(classifier))
        ])

        pipeline.fit(X_train,y_train)

        permutation_result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=10,
            random_state=random_state,
            scoring="balanced_accuracy"
        )

        # Store the mean importance from this CV fold
        fold_permutation_importances.append(permutation_result.importances_mean)

    fold_permutation_importances = np.asarray(fold_permutation_importances)

    permutation_importance_summaries = pd.DataFrame({
        "feature": feature_cols,
        "importance": (fold_permutation_importances.mean(axis=0)),
        "importance_std": (fold_permutation_importances.std(axis=0, ddof=1))
    })

    permutation_importance_summaries = (
        permutation_importance_summaries
        .sort_values("importance",ascending=False)
        .reset_index(drop=True)
    )

    permutation_importance_reduced_features = (
        permutation_importance_summaries.loc[permutation_importance_summaries["importance"] > 0,"feature"].tolist()
        )

    return (
        permutation_importance_summaries,
        permutation_importance_reduced_features
    )

def cluster_correlated_features(
    data,
    feature_cols,
    create_plots,
    plots_dir,
    n_clusters=4
):
    X, _ = standardise_model_inputs(data, feature_cols)

    corr = X.corr()

    # Handles constant columns that create NaN correlations
    corr = corr.fillna(0)

    distance = 1 - np.abs(corr)

    condensed_distance = squareform(distance, checks=False)

    linkage_matrix = linkage(condensed_distance, method="average")

    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    cluster_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "cluster": cluster_labels
        }
    ).sort_values("cluster")

    print("\n=== FEATURE CLUSTERS IDENTIFIED ===")
    print(cluster_df)
    
    if create_plots:
        # Plot feature correlations
        plot_feature_correlation_clustermap(data=data,
                                            feature_cols=feature_cols, 
                                            plots_dir=plots_dir
                    )

    return cluster_df

def merge_feature_clusters_and_importance(
    cluster_df,
    importance_df,
    create_plots,
    plots_dir
):
    merged = cluster_df.merge(importance_df, on="feature")
    merged_df = merged.sort_values(["cluster", "importance"], ascending=[True, False])
    
    if create_plots:
        # Plot feature importance
        plot_permutation_importance(importance_df, plots_dir=plots_dir)
            
    return merged_df

def run_rfecv_feature_selection(
    data,
    classifier,
    cv_strategy,
    automatic_feature_candidates,
    create_plots,
    plots_dir,
    cv_groups=None,
):
    """Find minimal feature set through recursive feature elimination with cross-validation"""
    X, y = standardise_model_inputs(data, automatic_feature_candidates)

    rfecv = RFECV(
        estimator=clone(classifier),
        step=1,
        cv=cv_strategy,
        scoring="balanced_accuracy",
        min_features_to_select=1,
        n_jobs=-1
    )

    if cv_groups is None:
        rfecv.fit(X, y)
    else:
        rfecv.fit(X, y, groups=cv_groups)

    rfecv_performance_curve_df = build_rfecv_curve_dataframe(rfecv)

    rfecv_selected_features = np.array(automatic_feature_candidates)[rfecv.support_].tolist()

    rfecv_feature_rankings = pd.DataFrame(
        {
            "feature": automatic_feature_candidates,
            "selected": rfecv.support_,
            "ranking": rfecv.ranking_
        }
    ).sort_values("ranking")

    rfecv_selection_metadata = fromat_rfecv_selection_result(
        result_name=f"{classifier.__class__.__name__}_rfecv_selection",
        fitted_rfecv=rfecv,
        automatic_feature_candidates=automatic_feature_candidates,
        feature_ranking_df=rfecv_feature_rankings
    )

    print("\n=== MINIMAL FEATURE SET FROM RFECV ===")
    print("Selected features:", rfecv_selected_features)
    print("\nFeature rankings:")
    print(rfecv_feature_rankings)

    print(f"\nOptimal number of features is: {rfecv.n_features_}")

    if create_plots:
        plot_rfecv_curve(
            rfecv_curve_df=rfecv_performance_curve_df,
            selected_feature_count=rfecv.n_features_,
            plots_dir=plots_dir
        )
    return rfecv_selected_features, rfecv_feature_rankings, rfecv_selection_metadata, rfecv_performance_curve_df

def build_automatic_selection_feature_sets(automatic_feature_candidates, permutation_importance, rfecv_features):
    # Reduced = only positive importance
    positive_permutation_importance_features = permutation_importance[
        permutation_importance["importance"] > 0
    ]["feature"].tolist()

    return {
        "all_candidates": list(automatic_feature_candidates),
        "positive_permutation": list(positive_permutation_importance_features),
        "rfecv_selected": list(rfecv_features)
    }

def evaluate_feature_sets_with_cv(
    data,
    feature_sets,
    classifier,
    cv_strategy,
    cv_groups,
    scoring_metrics,
    plots_dir,
    create_plots
):
    feature_set_cv_results = []

    for feature_set_name, feature_names in feature_sets.items():
        print(f"\n=== Evaluating: {feature_set_name} ===")
        print(f"Number of features passed in: {len(feature_names)}")
        print(feature_names)

        res = evaluate_classifier_with_cv(
            data=data,
            feature_cols=feature_names,
            classifier=classifier,
            cv_strategy=cv_strategy,
            cv_groups=cv_groups,
            scoring_metrics=scoring_metrics,
            plots_dir= plots_dir / feature_set_name,
            create_plots=create_plots,
            classifier_evaluated=f"{classifier.__class__.__name__}_{feature_set_name}",
            metadata={
                "feature_set": feature_set_name,
                "n_features": len(feature_names),
                "features": feature_names
            }
        )
        feature_set_cv_results.append(res)

    return feature_set_cv_results

def evaluate_permutation_importance_stability(
    data,
    feature_cols,
    classifier,
    cv_strategy,
    cv_groups,
    random_state
):
    """
    Evaluate the stability of permutation feature importance
    across cross-validation folds.

    Mean importance represents the average decrease in balanced
    accuracy when a feature is permuted
    
    Standard deviation represents the between-fold variability
    in permutation importance
    """

    X, y = standardise_model_inputs(data, feature_cols)

    fold_importances = []

    if cv_groups is None:
        split_iterator = cv_strategy.split(X,y)
    else:
        split_iterator = cv_strategy.split(X, y, cv_groups)

    for train_idx, test_idx in split_iterator:

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", clone(classifier))
        ])

        pipeline.fit(X_train, y_train)

        permutation_result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=10,
            random_state=random_state,
            scoring="balanced_accuracy"
        )

        # Store the mean importance for each feature in this CV fold
        fold_importances.append(permutation_result.importances_mean)

    fold_importances = np.asarray(fold_importances)

    stability_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_importance": (fold_importances.mean(axis=0)),
        "std_importance": (fold_importances.std(axis=0, ddof=1))
    })

    stability_df = (
        stability_df
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )

    return stability_df

def compare_feature_selection_methods(
    importance_df,
    rfecv_features,
    zero_dominated_features
):
    """
    Compare overlap between:
    - permutation importance features
    - RFECV-selected features
    - zero-dominated/inactivity-related features

    Returns both Jaccard scores and the actual overlapping feature names.
    """
    # Features with a positive importance only
    importance_features = set(
        importance_df.loc[importance_df["importance"] > 0, "feature"]
    )

    rfecv_set = set(rfecv_features)

    zero_set = set(
        zero_dominated_features
        .loc[zero_dominated_features["is_inactivity_related"], "feature"]
    )

    def jaccard(a, b):
        return len(a & b) / len(a | b) if len(a | b) > 0 else 0

    importance_rfecv_overlap = importance_features & rfecv_set
    importance_zero_overlap = importance_features & zero_set
    rfecv_zero_overlap = rfecv_set & zero_set
    all_three_overlap = importance_features & rfecv_set & zero_set

    results = {
        # Jaccard scores
        "permutation_vs_rfecv": jaccard(importance_features, rfecv_set),
        "permutation_vs_inactivity": jaccard(importance_features, zero_set),
        "rfecv_vs_inactivity": jaccard(rfecv_set, zero_set),

        # Counts
        "permutation_rfecv_overlap_count": len(importance_rfecv_overlap),
        "permutation_inactivity_overlap_count": len(importance_zero_overlap),
        "rfecv_inactivity_overlap_count": len(rfecv_zero_overlap),
        "all_overlap_count": len(all_three_overlap),

        # Feature names
        "permutation_rfecv_overlap_features": sorted(importance_rfecv_overlap),
        "permutation_inactivity_overlap_features": sorted(importance_zero_overlap),
        "rfecv_inactivity_overlap_features": sorted(rfecv_zero_overlap),
        "all_overlap_features": sorted(all_three_overlap),

        # Optional metadata
        "n_importance_features": len(importance_features),
        "n_rfecv_features": len(rfecv_set),
        "n_zero_dominated_features": len(zero_set),
    }

    return results

def compare_final_feature_sets(
    results,
    primary_metric="balanced_accuracy"
):
    summary_df = summarise_cv_performance(results)
    summary_df = summary_df[summary_df["feature_set"] != "baseline"].copy()

    mean_col = f"{primary_metric}_mean"
    # std_col = f"{primary_metric}_std" # unused

    summary_df = summary_df.sort_values(mean_col, ascending=False)

    summary_df["rank"] = summary_df[mean_col].rank(
        ascending=False,
        method="min"
    ).astype(int)

    comparison_df = compare_feature_sets_across_cv_folds(
        results,
        scoring_metric=primary_metric
    )

    return summary_df, comparison_df

def add_performance_efficiency_metrics(performance_summary, scoring_metric="balanced_accuracy"):
    result = performance_summary.copy()

    mean_score_col = f"{scoring_metric}_mean"
    std_score_col = f"{scoring_metric}_std"

    result["performance_per_feature"] = result[mean_score_col] / result["n_features"]

    result["stability_adjusted_score"] = result[mean_score_col] - result[std_score_col]

    return result

############################################################################################################################
# FUNCTIONAL
############################################################################################################################

def save_dataframe_to_csv(
    data,
    path,
    custom_index=None,
    save_index=False,
    format_text=False,
    round_decimals=None
):
    """
    Safely saves DataFrame-like objects to CSV.

    Handles:
    - DataFrames
    - Series
    - dictionaries
    - lists
    - NumPy arrays

    Optional:
    - custom index
    - save/hide index
    - round numeric values
    - format text labels for presentation
    """

    if isinstance(data, pd.DataFrame):
        df = data.copy()

    elif isinstance(data, pd.Series):
        df = data.to_frame()

    else:
        df = pd.DataFrame(data)

    if custom_index is not None:
        df.index = custom_index

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if round_decimals is not None:
        df = df.map(
            lambda v: round(v, round_decimals)
            if isinstance(v, (float, np.float32, np.float64))
            else v
        )

    if format_text:
        df = df.map(
            lambda v: v.replace("_", " ")
            if isinstance(v, str)
            else v
        )

        df.columns = [
            str(col).replace("_", " ")
            for col in df.columns
        ]

    df.to_csv(path, index=save_index)

    return df

def remove_zero_mean_days(df, activity_col="activity", date_col="date"):
    """
    Remove days where the mean activity for that day is zero.
    """

    daily_mean = df.groupby(date_col)[activity_col].mean()

    non_zero_days = daily_mean[daily_mean > 0].index

    removed_days = daily_mean[daily_mean == 0].index

    if len(removed_days) > 0:
        print(f"Removed {len(removed_days)} zero-mean days: {list(removed_days)}")

    df = df[df[date_col].isin(non_zero_days)].copy()

    return df

def ensure_safe_divide(numerator, denominator, default=0):
    '''Prevent runtime warning when denominator is 0'''

    if denominator == 0 or np.isnan(denominator):
        return default

    return numerator / denominator

def summarise_preprocessing_removals(
    groups_config=None,
    min_samples_per_day=1440,
    save_path=None
):
    """
    Summarise how many rows/days were removed during preprocessing.

    Reports:
    - raw rows
    - duplicate timestamps removed
    - number of recorded days
    - number of retained complete days
    - number of removed incomplete or incorrect zero days
    - retained/removed rows after duplicate removal
    """
    if groups_config is None:
        groups_config = {
            "schizophrenia": 22,
            "control": 32
        }

    records = []

    for group_name, n_people in groups_config.items():
        for person_id in range(1, n_people + 1):

            file_path = Path(f"data/{group_name}/{group_name}_{person_id}.csv")
            df = pd.read_csv(file_path)

            raw_rows = len(df)

            df["timestamp"] = pd.to_datetime(df["timestamp"])

            duplicate_rows = df.duplicated(subset="timestamp").sum()

            df = df.drop_duplicates(subset="timestamp")

            rows_after_duplicates = len(df)

            df["date"] = df["timestamp"].dt.date

            day_counts = df.groupby("date").size()

            total_days = len(day_counts)

            retained_day_counts = day_counts[day_counts >= min_samples_per_day]
            removed_day_counts = day_counts[day_counts < min_samples_per_day]

            retained_days = len(retained_day_counts)
            removed_days = len(removed_day_counts)

            retained_rows = retained_day_counts.sum()
            removed_rows = removed_day_counts.sum()

            records.append({
                "group": group_name,
                "person_id": person_id,
                "raw_rows": raw_rows,
                "duplicate_rows_removed": duplicate_rows,
                "rows_after_duplicate_removal": rows_after_duplicates,
                "total_recorded_days": total_days,
                "retained_complete_days": retained_days,
                "removed_incomplete_days": removed_days,
                "retained_rows": retained_rows,
                "removed_rows_from_incomplete_days": removed_rows,
                "min_samples_per_day": min_samples_per_day
            })

    summary_df = pd.DataFrame(records)

    group_summary = (
        summary_df
        .groupby("group", as_index=False)
        .agg(
            participants=("person_id", "nunique"),
            raw_rows=("raw_rows", "sum"),
            duplicate_rows_removed=("duplicate_rows_removed", "sum"),
            rows_after_duplicate_removal=("rows_after_duplicate_removal", "sum"),
            total_recorded_days=("total_recorded_days", "sum"),
            retained_complete_days=("retained_complete_days", "sum"),
            removed_incomplete_days=("removed_incomplete_days", "sum"),
            retained_rows=("retained_rows", "sum"),
            removed_rows_from_incomplete_days=("removed_rows_from_incomplete_days", "sum")
        )
    )

    group_summary["pct_days_removed"] = (
        group_summary["removed_incomplete_days"]
        / group_summary["total_recorded_days"]
        * 100
    )

    group_summary["pct_rows_removed_after_duplicates"] = (
        group_summary["removed_rows_from_incomplete_days"]
        / group_summary["rows_after_duplicate_removal"]
        * 100
    )

    print("\n" + "=" * 80)
    print("PREPROCESSING REMOVAL SUMMARY BY GROUP")
    print("=" * 80)
    print(group_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("PREPROCESSING REMOVAL SUMMARY BY PARTICIPANT")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(
            save_path.with_name(save_path.stem + "_by_participant.csv"),
            index=False
        )

        group_summary.to_csv(
            save_path.with_name(save_path.stem + "_by_group.csv"),
            index=False
        )

    return summary_df, group_summary

def get_model_feature_columns(data):
    """
    Return the list of created numeric feature columns.

    Excludes metadata columns that should not be used as model features.
    """

    metadata_cols = {
        "group",
        "person_id",
        "date",
        "label",
        "Unnamed: 0"
    }

    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

    feature_cols = [
        col for col in numeric_cols
        if col not in metadata_cols
        and not col.startswith("Unnamed")
    ]

    return feature_cols

def filter_existing_features(data, feature_cols, name="feature set"):
    """
    Keep only features that exist in the dataframe.
    """

    feature_cols = list(feature_cols)

    valid_features = [
        f for f in feature_cols
        if f in data.columns
    ]

    missing = set(feature_cols) - set(valid_features)

    if missing:
        print(f"Warning: missing features in {name}: {missing}")

    if len(valid_features) == 0:
        raise ValueError(f"No valid features found for {name}")

    return valid_features

def get_candidate_features_for_automatic_feature_selection(feature_cols):
    """
    Return the feature columns used for automatic feature selection.

    Hourly features are excluded because they represent a 24-hour profile
    and are already evaluated separately as a manual feature set.
    """

    exclude_exact = {
        "min",
        "max",
        "median",
        "is_constant_activity"
    }

    automatic_features = [
        col for col in feature_cols
        if not col.startswith("mean_activity_")
        and col not in exclude_exact
    ]

    return automatic_features

def standardise_model_inputs(data, feature_cols):
    le = LabelEncoder()
    y = le.fit_transform(data["group"])

    X = data[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y

def format_cv_result(result_name, cross_validate_output, scoring_metrics, metadata=None):
    """
    Standardises cross_validate output into:
    {
        "name": str,
        "scores": {
            metric: np.array([...])  # per-fold scores
        }
    }
    """
    cv_scores = {}

    for metric in scoring_metrics:
        test_score_key = f"test_{metric}"

        if test_score_key not in cross_validate_output:
            raise ValueError(
                f"Missing '{test_score_key}' in cross_validate output. "
                f"Available keys: {list(cross_validate_output.keys())}"
            )

        fold_scores = np.asarray(cross_validate_output[test_score_key])

        if fold_scores.ndim != 1:
            raise ValueError(
                f"Expected 1D array for {test_score_key}, got shape {fold_scores.shape}"
            )

        cv_scores[metric] = fold_scores

    if len(cv_scores) == 0:
        raise ValueError(f"[make_result] No scores generated for {result_name}")

    return {
        "name": result_name,
        "scores": cv_scores,
        "metadata": metadata or {}
    }

def fromat_rfecv_selection_result(result_name, fitted_rfecv, automatic_feature_candidates, feature_ranking_df):
    """
    Stores RFECV feature-selection information in a standard result-like format.

    This is not the same as a normal classifier evaluation result.
    It should not be passed into metric comparison functions.
    """

    refcv_selected_features = np.array(automatic_feature_candidates)[fitted_rfecv.support_].tolist()

    selection_result = {
        "name": result_name,
        "type": "feature_selection",
        "scores": {},
        "metadata": {
            "method": "RFECV",
            "selected_features": refcv_selected_features,
            "n_selected_features": fitted_rfecv.n_features_,
            "feature_rankings": feature_ranking_df
        }
    }

    if hasattr(fitted_rfecv, "cv_results_"):
        selection_result["metadata"]["rfecv_cv_results"] = fitted_rfecv.cv_results_

    return selection_result

def build_rfecv_curve_dataframe(rfecv):
    return pd.DataFrame({
        "n_features": np.arange(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
        "mean_test_score": rfecv.cv_results_["mean_test_score"],
        "std_test_score": rfecv.cv_results_["std_test_score"]
    })

def result_value_to_dataframe(value):
    """
    Converts a result dictionary value into a DataFrame.

    This allows the combiner to handle:
    - DataFrames
    - Series
    - dictionaries
    - lists
    - NumPy arrays
    - scalar values
    """

    if value is None:
        return None

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, pd.Series):
        df = value.to_frame(name=value.name or "value")
        df = df.reset_index()
        return df

    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.json_normalize(value)

    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.DataFrame({"value": list(value)})

    return pd.DataFrame({"value": [value]})

def add_result_metadata(
    df,
    classifier_name,
    sample_type=None,
    classifier_col="classifier"
):
    """
    Adds classifier and sample type information to a result DataFrame.

    If the classifier column already exists, it is not duplicated.
    """

    df = df.copy()

    if sample_type is not None and "sample_type" not in df.columns:
        df.insert(0, "sample_type", sample_type)

    if classifier_col not in df.columns:
        insert_position = 1 if "sample_type" in df.columns else 0
        df.insert(insert_position, classifier_col, classifier_name)

    else:
        existing_classifiers = (
            df[classifier_col]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(existing_classifiers) > 0:
            existing_classifiers = set(existing_classifiers)

            if classifier_name not in existing_classifiers:
                print(
                    f"Warning: result already has classifier values "
                    f"{existing_classifiers}, but dictionary key is "
                    f"{classifier_name!r}."
                )

    return df

def combine_classifier_results_by_key(
    classifier_results,
    sample_type=None,
    skip_keys=None
):
    """
    Combines matching result entries across different classifier runs.

    Parameters
    ----------
    classifier_results : dict
        Dictionary where each key is a classifier name and each value is
        that classifier's returned result dictionary.
    sample_type : str, optional - "person" or "daily"
    skip_keys : set or list, optional - result keys that should not be combined or saved.
        """
    if skip_keys is None:
        skip_keys = {
            "data",
            "feature_cols",
            "pca_projection",
            "pca_loadings",
            "automatic_feature_sets",
        }

    skip_keys = set(skip_keys)

    all_result_keys = sorted(
        set().union(
            *(results.keys() for results in classifier_results.values())
        )
    )

    combined_results = {}

    for result_name in all_result_keys:

        if result_name in skip_keys:
            continue

        frames = []

        for classifier_name, results in classifier_results.items():

            if result_name not in results:
                continue

            value = results[result_name]

            df = result_value_to_dataframe(value)

            if df is None or df.empty:
                continue

            df = add_result_metadata(
                df=df,
                classifier_name=classifier_name,
                sample_type=sample_type
            )

            frames.append(df)

        if frames:
            combined_results[result_name] = pd.concat(
                frames,
                ignore_index=True,
                sort=False
            )

    return combined_results

def save_combined_results_by_key(
    combined_results,
    output_dir,
    save_index=False,
    format_text=False,
    round=3
):
    """
    Saves each combined result DataFrame as a separate CSV file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for result_name, df in combined_results.items():

        if not isinstance(df, pd.DataFrame):
            continue

        if df.empty:
            continue

        save_path = output_dir / f"{result_name}.csv"

        save_dataframe_to_csv(
            data=df,
            path=save_path,
            save_index=save_index,
            format_text=format_text,
            round_decimals=round
        )

def combine_and_save_classifier_results(
    classifier_results,
    sample_type,
    output_dir,
    skip_keys=None,
    save_index=False,
    format_text=False,
    round=3
):
    """
    Combines matching result entries across classifiers and saves each
    combined result as a separate CSV file.
    """

    combined_results = combine_classifier_results_by_key(
        classifier_results=classifier_results,
        sample_type=sample_type,
        skip_keys=skip_keys
    )

    save_combined_results_by_key(
        combined_results=combined_results,
        output_dir=output_dir,
        save_index=save_index,
        format_text=format_text,
        round=round
    )

    return combined_results

############################################################################################################################
# MATHEMATICAL/COMPARISON FUNCTIONS
############################################################################################################################

def summarise_cv_performance(results):
    """
    Converts one result dict or a list of result dicts into a summary DataFrame.
    """

    # Allow a single result dict to be passed in
    if isinstance(results, dict):
        results = [results]

    rows = []

    for result in results:
        row = {
            "name": result["name"],
            "feature_set": result["metadata"].get("feature_set"),
            "feature_set_type": result["metadata"].get("feature_set_type"),
            "n_features": result["metadata"].get("n_features")
        }

        for metric, values in result["scores"].items():
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"] = np.std(values)

        rows.append(row)

    return pd.DataFrame(rows)

def rank_feature_sets_results_by_metric(summary_df, primary_metric="balanced_accuracy_mean"):
    ranked = summary_df.copy()

    ranked["rank"] = ranked[primary_metric].rank(
        ascending=False,
        method="min"
    ).astype(int)

    return ranked.sort_values("rank")

def compare_feature_sets_across_cv_folds(results, scoring_metric="balanced_accuracy"):
    """
    Pairwise comparison of feature-set results using per-fold CV scores.
    """

    if isinstance(results, dict):
        results = [results]

    comparisons = []

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            r1 = results[i]
            r2 = results[j]

            scores1 = np.asarray(r1["scores"][scoring_metric])
            scores2 = np.asarray(r2["scores"][scoring_metric])

            if len(scores1) != len(scores2):
                raise ValueError(
                    f"Cannot compare {r1['name']} and {r2['name']} because "
                    f"they have different numbers of folds."
                )

            diff = scores1 - scores2

            comparisons.append({
                "result_1": r1["name"],
                "result_2": r2["name"],
                "feature_set_1": r1["metadata"].get("feature_set"),
                "feature_set_2": r2["metadata"].get("feature_set"),
                "scoring_metric": scoring_metric,
                "mean_1": scores1.mean(),
                "mean_2": scores2.mean(),
                "mean_difference": diff.mean(),
                "std_difference": diff.std(),
                "result_1_better_folds": int((diff > 0).sum()),
                "result_2_better_folds": int((diff < 0).sum()),
                "tied_folds": int((diff == 0).sum())
            })

    return pd.DataFrame(comparisons)

def get_best_manual_and_automatic_sets(
    summary_df,
    scoring_metric="balanced_accuracy_mean",
    exclude_manual=("all",),
    exclude_automatic=("full",)
):
    """
    Finds the best manual and automatic feature sets.

    Excludes:
    - manual 'all', because it is not a specific manually designed feature family
    - automatic 'full', because it is not actually feature selection
    """

    required = ["feature_set", "feature_set_type", scoring_metric]

    for col in required:
        if col not in summary_df.columns:
            raise ValueError(
                f"Missing column '{col}'. Available columns: {list(summary_df.columns)}"
            )

    manual_df = summary_df[
        summary_df["feature_set_type"].eq("manual")
        & ~summary_df["feature_set"].isin(exclude_manual)
    ]

    automatic_df = summary_df[
        summary_df["feature_set_type"].eq("automatic")
        & ~summary_df["feature_set"].isin(exclude_automatic)
    ]

    if manual_df.empty:
        raise ValueError(
            "No manual feature sets left after exclusions. "
            "Check your manual feature sets or change exclude_manual."
        )

    if automatic_df.empty:
        raise ValueError(
            "No automatic feature sets left after exclusions. "
            "Check your automatic feature sets or change exclude_automatic."
        )

    best_manual = manual_df.sort_values(scoring_metric, ascending=False).iloc[0]
    best_automatic = automatic_df.sort_values(scoring_metric, ascending=False).iloc[0]

    return pd.DataFrame([best_manual, best_automatic])


if __name__ == '__main__':
    main()
