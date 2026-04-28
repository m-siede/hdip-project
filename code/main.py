from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    cross_validate, cross_val_score, cross_val_predict,
    StratifiedKFold, StratifiedGroupKFold, LeaveOneOut,
    KFold, GroupKFold, LeaveOneGroupOut, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
from sklearn.utils import shuffle
import datetime

def main():
    
    feature_cols = [
        "mean", "sd", "inactive_ratio",
        "day_activity", "night_activity",
        "day_night_ratio", "IS", "IV", "RA",
        "longest_inactivity"
    ]

    
    # Classification parameters
    rf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=5,
            min_samples_leaf=5,
            random_state=42)
    
    lr = LogisticRegression
    
    group_k_fold = GroupKFold(n_splits=5)
    logo = LeaveOneGroupOut()
    
    feature_select = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold="median")
    
    # daily_samples_analysis(classifier=rf, feature_cols=feature_cols, validation=group_k_fold, feature_select=False, plots=True)
    
    person_samples_analysis(classifier=rf, feature_cols=feature_cols, validation=group_k_fold, feature_select=False, plots=True)
    
    

def daily_samples_analysis(classifier, feature_cols, validation, feature_select, plots):
    ## Create a dataset where each day per person is one sample
    daily_samples = Path(f"data/outputs/daily/combined.csv")
    daily_data = Path(f"data/outputs/daily/{classifier.__class__.__name__}/")
    daily_plots = Path(f"/plots/daily")
    
    if daily_samples.exists():
        daily_samples = pd.read_csv(daily_samples)
    else:
        daily_samples = build_daily_dataset()
    
    groups=daily_samples["person_id"]
    data=daily_samples.copy()
    
        
    # Check all generated features for zero values, NaNs etc
    features_issues_report = audit_features(data=data)
    save_to_csv(features_issues_report, f"{daily_data}")
    
    
    
    ##### Inactivity ######
    
    # Visualise the inactive vs active ratio per group
    plot_inactive_ratios(data=data,path=daily_plots)
    
    ## Check which features are dominated or highly correlated with no activity of subjects and Testing if classifiers overly rely on inactive ratio
    # zero_dominated_features = detect_zero_dominated_features(data=data, feature_cols=None, zero_threshold=0.8, corr_threshold=0.7, plots=False)
    # X = combined_statistics[["inactive_ratio"]]
    # y = LabelEncoder().fit_transform(combined_statistics["group"])
    # scores = cross_val_score(LogisticRegression(), X, y, cv=5)
    # print(scores.mean())
    
    # Check if model relies on zero dominated features to make predictions
    zero_dominance_sensitivity_results, zero_dominated_features = zero_dominance_sensitivity_analysis(
        data=data, 
        groups=groups,
        classifier=classifier,
        validation=validation, 
        feature_cols=feature_cols,
        plots=plots,
        path=daily_plots)
    save_to_csv(zero_dominance_sensitivity_results, f"{daily_data}/zero_analysis/zero_dominance_sensitivity.csv")
    save_to_csv(zero_dominated_features, f"{daily_data}/zero_analysis/zero_dominated_features.csv")
    
    
    # Determine which features are the most important for the model, plot them and their associated clusters
    feature_importance = compute_feature_importance(
        data=data, 
        classifier=classifier, 
        feature_cols=feature_cols,
        plots=plots,
        path=daily_plots)
    save_to_csv(feature_importance, f"{daily_data}/importance/feature_importance.csv")
    
    # Extract the found feature clusters
    clusters = extract_feature_clusters(
        data=data, 
        feature_cols=feature_cols, 
        n_clusters=4,
        )
    save_to_csv(clusters, f"{daily_data}/importance/feature_clusters.csv")
    
    # Combine the feature clusters, determine the feature importance and plot
    merged = combine_clusters_and_importance(clusters, feature_importance, plots=plots, path=daily_plots)
    save_to_csv(merged, f"{daily_data}/importance/clusters_merged.csv")
    
    # Find the smallest necessary feature set through Recursive Feature Elimination with Cross-Validation
    selected_features, results_minimal_features = run_rfecv(
    data=data,
    classifier=classifier,
    validation=validation,
    groups=groups,
    path=daily_plots,
    plots=plots,
    feature_cols=feature_cols)
    save_to_csv(selected_features, f"{daily_data}/rfecv/selected_features.csv")
    save_to_csv(results_minimal_features, f"{daily_data}/rfecv/results_minimal_features.csv")
    
    # Compare the performance of the model using the full vs the minimal dataset
    scores_all_features, scores_minimum_features = compare_rfecv_features_vs_full_set(
        data=data, 
        classifier=classifier, 
        validation=validation, 
        groups=groups
        )
    save_to_csv(scores_all_features, f"{daily_data}/rfecv/scores_comparison_all_features.csv")
    save_to_csv(scores_minimum_features, f"{daily_data}/rfecv/scores_comparison_minimum_features.csv")
    
    
    ##### Check full feature set
    check_leakage, classifier_results, cross_val_accuracy_scores = run_classifier(
        data=data, 
        feature_cols=feature_cols,
        classifier=classifier, 
        validation=validation, 
        feature_select = feature_select,
        groups=groups,
        feature_sets=None,
        name=None,
        path=daily_plots
    )
    save_to_csv(check_leakage, f"{daily_data}/classifier/leakage_check.csv")
    save_to_csv(classifier_results, f"{daily_data}/classifier/classifier_results.csv")
    save_to_csv(cross_val_accuracy_scores, f"{daily_data}/classifier/cross_val_accuracy_scores.csv")
    
    #### Comparing Feature Sets
    # feature_sets = get_feature_sets(data=data)

    # feature_sets_comparison_results = {}

    # for name, features in feature_sets.items():
    #     if len(features) == 0:
    #         continue
    #     score = run_classifier(
    #     data=data,
    #     classifier=classifier, 
    #     validation=validation, 
    #     feature_select=False,
    #     groups=groups, 
    #     features, 
    #     name
    #     path=daily_plots)
    #     feature_sets_comparison_results[name] = score

    # save_to_csv(feature_sets_comparison_results, f"{daily_data}/feature_sets/feature_sets_comparison_results.csv")
    # print("\n=== FINAL COMPARISON ===")
    # for k, v in feature_sets_comparison_results.items():
    #     print(f"{k}: {v:.3f}")        
    #     run_classifier(data=data,
        # classifier=classifier, 
    #     validation=validation, 
    #     feature_select=False,
    #     groups=groups,
    #     path=daily_plots)
    
    # plt.bar(feature_sets_comparison_results.keys(), feature_sets_comparison_results.values())
    # plt.ylabel("Accuracy")
    # plt.title("Feature Set Comparison")
    # plt.savefig(f"{daily_plots}/feature_comparison.png")
    # plt.close()

def person_samples_analysis(classifier, feature_cols, validation, feature_select, plots):
    ###############################################
    ## Procedure for each person being one sample
    
    # Process CSV files
    combined_csv = Path(f"data/outputs/person/combined.csv")
    person_samples = Path(f"data/outputs/person/{classifier.__class__.__name__}/")
    person_plots = Path(f"plots/person/")
    
    if combined_csv.exists():
        combined_statistics = pd.read_csv(combined_csv)
        print("Read csv")
    else:
        combined_statistics = create_combined_dataset(combined_csv)
        print("create csv")
    
    
    groups=combined_statistics["person_id"]
    data=combined_statistics.copy()
    
    ## Check all generated features for zero values, NaNs etc
    features_issues_report = audit_features(data=data)
    save_to_csv(features_issues_report, f"{person_samples}/preprocessing/feature_audit.csv")
    
    plot_inactive_ratios(data=data,path=person_plots)
    
    plot_clustered_importance(merged, path=person_plots)
    
    ## Check which features are dominated or highly correlated with no activity of subjects and Testing if classifiers overly rely on inactive ratio
    # zero_dominated_features = detect_zero_dominated_features(data=data, feature_cols=None, zero_threshold=0.8, corr_threshold=0.7, plots=False)
    # X = combined_statistics[["inactive_ratio"]]
    # y = LabelEncoder().fit_transform(combined_statistics["group"])
    # scores = cross_val_score(LogisticRegression(), X, y, cv=5)
    # print(scores.mean())

    ## Check which features are dominated or highly correlated with no activity of subjects
    # Check if model relies on zero dominated features to make predictions
    zero_dominance_sensitivity_results, zero_dominated_features = zero_dominance_sensitivity_analysis(
        data=data, 
        groups=groups,
        classifier=classifier,
        validation=validation, 
        feature_cols=feature_cols,
        plots=plots,
        path=person_plots)
    save_to_csv(zero_dominance_sensitivity_results, f"{person_samples}/zero_analysis/zero_dominance_sensitivity.csv")
    save_to_csv(zero_dominated_features, f"{person_samples}/zero_analysis/zero_dominated_features.csv")
    
    # Determine which features are the most important for the model, plot them and their associated clusters
    reduced_features, feature_importance = compute_feature_importance(
        data=data, 
        classifier=classifier, 
        feature_cols=feature_cols,
        plots=plots,
        path=person_plots)
    save_to_csv(feature_importance, f"{person_samples}/importance/feature_importance.csv")
    
    # Extract the found feature clusters
    clusters = extract_feature_clusters(
        data=data, 
        feature_cols=feature_cols, 
        n_clusters=4,
        )
    save_to_csv(clusters, f"{person_samples}/importance/feature_clusters.csv")
    
    # Combine the feature clusters, determine the feature importance and plot
    merged = combine_clusters_and_importance(clusters, feature_importance, plots=plots, path=person_plots)
    save_to_csv(merged, f"{person_samples}/importance/clusters_merged.csv")
    
    # Find the smallest necessary feature set through Recursive Feature Elimination with Cross-Validation
    selected_features, results_minimal_features = run_rfecv(
    data=data,
    classifier=classifier,
    validation=validation,
    groups=groups,
    path=person_plots,
    plots=plots,
    feature_cols=feature_cols)
    save_to_csv(selected_features, f"{person_samples}/rfecv/selected_features.csv")
    save_to_csv(results_minimal_features, f"{person_samples}/rfecv/results_minimal_features.csv")
    
    # Compare the performance of the model using the full vs the minimal dataset
    scores_all_features, scores_minimum_features = compare_rfecv_features_vs_full_set(
        data=data, 
        classifier=classifier, 
        validation=validation, 
        groups=groups
        )
    save_to_csv(scores_all_features, f"{person_samples}/rfecv/scores_comparison_all_features.csv")
    save_to_csv(scores_minimum_features, f"{person_samples}/rfecv/scores_comparison_minimum_features.csv")
    
    
    ##### Check full feature set
    check_leakage, classifier_results, cross_val_accuracy_scores = run_classifier(
        data=data, 
        feature_cols=feature_cols,
        classifier=classifier, 
        validation=validation, 
        feature_select = feature_select,
        groups=groups,
        feature_sets=None,
        name=None,
        path=person_plots
    )
    save_to_csv(check_leakage, f"{person_samples}/classifier/leakage_check.csv")
    save_to_csv(classifier_results, f"{person_samples}/classifier/classifier_results.csv")
    save_to_csv(cross_val_accuracy_scores, f"{person_samples}/classifier/cross_val_accuracy_scores.csv")
    
    #### Comparing Feature Sets
    # feature_sets = get_feature_sets(data=combined_statistics)

    # feature_sets_comparison_results = {}

    # for name, features in feature_sets.items():
    #     if len(features) == 0:
    #         continue
    #     score = run_classifier(
    #     data=data,
    #     classifier=classifier, 
    #     validation=validation, 
    #     feature_select=False,
    #     groups=groups, 
    #     feature_sets=features, 
    #     name=name,
    #     path=person_plots)
    #     feature_sets_comparison_results[name] = score

    # save_to_csv(feature_sets_comparison_results, f"{person_samples}/feature_sets/feature_sets_comparison_results.csv")
    # print("\n=== FINAL COMPARISON ===")
    # for k, v in feature_sets_comparison_results.items():
    #     print(f"{k}: {v:.3f}")        
    #     run_classifier(data=data,
        # classifier=classifier, 
    #     validation=validation, 
    #     feature_select=False,
    #     groups=groups,
    #     path=person_plots)
    
    # plt.bar(feature_sets_comparison_results.keys(), feature_sets_comparison_results.values())
    # plt.ylabel("Accuracy")
    # plt.title("Feature Set Comparison")
    # plt.savefig(f"{person_plots}/feature_comparison.png")
    # plt.close()
    
    

############################################################################################################################
# BUILD DATASETS
############################################################################################################################
# Sample per Person Dataset
############################################################################################################################
def create_combined_dataset(path):
    schizophrenia_statistics, schizophrenia_data = read_group_files('schizophrenia', 22)
    control_statistics, control_data = read_group_files('control', 32)

    # Group data and combine into one dataframe
    schizophrenia_statistics['group'] = 'schizophrenia'
    control_statistics['group'] = 'control'
    combined_groups_statistics = pd.concat([schizophrenia_statistics, control_statistics], ignore_index=True)
    combined_groups_statistics.index.name = 'person_id'
        
    combined_groups_statistics = save_to_csv(combined_groups_statistics, path)
    
    return combined_groups_statistics

def read_group_files(group_name: str, num_files: int):
    """Read CSV files for a specified group (patients/control/etc.) and evaluate."""
    statistics_csv = Path(f"data/outputs/{group_name}_statistics.csv")
    all_data_csv = Path(f"data/outputs/{group_name}_data.csv")

    # Read CSV file if it exists, else import patient data and statistics and combine into one file
    if statistics_csv.exists() and all_data_csv.exists():
        combined_patients_statistics = pd.read_csv(statistics_csv)
        combined_patients_data = pd.read_csv(all_data_csv)
    else:
        all_patients_stats = []
        all_patients_data = []
        
        for person_id in range(1, num_files + 1):
            print(person_id)
            plot= person_id<5
            patient_statistics, patient_data = read_patient_file(person_id, group_name, plot=plot)
            
            features = extract_features(patient_data)
            patient_statistics.update(features)

            for key, value in patient_statistics["activity_by_hour"].items():
                patient_statistics[f"mean_activity_{key}"] = value
            del patient_statistics["activity_by_hour"]

            patient_data["person_no"] = person_id

            all_patients_stats.append(patient_statistics)
            all_patients_data.append(patient_data)
    
        # Save as CSV
        combined_patients_statistics = save_to_csv(all_patients_stats, statistics_csv, index=range(1, num_files + 1))
        
        all_patients_activity_data = pd.concat(all_patients_data, ignore_index=True)
        combined_patients_data = save_to_csv(all_patients_activity_data, all_data_csv)

    return combined_patients_statistics, combined_patients_data

def read_patient_file(patient_num: int, group_name: str, plot=True):
    """Read data for each patient, calculate their individual statistics and create plots"""
    patient_data = pd.read_csv(f"data/{group_name}/{group_name}_{patient_num}.csv")
    
    # create new var for time and date
    patient_data["timestamp"] = pd.to_datetime(patient_data["timestamp"])
    patient_data = patient_data.drop_duplicates(subset="timestamp")
    patient_data["date"] = patient_data["timestamp"].dt.date
    patient_data["hour"] = patient_data["timestamp"].dt.hour.astype(int)
    
    # Remove incomplete days
    full_days = patient_data.groupby("date").size()
    full_days = full_days[full_days >= 1400].index
    patient_data = patient_data[patient_data["date"].isin(full_days)]
    
    patient_statistics = extract_statistics(patient_data)
    return patient_statistics, patient_data

############################################################################################################################
# Sample per Day Dataset
############################################################################################################################

def build_daily_dataset():
    '''Build a dataset that calculates features for each day that will be used as samples'''
    
    schizophrenia = process_group_for_daily("schizophrenia", 22)
    control = process_group_for_daily("control", 32)

    df = pd.concat([schizophrenia, control], ignore_index=True)
    return df

def process_group_for_daily(group_name, n_people):
    '''
    '''
    
    rows = []
    
    for person_id in range(1, n_people + 1):
        df = load_patient_for_daily(group_name, person_id)

        # group by day
        for date, day_df in df.groupby("date"):
            if len(day_df) < 1440:
                continue

            statistics = extract_statistics(day_df)
            features = extract_features(day_df)

            row = {**statistics, **features}
            row["person_id"] = person_id
            row["date"] = date
            row["group"] = group_name

            rows.append(row)

    return pd.DataFrame(rows)

def load_patient_for_daily(group_name, person_id):
    df = pd.read_csv(f"data/{group_name}/{group_name}_{person_id}.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp")

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour.astype(int)

    return df

############################################################################################################################
# EXPLORATORY
############################################################################################################################
# Extract features and statistics
############################################################################################################################
def extract_statistics(data):
    statistics = {
        "min" : data["activity"].min(),
        "max" : data["activity"].max(),
        "mean" : data["activity"].mean(),
        "median" : data["activity"].median(),
        "sd" : data["activity"].std(),
        "activity_by_hour" : data[["activity", "hour"]].groupby(by="hour").mean()["activity"].to_dict()
    }
        
    return statistics

def extract_features(data):
    df = data.copy()
    
    # create day and night
    day = df[(df["hour"] >= 8) & (df["hour"] < 20)]
    night = df[(df["hour"] < 8) | (df["hour"] >= 20)]
    
    # Day and night mean activity
    day_mean = day["activity"].mean()
    night_mean = night["activity"].mean()

    # Percentage of inactivity per sample
    inactive_ratio = (df["activity"] == 0).mean()

    # Longest inactivity streak
    is_zero = (df["activity"] == 0).astype(int)
    streaks = is_zero.groupby((is_zero != is_zero.shift()).cumsum()).cumsum()
    longest_streak = streaks.max()

    # Mean activity and variability per day
    daily_mean = df.groupby("date")["activity"].mean()
    
    interday_variability = None
    
    if len(daily_mean) < 1:
        interday_variability = daily_mean.var()
        
    
    # Circadian metrics
    IS = compute_IS(df)
    IV = compute_IV(df)
    RA = compute_RA(df)

    # Base features
    features = {
        "inactive_ratio": inactive_ratio,
        "longest_inactivity": longest_streak,
        "is_constant_activity": int(df["activity"].var() == 0),
        "day_activity": day_mean,
        "night_activity": night_mean,
        "day_night_ratio": safe_divide(day_mean, night_mean),
        "intraday_variability": df["activity"].diff().abs().mean(),
        "IS": IS,
        "IV": IV,
        "RA": RA,
    }
    
    if interday_variability is not None:
        features["interday_variability"] = interday_variability
        # Check for constant activity per person
        if df["activity"].var() == 0:
            print(f"Warning: constant activity detected for person {df.get('person_id', 'unknown')}")

    # Create 6-hour time bins
    df["time_bin"] = pd.cut(
        df["hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        right=False
    )
    # 6-hour bin features 
    bin_means = df.groupby("time_bin")["activity"].mean().to_dict()
    # Ensure all bins exist
    for key in ["night", "morning", "afternoon", "evening"]:
        features[f"activity_{key}"] = bin_means.get(key, 0)
    # Inactivity per bin
    bin_inactive = df.groupby("time_bin")["activity"].apply(lambda x: (x == 0).mean()).to_dict()
    for key in ["night", "morning", "afternoon", "evening"]:
        features[f"inactive_{key}"] = bin_inactive.get(key, 0)

    return features

def compute_IS(data):
    hourly_mean = data.groupby("hour")["activity"].mean()
    overall_mean = data["activity"].mean()

    num = ((hourly_mean - overall_mean) ** 2).sum()
    den = ((data["activity"] - overall_mean) ** 2).sum()

    return safe_divide(num, den)

def compute_IV(data):
    diff = data["activity"].diff().dropna()
    num = (diff ** 2).mean()
    den = data["activity"].var()

    return safe_divide(num, den)

def compute_RA(data):
    hourly = data.groupby("hour")["activity"].mean()
    M10 = hourly.sort_values(ascending=False).head(10).mean()
    L5 = hourly.sort_values().head(5).mean()
    return (M10 - L5) / (M10 + L5 + 1e-5)

############################################################################################################################
# Audit created features
############################################################################################################################
def audit_features(data, feature_cols=None, verbose=True):

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
        n_zero = (series == 0).sum()

        variance = series.var()
        std = series.std()

        report.append({
            "feature": col,
            "nan_count": n_nan,
            "nan_pct": n_nan / n_total,
            "inf_count": n_inf,
            "zero_count": n_zero,
            "zero_pct": n_zero / n_total,
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

        if row["zero_pct"] > 0.95:
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

def get_feature_sets(data):
    all_cols = data.select_dtypes(include=["number"]).columns

    # Hourly features
    hourly_features = [col for col in all_cols if "mean_activity_" in col]

    # 6-hour bin features
    bin_features = [col for col in all_cols if "activity_" in col or "inactive_" in col]
    bin_features = [col for col in bin_features if col not in hourly_features]

    # Circadian features
    circadian_features = ["IS", "IV", "RA"]
    
    # Base features
    base_features = ["mean", "sd", "inactive_ratio", "day_activity", "night_activity"]

    feature_sets = {
        "hourly": hourly_features,
        "bins": bin_features,
        "circadian": circadian_features,
        "base": base_features,
        "all": list(all_cols)
    }
    return feature_sets

def detect_zero_dominated_features(
    data,
    path,
    feature_cols=None, 
    zero_threshold=0.8, 
    corr_threshold=0.7, 
    plots=False 
    ):
    
    if feature_cols is None:
        X = data.select_dtypes(include=["number"]).copy()
    else:
        X = data[feature_cols].copy()

    results = []

    # Reference zero feature
    if "inactive_ratio" in X.columns:
        zero_ref = X["inactive_ratio"]
    else:
        zero_ref = None

    for col in X.columns:
        series = X[col]

        zero_pct = (series == 0).mean()

        # Correlation with inactivity
        if zero_ref is not None and col != "inactive_ratio":
            corr = series.corr(zero_ref)
        else:
            corr = np.nan

        results.append({
            "feature": col,
            "zero_pct": zero_pct*100,
            "corr_with_inactive_ratio": corr,
            "is_zero_dominated": (
                zero_pct > zero_threshold or 
                (not np.isnan(corr) and abs(corr) > corr_threshold)
            )
        })

    df_results = pd.DataFrame(results).sort_values(by="zero_pct", ascending=False)


    print("\n=== ZERO-DOMINATED FEATURE ANALYSIS ===")
    print(df_results)

    print("\n=== FLAGGED FEATURES ===")
    print(df_results[df_results["is_zero_dominated"]])
    
    if plots is True:
        plot_zero_dominance(df_results, path=path)
        plot_zero_correlation(data=data, path=path, feature_cols=feature_cols)
    
    return df_results

def zero_dominance_sensitivity_analysis(
    data, 
    classifier, 
    validation, 
    feature_cols,
    plots,
    path,
    groups, 
    thresholds=[0.7, 0.8, 0.9]
    ):

    results = []

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(data["group"])

    # Clean features
    X_full = data[feature_cols].copy()
    X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0)

    for threshold in thresholds:
        print(f"\n=== Threshold: {threshold} ===")

        # Detect zero-dominated features
        zero_df = detect_zero_dominated_features(
            data, path, feature_cols,
            zero_threshold=threshold,
            corr_threshold=0.7,
            plots=plots
        )
    

        # drop columns based on zeros
        flagged = zero_df[zero_df["is_zero_dominated"]]["feature"].tolist()

        print(f"Removed features: {flagged}")

        # Remove them
        X = X_full.drop(columns=[f for f in flagged if f in X_full.columns])

        ## Further test, drop both inactivity related columns
        # drop_cols = ["inactive_ratio", "longest_inactivity"]

        # X = X_full.drop(columns=drop_cols)
        
        if X.shape[1] == 0:
            print("No features left, skipping...")
            continue

        metrics = [
        "accuracy", "balanced_accuracy",
        "average_precision", "f1", "precision", "recall",
        ]
        
        # Evaluate
        scores = cross_validate(classifier, X, y, cv=validation, scoring=metrics, groups=groups)
        print(scores)

        results.append({
            "threshold": threshold,
            "n_features": X.shape[1],
            "accuracy_mean": np.mean(scores['test_accuracy']),
            "accuracy_std": np.std(scores['test_accuracy']),
            "balanced_accuracy" : scores['test_balanced_accuracy'],
            "f1": scores['test_f1'],
            # "removed_features": ", ".join(flagged)
        })

        print(f"Accuracy: {np.mean(scores['test_accuracy']):.3f} ± {np.std(scores['test_accuracy']):.3f}")

    results_df = pd.DataFrame(results)

    print("\n=== FINAL COMPARISON ===")
    print(results_df)
    
    return results_df, zero_df

############################################################################################################################
# MODELS
############################################################################################################################
def run_classifier(
    data, 
    feature_cols,
    classifier, 
    validation,
    groups,
    path, 
    feature_select=None,
    feature_sets=None,
    name=None
    ):
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(data["group"])
    
    if feature_sets is not None:
        # Compare feature_sets
        valid_features = [f for f in feature_sets if f in data.columns]

        missing = set(feature_sets) - set(valid_features)
        if missing:
            print(f"Warning: missing features {missing}")

        X = data[valid_features].copy()
        X = X .drop(columns=["label", "person_id"], errors="ignore")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
    else:
        # Select features
        X = data.select_dtypes(include=["number"]).copy()

        # Drop anything that shouldn't be a feature
        X = X.drop(columns=["label", "person_id"], errors="ignore")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    
    # Sanity check to ensure no leakage
    check_leakage = X.copy()
    
    if feature_select is not None:
        pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("feature_select", feature_select),
        ("model", classifier)
    ])
    else:
        pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", classifier)
    ])

    cv = validation
    
    # Metrics used for scoring
    metrics = [
        "accuracy", "balanced_accuracy",
        "average_precision", "f1", "precision", "recall",
    ]
    
    classifier_results = cross_validate(
        pipeline, X, y, cv=cv, return_train_score=True, scoring=metrics, groups=groups
    )
    
    for metric in metrics:
        print(f"Train {metric}: {classifier_results['train_' + metric].mean():.3f}")
        print(f"Test {metric}: {classifier_results['test_' + metric].mean():.3f}")
    
    # Cross-validation accuracy scores
    cross_val_accuracy_scores = cross_val_score(pipeline, X, y, cv=cv, groups=groups)
    print(f"\nCross-validation accuracy: {np.mean(cross_val_accuracy_scores):.3f} ± {np.std(cross_val_accuracy_scores):.3f}")

    # Predictions for confusion matrix
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, groups=groups)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    # Heatmap for CM
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"{path}/confusion_matrix.png")
    plt.close()
    
    return check_leakage, classifier_results, cross_val_accuracy_scores

############################################################################################################################
# FEATURES THROUGH CLUSTERS
############################################################################################################################
def compute_feature_importance(
    data, 
    classifier, 
    feature_cols,
    path,
    plots=False,
    ):
    
    # Prepare data
    le = LabelEncoder()
    y = le.fit_transform(data["group"])
    X = data[feature_cols].copy()

    # Clean
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = classifier.fit(X_train, y_train)

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy"
    )

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": result.importances_mean
    }).sort_values(by="importance", ascending=False)
    
    importances = result.importances_mean
    
    # Keep only useful features
    reduced_features = [
        f for f, imp in zip(feature_cols, importances)
        if imp > 0
    ]

    print("\n=== FEATURE IMPORTANCE ===")
    print(importance_df)
    
    if plots is True:
        plot_feature_importance(importance_df, path=path)
        plot_feature_clustering(data=data, feature_cols=feature_cols, path=path)

    return importance_df, reduced_features

def extract_feature_clusters(
    data,
    feature_cols,
    n_clusters=3):
    
    X = data[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    corr = X.corr()
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

    print("\n=== FEATURE CLUSTERS ===")
    print(cluster_df)

    return cluster_df

def combine_clusters_and_importance(
    cluster_df, 
    importance_df, 
    plots, 
    path):
    
    merged = cluster_df.merge(importance_df, on="feature")
    merged_df = merged.sort_values(["cluster", "importance"], ascending=[True, False])
    
    if plots is True:
        plot_clustered_importance(merged_df, path=path)

def run_rfecv(
    data, 
    classifier, 
    validation, 
    feature_cols,
    plots,
    path, 
    groups=None 
    ):
    
    '''Find minimal feature set through recursive feature elimination with cross-validation'''
    
    # Labels
    le = LabelEncoder()
    y = le.fit_transform(data["group"])


    X = data[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)


    rfecv = RFECV(
        estimator=classifier,
        step=1,
        cv=validation,
        scoring="accuracy",
        min_features_to_select=1,
        n_jobs=-1
    )

    rfecv.fit(X, y, groups=groups)

    RFECV_features = np.array(feature_cols)[rfecv.support_]

    results_df = pd.DataFrame(
        {
        "feature": feature_cols,
        "selected": rfecv.support_,
        "ranking": rfecv.ranking_
    }
        ).sort_values("ranking")

    print("\n=== MINIMAL FEATURE SET ===")
    print("Selected features:", list(RFECV_features))
    print("\nFeature rankings:")
    print(results_df)

    print(f"\nOptimal number of features: {rfecv.n_features_}")
    
    if plots is True:
        plot_rfecv_curve(rfecv, path=path)

    return RFECV_features, results_df

def compare_feature_sets(
    data,
    feature_cols,
    reduced_features,
    RFECV_features,
    classifier,
    validation,
    groups,
    plots,
    path
):

    results = []
    
    results.append(run_classifier(
        data=data, 
        feature_cols=feature_cols, 
        classifier=classifier,
        validation=validation, 
        groups=groups))
    
    

############################################################################################################################
# FUNCTIONAL
############################################################################################################################
def save_to_csv(data, path, index=None):
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if index is not None:
            df.index = index
    else:
        df = pd.DataFrame(data, index=index)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

    return df

def safe_divide(numerator, denominator, default=0):
    '''Prevent runtime warning when denominator is 0'''
    
    if denominator == 0 or np.isnan(denominator):
        return default
    
    return numerator / denominator

def calculate_scores(y_true, y_pred, prefix=''):
    return {
        prefix + 'accuracy': accuracy_score(y_true, y_pred),
        prefix + 'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        prefix + 'average_precision': average_precision_score(y_true, y_pred),
        prefix + 'f1_score': f1_score(y_true, y_pred),
        prefix + 'precision': precision_score(y_true, y_pred),
        prefix + 'recall_score': recall_score(y_true, y_pred),
    }
############################################################################################################################
# PLOTS
############################################################################################################################
def plot_inactive_ratios(data, path):

    sns.set_theme(style="whitegrid", context="paper")

    # 1. Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data, x="group", y="inactive_ratio")
    sns.stripplot(data=data, x="group", y="inactive_ratio", 
                color="black", alpha=0.3, jitter=0.2)
    plt.title("Inactive Ratio by Group")
    plt.ylabel("Percentage of Zero Activity Per Time")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(f"{path}/zero_analysis/inactive_ratio_boxplot.png", dpi=300)
    plt.close()

    # 2. Density plot 
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=data, x="inactive_ratio", hue="group", fill=True, alpha=0.4)
    plt.title("Distribution of Inactive Ratio per Group")
    plt.xlabel("Inactive Ratio")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{path}/zero_analysis/inactive_ratio_density.png", dpi=300)
    plt.close()

    #3. Per-subject
    subject_df = data.groupby(["person_id", "group"])["inactive_ratio"].mean().reset_index()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=subject_df, x="group", y="inactive_ratio")
    plt.title("Average Inactive Ratio per Subject")
    plt.tight_layout()
    plt.savefig(f"{path}/zero_analysis/inactive_ratio_per_subject.png", dpi=300)
    plt.close()
def plot_zero_dominance(data, path):

    sns.set_theme(style="whitegrid", context="paper")

    plt.figure(figsize=(7, 5))
    sns.barplot(data=data, x="zero_pct", y="feature")

    plt.axvline(0.8, linestyle="--")  # threshold line

    plt.title("Zero Percentage per Feature")
    plt.xlabel("Percentage of Zeros")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig(f"{path}/zero_analysis/zero_dominance.png", dpi=300)
    plt.close()
def plot_zero_correlation(data, feature_cols, path):
    
    X = data[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    corr = X.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0)

    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{path}/zero_analysis/correlation_matrix.png", dpi=300)
    plt.close()
def plot_feature_importance(importance_df, path):
    sns.set_theme(style="whitegrid", context="paper")

    plt.figure(figsize=(7, 5))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Permutation Feature Importance")
    plt.xlabel("Importance (Decrease in Accuracy)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{path}/feature_importance/feature_importance.png", dpi=300)
    plt.close()
def plot_feature_clustering(data, feature_cols, path):

    sns.set_theme(style="white", context="paper")

    # Prepare data
    X = data[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Compute correlation matrix
    corr = X.corr()

    # Create clustered heatmap
    g = sns.clustermap(
        corr,
        method="ward",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        figsize=(8, 8)
    )

    plt.title("Feature Correlation Clustering", pad=20)

    plt.savefig(f"{path}/feature_importance/feature_clustering.png", dpi=300)
    plt.close()
def plot_clustered_importance(merged_df, path):
    
    sns.set_theme(style="whitegrid", context="paper")

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=merged_df,
        x="importance",
        y="feature",
        hue="cluster",
        dodge=False
    )

    plt.title("Feature Importance by Cluster")
    plt.xlabel("Importance (Permutation)")
    plt.ylabel("Feature")

    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"{path}/feature_importance/clustered_feature_importance.png", dpi=300)
    plt.close()
def plot_rfecv_curve(rfecv, path):

    plt.figure()

    plt.plot(
        range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
        rfecv.cv_results_["mean_test_score"]
    )

    plt.xlabel("Number of Features")
    plt.ylabel("Cross-Validation Accuracy")
    plt.title("RFECV Feature Selection Curve")

    plt.tight_layout()
    plt.savefig(f"{path}/feature_importance/rfecv_curve.png", dpi=300)
    plt.close()
############################################################################################################################

if __name__ == '__main__':
    main()
