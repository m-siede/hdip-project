from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def main():
    # Process CSV files
    combined_csv = Path(f"data/outputs/combined.csv")

    if combined_csv.exists():
        combined_statistics = pd.read_csv(combined_csv)
    else:
        combined_statistics = create_combined_dataset(combined_csv)


def create_combined_dataset(combined_csv):
    schizophrenia_statistics, schizophrenia_data = read_group_files('schizophrenia', 22)
    control_statistics, control_data = read_group_files('control', 32)

    # Group data and combine into one dataframe
    schizophrenia_statistics['group'] = 'schizophrenia'
    control_statistics['group'] = 'control'
    combined_groups_statistics = pd.concat([schizophrenia_statistics, control_statistics], ignore_index=True)
    # combined_statistics = combined_statistics.set_index('Unnamed: 0', drop=True)
    combined_groups_statistics.index.name = 'person_id'
        
    combined_groups_statistics = save_to_csv(combined_groups_statistics, combined_csv)
    
    return combined_groups_statistics

def read_group_files(group_name: str, num_files: int) -> pd.DataFrame:
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

    day = df[(df["hour"] >= 8) & (df["hour"] < 20)]
    night = df[(df["hour"] < 8) | (df["hour"] >= 20)]

    inactive_ratio = (df["activity"] == 0).mean()

    # Longest inactivity streak
    is_zero = (df["activity"] == 0).astype(int)
    streaks = is_zero.groupby((is_zero != is_zero.shift()).cumsum()).cumsum()
    longest_streak = streaks.max()

    # Variability
    daily_mean = df.groupby("date")["activity"].mean()

    # Circadian metrics
    IS = compute_IS(df)
    IV = compute_IV(df)
    RA = compute_RA(df)

    features = {
        "inactive_ratio": inactive_ratio,
        "longest_inactivity": longest_streak,
        "day_activity": day["activity"].mean(),
        "night_activity": night["activity"].mean(),
        "day_night_ratio": day["activity"].mean() / (night["activity"].mean() + 1e-5),
        "interday_variability": daily_mean.var(),
        "intraday_variability": df["activity"].diff().abs().mean(),

        "IS": IS,
        "IV": IV,
        "RA": RA,
    }

    return features

def compute_IS(df):
    hourly_mean = df.groupby("hour")["activity"].mean()
    overall_mean = df["activity"].mean()
    return ((hourly_mean - overall_mean) ** 2).sum() / ((df["activity"] - overall_mean) ** 2).sum()

def compute_IV(df):
    diff = df["activity"].diff().dropna()
    return (diff ** 2).mean() / df["activity"].var()

def compute_RA(df):
    hourly = df.groupby("hour")["activity"].mean()
    M10 = hourly.sort_values(ascending=False).head(10).mean()
    L5 = hourly.sort_values().head(5).mean()
    return (M10 - L5) / (M10 + L5 + 1e-5)



def save_to_csv(data, path, index=None):
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if index is not None:
            df.index = index
    else:
        df = pd.DataFrame(data, index=index)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

    return df


if __name__ == '__main__':
    main()
