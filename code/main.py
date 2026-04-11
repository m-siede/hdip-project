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

    # Box plots of mean, max and SD of activity for groups
    for col in ['mean', 'max', 'sd']:
        plots(
            combined_groups_statistics, x='group', y=col, kind='box', show=False, legend=False,
            output_path=Path(f'plots/summary/box_ALL_{col}.png'), 
            figsize=(10, 8),
            title=f"Comparison of {col} activity between groups"
        )
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

            for key, value in patient_statistics["activity_by_hour"].items():
                patient_statistics[f"mean_activity_{key}"] = value
            del patient_statistics["activity_by_hour"]

            patient_data["person_no"] = person_id

            all_patients_stats.append(patient_statistics)
            all_patients_data.append(patient_data)
    
        # Save as CSV
        combined_patients_statistics = save_to_csv(all_patients_stats, statistics_csv, index=range(1, num_files + 1))
        

        all_patients_activity_data = pd.concat(all_patients_data, ignore_index=True)
        
        combined_patients_data = pd.DataFrame(all_patients_activity_data)
        all_data_csv.parent.mkdir(exist_ok=True, parents=True)
        all_patients_activity_data.to_csv(all_data_csv)

        # Create Subdirectory for summary plots
        # plots_subdir = Path('plots/summary/')

        # # Bar plot of mean activity per hour per person in group
        # combined_patients_statistics['person_no'] = combined_patients_statistics.index
        # plots(
        #     combined_patients_statistics, x='person_no', y='mean', kind='bar', show=False, legend=False,
        #     title= f"Bar chart of mean activity per person in the {group_name} group",
        #     output_path=plots_subdir / f"bar_{group_name}_mean_activity.png"
        # )

        # # Box plot of activity per person in group
        # plots(
        #     all_patients_activity_data, x='person_no', y="activity", kind='box', legend=False,
        #     show=False,title= f"Boxplot of activity per person in the {group_name} group", 
        #     output_path=plots_subdir / f"box_{group_name}_activity.png"
        # )
    
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
    
    patient_statistics = extract_features(patient_data)


    return patient_statistics, patient_data

def extract_features(data):
    statistics = {
        "min" : data["activity"].min(),
        "max" : data["activity"].max(),
        "mean" : data["activity"].mean(),
        "median" : data["activity"].median(),
        "sd" : data["activity"].std(),
        "activity_by_hour" : data[["activity", "hour"]].groupby(by="hour").mean()["activity"].to_dict()
    }
    return statistics

def patient_plots(data):
    plot_dir = Path(f"plots/{group_name}/{patient_num}")
    show_plots = False
    y = "activity"
    if plot:
        plots(data, x="timestamp", y=y, show=show_plots, output_path=plot_dir / "line_timestamp_vs_activity.png")
        plots(data, x="timestamp", y=y, kind="scatter", show=show_plots, output_path=plot_dir / "scatter_timestamp_vs_activity.png")
        plots(data, x="hour", y=y, kind="box", show=show_plots, output_path=plot_dir / "box_hour_vs_activity.png")
        plots(data, x="hour", y=y, kind="box", show=show_plots, output_path=plot_dir / "box_hour_vs_activity_short.png")
        plots(data, x="hour", y=y, kind="box", show=show_plots, yscale="log", output_path=plot_dir / "box_hour_vs_log_activity.png")

def group_plots():
    # Create Subdirectory for summary plots
        plots_subdir = Path('plots/summary/')

        # Bar plot of mean activity per hour per person in group
        combined_patients_statistics['person_no'] = combined_patients_statistics.index
        plots(
            combined_patients_statistics, x='person_no', y='mean', kind='bar', show=False, legend=False,
            title= f"Bar chart of mean activity per person in the {group_name} group",
            output_path=plots_subdir / f"bar_{group_name}_mean_activity.png"
        )

        # Box plot of activity per person in group
        plots(
            all_patients_activity_data, x='person_no', y="activity", kind='box', legend=False,
            show=False,title= f"Boxplot of activity per person in the {group_name} group", 
            output_path=plots_subdir / f"box_{group_name}_activity.png"
        )

def combined_plots():
    for col in ['mean', 'max', 'sd']:
        plots(
            combined_groups_statistics, x='group', y=col, kind='box', show=False, legend=False,
            output_path=Path(f'plots/summary/box_ALL_{col}.png'), 
            figsize=(10, 8),
            title=f"Comparison of {col} activity between groups"
        )

def plots(
    df, x, y, kind="line", hue=None, figsize=(12,6), show=False, 
    output_path=None, yscale="linear", ylimits=None, legend=False, title=None
):
    """Create plots."""
    if not show and output_path and Path(output_path).exists():
        return

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", context="talk",  palette="colorblind", rc=custom_params)

    # df = df.copy()
    fig, ax = plt.subplots(figsize=figsize)

    if kind == "box":
        sns.boxplot(
            data=df, x=x, y=y, ax=ax, hue=x, width=0.6, showfliers=False,
            showmeans=True,  meanprops={
                                        "marker": "o",
                                        "markerfacecolor": "black",
                                        "markeredgecolor": "white",
                                        "markersize": 6
                                        },
            legend=legend
        )
        # ax.legend(handles=[], labels=[])

    elif kind == "line":
        sns.lineplot(data=df, x=x, y=y, hue=x, marker="o", linewidth=2.5,ax=ax, legend=legend)

    elif kind == "scatter":
        sns.scatterplot(data=df, x=x, y=y, hue=x, alpha=0.7, s=60, ax=ax,  legend=legend)

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

    # if title:
    #     title = f"{kind} plot of {x} and {y}"
    plt.title(title)
    # transform y to different kind e.g. linear or log scale
    plt.yscale(yscale)

    # eliminate whitespace
    # plt.tight_layout()

    # save graph to file
    if output_path:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def save_to_csv(data, path, index):
    df = pd.DataFrame(data).copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    
    return df


if __name__ == '__main__':
    main()
