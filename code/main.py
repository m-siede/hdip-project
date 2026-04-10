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
        schizophrenia_statistics, schizophrenia_data = read_group_files('schizophrenia', 22)
        control_statistics, control_data = read_group_files('control', 32)

        # Group data and combine into one dataframe
        schizophrenia_statistics['group'] = 'schizophrenia'
        control_statistics['group'] = 'control'
        combined_statistics = pd.concat([schizophrenia_statistics, control_statistics], ignore_index=True)
        # combined_statistics = combined_statistics.set_index('Unnamed: 0', drop=True)
        combined_statistics.index.name = 'person_id'
        
        combined_csv.parent.mkdir(exist_ok=True, parents=True)
        combined_statistics.to_csv(combined_csv)

    # Box plots of mean, max and SD of activity for groups
    for col in ['mean', 'max', 'sd']:
        plots(
            combined_statistics, x='group', y=col, kind='box', show=False, legend=False,
            output_path=Path(f'plots/summary/box_ALL_{col}.png'), 
            figsize=(10, 8),
            title=f"Comparison of {col} activity between groups"
        )


def read_group_files(group_name: str, num_files: int) -> pd.DataFrame:
    """Read CSV files for a specified group (patients/control/etc.) and evaluate."""
    statistics_csv = Path(f"data/outputs/{group_name}_statistics.csv")
    all_data_csv = Path(f"data/outputs/{group_name}_data.csv")

    # Read CSV file if it exists
    if statistics_csv.exists() and all_data_csv.exists():
        all_statistics = pd.read_csv(statistics_csv)
        all_data = pd.read_csv(all_data_csv)
    else:
        all_stats = []
        all_dataframes = []
        for i in range(1, num_files + 1):
            print(i)
            plot= i<5
            statistics, patient_data = read_patient_file(i, group_name, plot=plot)

            for key, value in statistics["activity_by_hour"].items():
                statistics[f"mean_activity_{key}"] = value
            del statistics["activity_by_hour"]

            patient_data["person_no"] = i

            all_stats.append(statistics)
            all_dataframes.append(patient_data)
    
        # Save as CSV
        all_statistics = pd.DataFrame(all_stats, index=range(1, num_files + 1))
        statistics_csv.parent.mkdir(exist_ok=True, parents=True)
        all_statistics.to_csv(statistics_csv)

        all_activity = pd.concat(all_dataframes, ignore_index=True)
        
        all_data = pd.DataFrame(all_activity)
        all_data_csv.parent.mkdir(exist_ok=True, parents=True)
        all_activity.to_csv(all_data_csv)

        # Create Subdirectory for summary plots
        plots_subdir = Path('plots/summary/')

        # Bar plot of mean activity per hour per person in group
        all_statistics['person_no'] = all_statistics.index
        plots(
            all_statistics, x='person_no', y='mean', kind='bar', show=False, legend=False,
            title= f"Bar chart of mean activity per person in the {group_name} group",
            output_path=plots_subdir / f"bar_{group_name}_mean_activity.png"
        )

        # Box plot of activity per person in group
        plots(
            all_activity, x='person_no', y="activity", kind='box', legend=False,
            show=False,title= f"Boxplot of activity per person in the {group_name} group", 
            output_path=plots_subdir / f"box_{group_name}_activity.png"
        )
    
    return all_statistics, all_data


def read_patient_file(patient_num: int, group_name: str, plot=True):
    """Read data for each patient, calculate their individual statistics and create plots"""
    patient_data = pd.read_csv(f"data/{group_name}/{group_name}_{patient_num}.csv")
    patient_data["timestamp"] = pd.to_datetime(patient_data["timestamp"])

    # create new column called hour
    patient_data["hour"] = patient_data["timestamp"].dt.hour.astype(int)
    
    ''' if len(hour) per day less than 1440, delete day
    '''

    statistics = {
        "min" : patient_data["activity"].min(),
        "max" : patient_data["activity"].max(),
        "mean" : patient_data["activity"].mean(),
        "median" : patient_data["activity"].median(),
        "sd" : patient_data["activity"].std(),
        "variance" : patient_data["activity"].var(),
        "activity_by_hour" : patient_data[["activity", "hour"]].groupby(by="hour").mean()["activity"].to_dict()
    }
    
    plot_dir = Path(f"plots/{group_name}/{patient_num}")
    show_plots = False

    y = "activity"
    if plot:
        plots(patient_data, x="timestamp", y=y, show=show_plots, output_path=plot_dir / "line_timestamp_vs_activity.png")
        plots(patient_data, x="timestamp", y=y, kind="scatter", show=show_plots, output_path=plot_dir / "scatter_timestamp_vs_activity.png")
        plots(patient_data, x="hour", y=y, kind="box", show=show_plots, output_path=plot_dir / "box_hour_vs_activity.png")
        plots(patient_data, x="hour", y=y, kind="box", show=show_plots, output_path=plot_dir / "box_hour_vs_activity_short.png")
        plots(patient_data, x="hour", y=y, kind="box", show=show_plots, yscale="log", output_path=plot_dir / "box_hour_vs_log_activity.png")

    return statistics, patient_data


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

# get minutes per day
# check for full days
# check for how many days per full days


if __name__ == '__main__':
    main()
