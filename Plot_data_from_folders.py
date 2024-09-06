"""
    Plot data from ACA/ACA-GP algorithm tested for clouds of points.

    Author: Vladislav A. Yastrebov
    Affiliation: CNRS, MINES Paris, PSL University, Evry/Paris, France
    Date: Sep 2024
    License: CC0
"""

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import json
from collections import defaultdict

# Specify the folder name
# folder_name = "FullTestCycle_gamma_linear/"  # Change this to the folder you want to process
# folder_name = "TimeChecks10/"  # Change this to the folder you want to process
folder_name = sys.argv[1]+"/"
adhoc = True

if adhoc:
    prefix = "Adhoc"
else:
    prefix = "NoAdhoc"

# Update the directory paths
plt.rcParams.update({
            "font.size": 10,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "lmodern",
            "pgf.rcfonts": False,  # Use the LaTeX preamble settings from the document
            "pgf.preamble": (
                r"\usepackage{amsmath}"
                r"\usepackage{amssymb}"
                r"\usepackage{lmodern}"  # Ensure Latin Modern is used
            )
        })

Data = []

all_files_list = os.path.join(folder_name, "files.list")

# Check if the folder exists first
if not os.path.exists(folder_name):
    raise FileNotFoundError(f"Folder '{folder_name}' does not exist!")

# Construct list of files if it does not exist or if empty
if not os.path.exists(all_files_list) or os.path.getsize(all_files_list) == 0:
    with open(all_files_list, "w") as f:  # No need for f-string here
        for filename in os.listdir(folder_name):  # List files in folder_name
            if filename.endswith(".json"):
                f.write(f"{filename}\n")                

if os.path.exists(all_files_list):
    # Read the list of files
    with open(all_files_list, "r") as f:
        for l in f.readlines():
            filename = l.strip()
            json_data = json.load(open(os.path.join(folder_name, filename)))  # Use folder_name
            Data.append([filename, json_data])


# Collect data by distance for grouping
data_by_dist = defaultdict(list)

# Group the data by 'dist' for xi values within the same 'dist' group
for data in Data:
    fname = data[0]
    json_data = data[1]

    # Filter out data where 'rank3treatment' == False
    if adhoc :
        if not json_data['rank3treatment']:
            continue
    else:
        if json_data['rank3treatment']:
            continue

    # Load the corresponding npz file
    npz_path = os.path.join(folder_name, fname.replace("json", "npz"))  # Use folder_name
    num_data = np.load(npz_path)
    
    # Extract the distance and group by it
    dist = json_data['target_distance']
    data_by_dist[dist].append((fname, json_data, num_data))

##########################
#  Plot relative errors  #
##########################
 
for dist, data_group in data_by_dist.items():
    
    # Initialize figure with 1 row and 3 columns for xi = [0.25, 0.5, 1]
    fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharey=True)  # 1 row, 3 columns, sharing y-axis
    xi_values = [0.25, 0.5, 1]  # Desired xi values to plot
    xi_to_index = {0.25: 0, 0.5: 1, 1.0: 2}  # Mapping xi values to subplot index

    for fname, json_data, num_data in data_group:
        xi = json_data['xi']
        
        if xi not in xi_values:
            continue  # Only process data for the defined xi values

        ax = axes[xi_to_index[xi]]  # Select the correct axis based on the xi value

        ACA = num_data["ACA"]
        ACA_GP = num_data["ACA_GP"]
        # if SVD is available, use it, otherwise use ACA
        if "SVD" in num_data:
            SVD = num_data["SVD"]

        ACA_log = np.log10(ACA)
        ACA_GP_log = np.log10(ACA_GP)
        if "SVD" in num_data:
            SVD_log = np.log10(SVD)

        mean_ACA_log = np.mean(ACA_log, axis=0)
        mean_ACA_GP_log = np.mean(ACA_GP_log, axis=0)
        if "SVD" in num_data:
            mean_SVD_log = np.mean(SVD_log, axis=0)

        std_ACA_log = np.std(ACA_log, axis=0)
        std_ACA_GP_log = np.std(ACA_GP_log, axis=0)
        if "SVD" in num_data:
            std_SVD_log = np.std(SVD_log, axis=0)

        ranks = np.arange(1, ACA.shape[1] + 1)

        # Plot the data
        ax.fill_between(ranks, 10**(mean_ACA_log + std_ACA_log), 10**(mean_ACA_log - std_ACA_log), color="r", alpha=0.2)
        ax.plot(ranks, 10**mean_ACA_log, "v-", markersize=5, color="r", markeredgewidth=0.5, markeredgecolor='k', label="ACA+")

        ax.fill_between(ranks, 10**(mean_ACA_GP_log + std_ACA_GP_log), 10**(mean_ACA_GP_log - std_ACA_GP_log), color="g", alpha=0.2)
        ax.plot(ranks, 10**mean_ACA_GP_log, "*-", markersize=8, markeredgewidth=0.5, color="g", markeredgecolor='k', label="ACA-GP", zorder=3)

        if "SVD" in num_data:
            ax.fill_between(ranks, 10**(mean_SVD_log + std_SVD_log), 10**(mean_SVD_log - std_SVD_log), color="b", alpha=0.2)
            ax.plot(ranks, 10**mean_SVD_log, "o-", markersize=5, markeredgewidth=0.5, color="b", markeredgecolor='k', label="SVD")

        ax.set_title(r"$\xi = {0:.2f}$".format(xi), fontsize=10)
        ax.set_xlabel("Approximation rank, $k$")
        
        if xi == 0.25:
            ax.set_ylabel(r"Relative error, $\|A_k - A\|_F \; / \; \|A\|_F$")
        else:
            ax.tick_params(labelleft=False)  # Hide y-axis labels for the other subplots

        ax.grid(True)
        ax.set_yscale("log")
        ax.set_xlim(1, np.max(ranks))
        if abs(dist - 5.) < 0.1:
            ax.set_ylim(1e-13, 0.01)
            yticks = np.logspace(-13, -2, 12)
            ax.set_yticks(yticks)
        else:
            ax.set_ylim(1e-10, 0.1)
            yticks = np.logspace(-10, -1, 10)
            ax.set_yticks(yticks)

        if xi == 1.0:
            ax.legend(loc='best', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle(f"Distance = ${dist:.1f} \\pm 0.1$", fontsize=10, y=0.95, x=0.1)

    # Save the figure in the provided folder
    unique_id = fname.split("_ID_")[1].split(".")[0]
    figure_name = os.path.join(folder_name, f"ACA_GP_error_{prefix}_Dist_{dist:.1f}")
    fig.savefig(figure_name + ".pdf", bbox_inches='tight')
    fig.savefig(figure_name + ".pgf", bbox_inches='tight')

    plt.close()



#######################
# Plot accuracy gain  #
#######################

    fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharey=True)  # 1 row, 3 columns, sharing y-axis
    plt.subplots_adjust(wspace=-0.1)  # Reduce the width space between subplots

    # Create empty plots first, then fill them with data corresponding to each xi
    for fname, json_data, num_data in data_group:
        xi = json_data['xi']
        
        if xi not in xi_values:
            continue  # Only process data for the defined xi values

        ax = axes[xi_to_index[xi]]  # Select the correct axis based on the xi value

        ACA = num_data["ACA"]
        ACA_GP = num_data["ACA_GP"]

        accuracy_gain = ACA / ACA_GP
        accuracy_gain_log = np.log10(accuracy_gain)
        accuracy_gain_mean_log = np.mean(accuracy_gain_log, axis=0)
        accuracy_gain_std_log = np.std(accuracy_gain_log, axis=0)

        ranks = np.arange(1, ACA.shape[1] + 1)
        average_over_whole_ranks_log = np.mean(accuracy_gain_log)

        ax.fill_between(ranks, 10**(accuracy_gain_mean_log + accuracy_gain_std_log), 10**(accuracy_gain_mean_log - accuracy_gain_std_log), color="g", alpha=0.2)
        ax.plot(ranks, 10**accuracy_gain_mean_log, "o-", markersize=6, markeredgewidth=0.5, color="g", markeredgecolor='k', label="ACA-GP", zorder=3)
        ax.axhline(y=10**average_over_whole_ranks_log, color='k', linestyle='--', label=r'Mean $g$, $k \le 15$', zorder=4)

        # Plot average gain over first 5 ranks
        average_over_first_5_ranks_log = np.mean(accuracy_gain_log[:, :5], axis=1)
        average_over_first_5_ranks_log_mean = np.mean(average_over_first_5_ranks_log)
        x = np.linspace(1, 5.2, 2)
        y = 10**average_over_first_5_ranks_log_mean * np.ones(x.shape[0])
        ax.plot(x, y, linestyle="-",color="r", label=r"Mean $g$, $k \le 5$", zorder=5)


        ax.set_title(r"$\xi = {0:.2f}$".format(xi), fontsize=10)
        ax.set_xlabel("Approximation rank, $k$")
        
        if xi == 0.25:
            ax.set_ylabel(r"Accuracy gain, $g = \frac{\mathrm{Er}_{\mathrm{ACA}} }{ \mathrm{Er}_{\mathrm{ACA-GP}}}$")
        else:
            ax.tick_params(labelleft=False)

        ax.grid(True)
        ax.set_xlim(1, np.max(ranks))
        ax.set_ylim(0., 5)

        if xi == 1.0:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle(f"Distance = ${dist:.1f} \\pm 0.1$", fontsize=10, y=0.95,x=0.1)

    # Save the accuracy gain figure
    figure_name = os.path.join(folder_name, f"ACA_GP_accuracy_{prefix}_Dist_{dist:.1f}")
    fig.savefig(figure_name + ".pdf", bbox_inches='tight')
    fig.savefig(figure_name + ".pgf", bbox_inches='tight')

    plt.close()


#######################
#  Print LaTeX table  #
#######################

latex_tables_by_dist = {}

for dist, data_group in data_by_dist.items():
    latex_table = []
    
    latex_table.append(r"\begin{tabular}{ c  p{2.cm} p{2.cm} p{2.cm} p{2.5cm}}")
    latex_table.append(r"    Rank  & \multicolumn{3}{l}{\hspace{1em}Relative error, $\|A'_k - A\|_F/\|A\|_F$} & Average gain \\")
    latex_table.append(r"    $k$ & ACA & ACA-GP & SVD & ACA-GP/ACA \\")
    latex_table.append(r"    \hline")
    
    for fname, json_data, num_data in data_group:
        xi = json_data['xi']
        
        if xi not in xi_values:
            continue

        latex_table.append(rf"\multicolumn{{5}}{{l}}{{$\xi = {xi:.2f} \vphantom{{A^{{A^A}}}}$}} \\")
        
        ACA = num_data["ACA"]
        ACA_GP = num_data["ACA_GP"]
        if "SVD" in num_data:
            SVD = num_data["SVD"]

        accuracy_gain = ACA / ACA_GP
        accuracy_gain_mean = np.mean(accuracy_gain, axis=0)

        ACA_mean = np.mean(ACA, axis=0)
        ACA_GP_mean = np.mean(ACA_GP, axis=0)
        if "SVD" in num_data:
            SVD_mean = np.mean(SVD, axis=0)

        ranks = np.arange(1, ACA.shape[1] + 1)

        for k in ranks:
            ACA_GP_value = rf"\textbf{{{ACA_GP_mean[k-1]:.2e}}}" if k in [1, 2] else f"{ACA_GP_mean[k-1]:.2e}"
            if "SVD" in num_data:
                SVD_value = rf"\textbf{{{SVD_mean[k-1]:.2e}}}" if k in [1, 2] else f"{SVD_mean[k-1]:.2e}"

            if "SVD" in num_data:
                latex_table.append(
                    f"{k} & {ACA_mean[k-1]:.2e} & {ACA_GP_value} & {SVD_value} & {accuracy_gain_mean[k-1]:.2e} \\\\"
                )
            else:
                latex_table.append(
                    f"{k} & {ACA_mean[k-1]:.2e} & {ACA_GP_value} & -- & {accuracy_gain_mean[k-1]:.2e} \\\\"
                )

    latex_table.append(r"    \hline")
    latex_table.append(r"\end{tabular}")

    latex_tables_by_dist[dist] = latex_table

# Save each LaTeX table to a separate .tex file in the provided folder
for dist, table in latex_tables_by_dist.items():
    filename = os.path.join(folder_name, f"Table_{prefix}_dist_{dist:.1f}_xi.tex")
    with open(filename, "w") as f:
        for line in table:
            f.write(line + "\n")

#########################################################
#  Time comparison LaTeX table (Dist columns, xi rows)  #
#########################################################


# Gather all unique xi and distance values
xi_values = sorted(set(json_data['xi'] for _, json_data in Data))
distances = sorted(data_by_dist.keys())


# Start building the LaTeX table
latex_time_table = []

# Create the table header with Dist values as columns
header = r"\begin{tabular}{ c " + " ".join([f"p{{2cm}}" for _ in distances]) + " }"
latex_time_table.append(header)
latex_time_table.append(r"    $\xi$ & " + " & ".join([f"Dist = {dist:.1f}" for dist in distances]) + r" \\")
latex_time_table.append(r"    \hline")

# Fill the table rows for each xi value
for xi in xi_values:
    row = [f"{xi:.2f}"]
    
    for dist in distances:
        data_group = data_by_dist[dist]
        
        # Find the entry that matches the current xi
        matching_data = [num_data for _, json_data, num_data in data_group if json_data['xi'] == xi]
        
        if matching_data and 'ACA_time' in matching_data[0] and 'ACA_GP_time' in matching_data[0]:            
            ACA_time = np.mean(matching_data[0]["ACA_time"])
            ACA_GP_time = np.mean(matching_data[0]["ACA_GP_time"])
            time_ratio = ACA_GP_time / ACA_time
            row.append(f"{time_ratio:.2f}")
        else:
            row.append("--")  # Use '--' if no data is available
    
    latex_time_table.append(" & ".join(row) + r" \\")

# Close the table
latex_time_table.append(r"    \hline")
latex_time_table.append(r"\end{tabular}")

# Save the LaTeX time comparison table
filename = os.path.join(folder_name, "time_comparison_table_xi_vs_dist.tex")
with open(filename, "w") as f:
    for line in latex_time_table:
        f.write(line + "\n")