import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# ideas
# number of fish normalised
# 30min bins
# plot only active?

# load in data
file_path = '/Users/annikanichols/Dropbox/Schier_lab/_cichlids/_Home_tank_measurements/Neonig.csv'
species = os.path.split(file_path)[1][0:-4]

fish_data = pd.read_csv(file_path)

# change_times_unit = '1-1-1900 20:00:00'
# lights_off ='1-1-1900 20:00:00'
# lights_on = '2-1-1900 07:45:00'
# lights_on = '2-1-1900 08:00:00'
change_times_str = ["19:45:00", "20:00:00", "07:45:00", "08:00:00"]

# chage data type from object to datetime
# need to change day from 0 as this is not valid.
fish_data['real_time'] = fish_data['real_time'].str.replace(r'^1-', '2-', regex=True)
fish_data['real_time'] = fish_data['real_time'].str.replace(r'^0-', '1-', regex=True)
fish_data['real_time_dt'] = pd.to_datetime(fish_data['real_time'], format='%d-%m-%Y %H:%M:%S')
fish_data['time_only'] = fish_data['real_time_dt'].dt.strftime('%H:%M:%S')

# bin data by 30mins
fish_data_30m = fish_data.resample('30T', on='real_time_dt').mean()
fish_data_30m.reset_index(inplace=True)

# Get the first and second dates from the dataset
unique_dates = fish_data['real_time_dt'].dt.date.unique()

# Convert change times to datetime format for both days, for the correct day
change_times_unit = [
    pd.to_datetime(f"{unique_dates[0]} {change_times_str[0]}"),  # 7:45 PM on Day 1
    pd.to_datetime(f"{unique_dates[0]} {change_times_str[1]}"),  # 8:00 PM on Day 1
    pd.to_datetime(f"{unique_dates[1]} {change_times_str[2]}"),  # 7:45 AM on Day 2 (Add one day)
    pd.to_datetime(f"{unique_dates[1]} {change_times_str[3]}")   # 8:00 AM on Day 2 (Add one day)
]

# Define desired tick times
# desired_times = ["06:00:00", "12:00:00", "18:00:00", "00:00:00"]
desired_times = ["00:00:00", "12:00:00"]

# Get unique dates in the dataset
unique_dates = fish_data['real_time_dt'].dt.date.unique()
# Create tick marks for every date at 6AM, 12PM, 6PM, and 12AM
tick_positions = [pd.to_datetime(f"{date} {time}") for date in unique_dates for time in desired_times]
# Create formatted labels (e.g., "06:00 AM", "12:00 PM" with the date)
# need to make this 24h time
tick_labels = [f"{pd.to_datetime(t).strftime('%H:%M')}" for t in tick_positions]

# font sizes
SMALLEST_SIZE = 5
SMALL_SIZE = 6
matplotlib.rcParams.update({'font.size': SMALL_SIZE})

# plot data
plt.figure(figsize=(1.2, 1.2))
ax = sns.lineplot(x=fish_data_30m.real_time_dt, y=fish_data_30m.active, linewidth=0.5, color='k')
# ax = sns.lineplot(x=fish_data.real_time_dt, y=fish_data.inactive, linewidth=4, color='tab:orange')

ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
ax.axvspan(change_times_unit[1], change_times_unit[2], color='lightblue', alpha=0.5, linewidth=0)
ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)

ax.set_xticks(tick_positions)
# ax.set_xticklabels(tick_labels, rotation=45)
ax.set_xticklabels(tick_labels)

# Define x-axis limits (6PM Day 1 to 9AM Day 2)
xlim_start = pd.to_datetime(f"{unique_dates[0]} 12:00:00")  # 6:00 PM on Day 1
xlim_end = pd.to_datetime(f"{unique_dates[1]} 11:59:59")    # 9:00 AM on Day 2

ax.set_xlim(xlim_start, xlim_end)

plt.xlabel("Time (hh:mm)", fontsize=SMALL_SIZE)
plt.ylabel("# fish active", fontsize=SMALL_SIZE)
plt.title(species, fontsize=SMALLEST_SIZE)

# Decrease the offset for tick labels on all axes
ax.xaxis.labelpad = 0.5
ax.yaxis.labelpad = 0.5

# Adjust the offset for tick labels on all axes
ax.tick_params(axis='x', pad=0.5, length=2)
ax.tick_params(axis='y', pad=0.5, length=2)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
ax.tick_params(width=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.title(species)
# plt.xlabel("Time (h:m)")
# plt.ylabel("# fish active")
plt.tight_layout()

plt.savefig(os.path.join(os.path.split(file_path)[0], "daily_activity_{}.pdf".format(species)))
plt.close()
