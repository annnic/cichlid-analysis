import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import matplotlib
import seaborn as sns
from matplotlib.dates import DateFormatter

from cichlidanalysis.utils.timings import output_timings_14_8

# ideas


# load in data
file_path = '/Users/annikanichols/Dropbox/Schier_lab/_cichlids/Home_tank_measurements/Neocra.csv'
species = 'Neocra'

fish_data = pd.read_csv(file_path)

# get time variables
change_times_s_14_8, change_times_ns_14_8, change_times_m_14_8, change_times_h_14_8, day_ns, day_s, \
change_times_d_14_8, change_times_datetime_14_8, change_times_unit_14_8 = output_timings_14_8()


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
desired_times = ["06:00:00", "12:00:00", "18:00:00", "00:00:00"]
# Get unique dates in the dataset
unique_dates = fish_data['real_time_dt'].dt.date.unique()
# Create tick marks for every date at 6AM, 12PM, 6PM, and 12AM
tick_positions = [pd.to_datetime(f"{date} {time}") for date in unique_dates for time in desired_times]
# Create formatted labels (e.g., "06:00 AM", "12:00 PM" with the date)
tick_labels = [f"{pd.to_datetime(t).strftime('%I:%M %p')}\n{pd.to_datetime(t).strftime('%b %d')}" for t in tick_positions]

# plot data
plt.figure(figsize=(6, 4))
ax = sns.lineplot(x=fish_data.real_time_dt, y=fish_data.active, linewidth=4, color='tab:blue')
ax = sns.lineplot(x=fish_data.real_time_dt, y=fish_data.inactive, linewidth=4, color='tab:orange')


ax.axvspan(change_times_unit[0], change_times_unit[1], color='wheat', alpha=0.5, linewidth=0)
ax.axvspan(change_times_unit[1], change_times_unit[2], color='lightblue', alpha=0.5, linewidth=0)
ax.axvspan(change_times_unit[2], change_times_unit[3], color='wheat', alpha=0.5, linewidth=0)

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45)

# Define x-axis limits (6PM Day 1 to 9AM Day 2)
xlim_start = pd.to_datetime(f"{unique_dates[0]} 16:00:00")  # 6:00 PM on Day 1
xlim_end = pd.to_datetime(f"{unique_dates[1]} 12:00:00")    # 9:00 AM on Day 2

# Apply x-axis limits
ax.set_xlim(xlim_start, xlim_end)

# plt.xlabel("Time (h:m)")
# plt.ylabel("Speed (mm/s)")
# ax.xaxis.set_major_locator(MultipleLocator(6))
plt.savefig(os.path.join(os.path.split(file_path)[0], "daily_activity_{}.png".format(species)))
plt.close()
