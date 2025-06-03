# cichlid-analysis
cichlid behaviour analysis code 

General pipeline
Get units and the fish size:

Run measure_units.py (assumes you're measuring the width of the divider which is divider_base_mm = 15) on each camera folder's background.
Run measure_fish.py (allows you to select a frame to measure the standard length of the fish, and update the fish sex)

Run run_fish_als.py (creates initial plots to allow you to see how well the fish is tracked, generates als.csv files which have the full track for the fish)

Quality control:
Run tracker_checker.py (a GUI for looking at the tracking of videos you think might not have tracked well - generally if thresholded plot has a lot of black)

Ways to improve tracking quality:
1. divide_tracking.py (allows you to retrack movies with different background images created from that video with the inputted time bins)
2. split_tracking.py (allows you to exclude a period of the tracking, this is often necessary if there was a bump to the camera, water level got too low and we can see when it was added etc).
3. retracking.py (can change the ROI, generate new backgrounds etc).

After you have gone through the videos, you now only need the _als.csv and _meta.csv files. We move these to an analysis folder which have these files for each fish of that species.

Now you can run
run_combine_fish.als (creates combined data and bins data for that species, generates the _als_30m.csv and _als_fv2.csv files)

You can move the _als_30m.csv and _als_fv2.csv files to a folder which has these files for all species. This will allow you to run:
run_binned.py (for plots and analysis across fish species using the _als_30m.csv data)
and
run_feature_vector.py (for plots and analysis across fish species using the _als_fv2.csv data)

run_weekly_daily_figure_plots.py (for the single plots for the fishes for the extended figures)

PCA (runs the PCA analysis in Figure 1)
