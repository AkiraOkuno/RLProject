#!/bin/bash

# process 3 raw files and merge into one and save all files in data/processed 
python src/data/process.py

# generate general statistics
# python src/exploratory/general_statistics.py

# generate all plots of group interaction
python src/exploratory/group_plot.py -ag

# generate group level statistics
# python src/exploratory/group_statistics.py