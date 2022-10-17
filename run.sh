#!/bin/bash

# process 3 raw files and merge into one and save all files in data/processed as csv and pickle
echo "Processing raw data..."
python src/data/process.py --csv --pickle

# generate general statistics
# python src/exploratory/general_statistics.py

# generate group level statistics
# python src/exploratory/group_statistics.py

# generate some plots of group interaction
echo "Generating timeline of interactions plots..."
python src/exploratory/group_plot.py -rg 10

# generate some graph plots of group network effects, where -it creates one graph per intervention type
# and -nw normalizes each graph weights
echo "Generating graph plots..."
python src/exploratory/graph_plot.py -rg 10 -it -nw

# generate some plots of timeline of interventions for each day 
# and also polar plots of group responses and DA interventions
echo "Generating intervention timeline and polar plots..."
python src/exploratory/interventions.py -rg 10

# generate database with daily features
echo "Generating database of daily features of interactions..."
python src/data/create_intervention_database_daily.py

# generate some plots of group activation probabilities heatmap
echo "Generating heatmap plots..."
python src/exploratory/activation_heatmap.py -rg 10


