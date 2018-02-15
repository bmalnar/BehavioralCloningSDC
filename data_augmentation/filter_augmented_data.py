import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

'''
The script takes the dataset that is potentially not very well balanced, in terms of
the distribution of the steering angle over the number of data points. 
The script creates the final more balanced dataset, so that the extra data points are
dropped (for the values of the steering angles which are represented with many more
examples than the average)
The script simply filters out some entries from the driving_log.csv file and creates
another csv file with those entries dropped
'''

# Number of bins to estimate the distribution of the steering angle
num_bins = 50

# Input csv file
file_name = "junk/driving_log_augm.csv"

# Output csv file
out_file_name = file_name.replace(".csv", "_filtered.csv")

# Read and plot the csv file
df = pd.read_csv(file_name)
fig, ax = plt.subplots()
df.hist('steering', ax=ax, bins=50)
fig.savefig('hist_before_filt.png')

# Calculate the distribution over the bins
st_col = df['steering']
bins, bounds = pd.cut(st_col, num_bins, retbins = True)
bounds_step = bounds[2] - bounds[1]
bins_sizes = []

for i in range(len(bounds)-1):
   lower = bounds[i]
   upper = bounds[i+1]
   print('Processing:', lower, upper)
   this_bin = st_col[(st_col > lower) & (st_col < upper)]
   print(len(this_bin))
   bins_sizes.append(len(this_bin))

print("Bins:", bins_sizes)
avg_bin_size = sum(bins_sizes) / len(bins_sizes)
print('Average bin size:', avg_bin_size)
avg_bin_size = 15000
print('Average bin size:', avg_bin_size)

# Create keep-probabilities for each bin
# The more examples a bin has, the lower the keep probability
# The goal is to get a more balanced distribution. 

bins_probs = [1 for i in range(len(bins_sizes))]

for i in range(len(bins_sizes)):
   if bins_sizes[i] <= avg_bin_size:
      bins_probs[i] = 1
   else:
      bins_probs[i] = 1 - (bins_sizes[i] - avg_bin_size) / bins_sizes[i]    

print("Bins keep-probabilities:", bins_probs)

f_out_csv = open(out_file_name, 'w')

f_out_csv.write("center,left,right,steering,throttle,brake,speed\n")

bins_count = [0 for i in range(len(bins_sizes))]

# Process the entries and create the new csv file
for i in range(len(df)):
   steering = df.iloc[i]['steering']
   bin_index = np.searchsorted(bounds, steering) - 1
   if bin_index == 0 or bin_index == len(bounds):
      continue

   # Keep the entry only if head=1 for the given probability
   head = bernoulli.rvs(bins_probs[bin_index])

   if head:
      # if bins_count[bin_index] < avg_bin_size:
      bins_count[bin_index] += 1
      f_out_csv.write(df.iloc[i]['center'] + ",")
      f_out_csv.write(df.iloc[i]['center'] + ",")
      f_out_csv.write(df.iloc[i]['center'] + ",")
      f_out_csv.write(str(steering))
      f_out_csv.write(",")
      f_out_csv.write(str(df.iloc[i]['throttle']))
      f_out_csv.write(",")
      f_out_csv.write(str(df.iloc[i]['brake']))
      f_out_csv.write(",")
      f_out_csv.write(str(df.iloc[i]['speed']))
      f_out_csv.write("\n")

f_out_csv.close()

# Create a new histogram based on the new distribution. 
df_filtered = pd.read_csv(out_file_name)
fig, ax = plt.subplots()
df_filtered.hist('steering', ax=ax, bins=50)

fig.savefig('hist_after_filt.png')


