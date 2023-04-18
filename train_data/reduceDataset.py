# script for cleaning the Darpa Data set

# input files are X.csv y.csc


# output files are
# X_scaled.csv -- all features scaled to 0..1
# X_normal.csv y.csv -- scaled features with 50/50 mix of both classes
# Xnn.csv ynn.csv where nn is the number of samples

# importing libraries and magic functions

import numpy as np  # linear algebra
import pandas as pd  # train_data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# read train_data
df = pd.read_csv('X_norm.csv')
dfy = pd.read_csv('y.csv')

# train_data shape
print("Initial train_data shape (samples, features)", df.shape)

# train_data types
print("train_data types")
print(df.dtypes)
print(df)
print("label types")
print(dfy.dtypes)
print(dfy)
#remove intercept (we will put it back later)

df_clean = df.drop(['intercept'], axis=1)

boxplot = df_clean.boxplot()
plt.xlabel("feature")
plt.ylabel("value")
plt.title("cleaned train_data box plots")
#plt.show()



scaler = MinMaxScaler(feature_range=(-1, 1))

# assign scaler to column:
df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

print("df_scaled")
print(df_scaled)
# X_scaled = scale * X + min - X.min(axis=0) * scale
# where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))
# where min, max = feature_range.

# concat scale factors
this_scaling = np.array([scaler.min_, scaler.scale_])
print("this_scaling ", this_scaling)
np.savetxt("X_scaling.csv", this_scaling, delimiter=',')

# now writeout the cleaned train_data frame
df_scaled.to_csv('X_scaled.csv', index=False)
boxplot = df_scaled.boxplot()
plt.xlabel("feature")
plt.ylabel("value")
plt.title("scaled train_data box plots")
#plt.show()

# verify we can create the train_data
test_array = df_clean.to_numpy() * this_scaling[1] + this_scaling[0]
zero_array = df_scaled.to_numpy() - test_array
print("this should be zero")
print(zero_array)


#df_scaled = df_clean

#append the label train_data
df_scaled = pd.concat([df_scaled, dfy], axis = 1, join="inner");
print(df_scaled.dtypes)
print("df_scaled with y")
print(df_scaled)

# Checking balance of outcome variable
target_count = df_scaled.IMORT.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

sns.countplot(df_scaled.IMORT, palette="OrRd")
plt.box(False)
plt.xlabel('Heart Disease No/Yes', fontsize=11)
plt.ylabel('Patient Count', fontsize=11)
plt.title('Count Outcome Heart Disease\n')
# plt.savefig('Balance Heart Disease.png')
plt.show()

# Shuffle df
#shuffled_df = df_scaled.sample(frac=1, random_state=4)

# Put all the true class in a separate dataset.
TRUE_df = df_scaled.loc[df_scaled['IMORT'] == 1]
TRUE_df=TRUE_df.sample(frac=1, random_state=4).reset_index(drop=True)
n_samp = TRUE_df.shape[0]  # number of true samples
print('length ', TRUE_df.shape[1])

# Randomly select n_samp observations from the false class
FALSE_df = df_scaled.loc[df_scaled['IMORT'] == 0].sample(n=n_samp, random_state=42)
FALSE_df = FALSE_df.sample(frac=1, random_state=4).reset_index(drop=True)



'''

#prune both down to n_small samp
small_n_samp = 32;

print("TRUE_df");
print(TRUE_df);
print("FALSE_df");
print(FALSE_df);
TRUE_df_pruned = TRUE_df.drop(index=range(small_n_samp, n_samp))
FALSE_df_pruned = FALSE_df.drop(index=range(small_n_samp, n_samp))

n_samp = TRUE_df_pruned.shape[0]  # number of true samples

print('t: n_samp pruned ', TRUE_df_pruned.shape[0])
print('t: length ', TRUE_df_pruned.shape[1])

n_samp = FALSE_df_pruned.shape[0]  # number of true samples

print('f: n_samp pruned ', FALSE_df_pruned.shape[0])
print('f: length ', FALSE_df_pruned.shape[1])

print("TRUE_df_pruned");
print(TRUE_df_pruned);
print("FALSE_df_pruned");
print(FALSE_df_pruned);

# Concatenate both dataframes again
normalized_df = pd.concat([TRUE_df_pruned, FALSE_df_pruned])

print("normalized_df");
print(normalized_df);

# check new class counts
normalized_df.IMORT.value_counts()
target_count = normalized_df.IMORT.value_counts()
print("after pruning");
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

# plot new count
sns.countplot(normalized_df.IMORT, palette="OrRd")
plt.box(False)
plt.xlabel('Heart Disease No/Yes', fontsize=11)
plt.ylabel('Patient Count', fontsize=11)
plt.title('Count Outcome Heart Disease after Resampling\n')
# plt.savefig('Balance Heart Disease.png')
plt.show()

y = normalized_df.iloc[:, -1]


normalized_df = normalized_df.drop(["IMORT"], axis = 1)
#add back intercept
normalized_df['intercept'] = 1
normalized_df.to_csv('X_norm_64.csv', index=False)
y.to_csv('y_64.csv', index=False)


'''
#Repeat for 1024 pts. NOte this could be a function.


#prune both down to n_small samp
small_n_samp = 512;

TRUE_df_pruned = TRUE_df.drop(index=range(small_n_samp, n_samp))
FALSE_df_pruned = FALSE_df.drop(index=range(small_n_samp, n_samp))

n_samp = TRUE_df_pruned.shape[0]  # number of true samples

print('t: n_samp pruned ', TRUE_df_pruned.shape[0])
print('t: length ', TRUE_df_pruned.shape[1])

n_samp = FALSE_df_pruned.shape[0]  # number of true samples

print('f: n_samp pruned ', FALSE_df_pruned.shape[0])
print('f: length ', FALSE_df_pruned.shape[1])

print("TRUE_df_pruned");
print(TRUE_df_pruned);
print("FALSE_df_pruned");
print(FALSE_df_pruned);

# Concatenate both dataframes again
normalized_df = pd.concat([TRUE_df_pruned, FALSE_df_pruned])

print("normalized_df");
print(normalized_df);

# check new class counts
normalized_df.IMORT.value_counts()
target_count = normalized_df.IMORT.value_counts()
print("after pruning");
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

# plot new count
sns.countplot(normalized_df.IMORT, palette="OrRd")
plt.box(False)
plt.xlabel('Heart Disease No/Yes', fontsize=11)
plt.ylabel('Patient Count', fontsize=11)
plt.title('Count Outcome Heart Disease after Resampling\n')
# plt.savefig('Balance Heart Disease.png')
plt.show()

y = normalized_df.iloc[:, -1]


normalized_df = normalized_df.drop(["IMORT"], axis = 1)
#add back intercept
normalized_df['intercept'] = 1
normalized_df.to_csv('X_norm_1024.csv', index=False)
y.to_csv('y_1024.csv', index=False)

