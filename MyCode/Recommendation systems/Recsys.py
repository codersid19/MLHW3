import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNWithMeans, model_selection, KNNBasic, accuracy
from surprise.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt

# Part A
# Load the data
data = pd.read_csv('S:/College Folder/UCF/Spring23/ML/HW3/Code/MyCode/Recommendation systems/data/ratings_small.csv')
print("Movie Data")
print(data.head())
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)



# Part B and C
# Computing average MAE and RMSE for  Probabilistic Matrix Factorization
# (PMF), User based Collaborative Filtering, Item based Collaborative Filtering,
# under the 5-folds cross-validation

pmf_model = SVD(biased=True, verbose=False)
ubcf_model = KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': True}, verbose=False)
ibcf_model = KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': False}, verbose=False)

resultsForEachModel = []

for model in [pmf_model, ubcf_model, ibcf_model]:
    model_results = cross_validate(model, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
    resultsForEachModel.append(model_results)

for i, model_result in enumerate(resultsForEachModel):
    print(f"Model {i}: MAE = {model_result['test_mae'].mean()}, RMSE = {model_result['test_rmse'].mean()}")

# # Part D
# # Compute Average MAE and RSME for user collaborative  with k

k_range = range(1, 31)

mae_scores = []
rmse_scores = []

for k in k_range:
    algo = KNNWithMeans(k=k, sim_options={'name': 'cosine', 'user_based': True})
    results = model_selection.cross_validate(algo, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
    mae_scores.append(results['test_mae'].mean())
    rmse_scores.append(results['test_rmse'].mean())


print("---------User-based collaborative filtering---------")

for k, mae, rmse in zip(k_range, mae_scores, rmse_scores):
    print(f"k = {k}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

avg_mae = sum(mae_scores) / len(mae_scores)
avg_rmse = sum(rmse_scores) / len(rmse_scores)
print(f"\nAverage MAE = {avg_mae:.4f}, Average RMSE = {avg_rmse:.4f}")

# Compute Average MAE and RSME for item collaborative  with k

k_range = range(1, 31)

mae_scores = []
rmse_scores = []

for k in k_range:
    algo = KNNWithMeans(k=k, sim_options={'name': 'cosine', 'user_based': False})
    results = model_selection.cross_validate(algo, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
    mae_scores.append(results['test_mae'].mean())
    rmse_scores.append(results['test_rmse'].mean())


print("---------Item-based collaborative filtering---------")

for k, mae, rmse in zip(k_range, mae_scores, rmse_scores):
    print(f"k = {k}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

avg_mae = sum(mae_scores) / len(mae_scores)
avg_rmse = sum(rmse_scores) / len(rmse_scores)
print(f"\nAverage MAE = {avg_mae:.4f}, Average RMSE = {avg_rmse:.4f}")

# Compute Average MAE and RSME for PMF  with k

k_range = range(1, 31)

mae_scores = []
rmse_scores = []

for k in k_range:
    algo = SVD(biased=True, verbose=False)
    results = model_selection.cross_validate(algo, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
    mae_scores.append(results['test_mae'].mean())
    rmse_scores.append(results['test_rmse'].mean())


print("---------Probabilistic Matrix Factorization---------")

for k, mae, rmse in zip(k_range, mae_scores, rmse_scores):
    print(f"k = {k}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

avg_mae = sum(mae_scores) / len(mae_scores)
avg_rmse = sum(rmse_scores) / len(rmse_scores)
print(f"\nAverage MAE = {avg_mae:.4f}, Average RMSE = {avg_rmse:.4f}")

# Part E
# Examine Cosine, MSD and Pearson similarities

# Define similarity metrics to test
similarity_metrics = ['cosine', 'msd', 'pearson']

# Create empty lists to store RMSE results
ubcf_rmse = []
ibcf_rmse = []

# Loop through similarity metrics and test on UBCF and IBCF
for metric in similarity_metrics:
    # UBCF
    sim_options = {'name': metric, 'user_based': True}
    ubcf = KNNWithMeans(k=30, sim_options=sim_options)
    ubcf_results = cross_validate(ubcf, data, measures=['RMSE'], cv=5, verbose=False)
    ubcf_rmse.append(sum(ubcf_results['test_rmse']) / len(ubcf_results['test_rmse']))

    # IBCF
    sim_options = {'name': metric, 'user_based': False}
    ibcf = KNNWithMeans(k=30, sim_options=sim_options)
    ibcf_results = cross_validate(ibcf, data, measures=['RMSE'], cv=5, verbose=False)
    ibcf_rmse.append(sum(ibcf_results['test_rmse']) / len(ibcf_results['test_rmse']))

# Plot UBCF results
x = similarity_metrics
y = ubcf_rmse
plt.bar(x, y)
plt.title('UBCF RMSE by Similarity Metric')
plt.xlabel('Similarity Metric')
plt.ylabel('RMSE')
plt.show()

# Plot IBCF results
y = ibcf_rmse
plt.bar(x, y)
plt.title('IBCF RMSE by Similarity Metric')
plt.xlabel('Similarity Metric')
plt.ylabel('RMSE')
plt.show()


from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
import matplotlib.pyplot as plt



# define the range of number of neighbors to evaluate
k_values = range(1, 30)

# evaluate UBCF and IBCF for different values of k
ubcf_mae = []
ibcf_mae = []
for k in k_values:
    # train UBCF with k neighbors
    ubcf = KNNWithMeans(k=k, sim_options={'name': 'cosine', 'user_based': True})
    ubcf.fit(trainset)
    ubcf_pred = ubcf.test(testset)
    ubcf_mae.append(mae(ubcf_pred))

    # train IBCF with k neighbors
    ibcf = KNNWithMeans(k=k, sim_options={'name': 'cosine', 'user_based': False})
    ibcf.fit(trainset)
    ibcf_pred = ibcf.test(testset)
    ibcf_mae.append(mae(ibcf_pred))

# plot the results
plt.plot(k_values, ubcf_mae, label='UBCF')
plt.plot(k_values, ibcf_mae, label='IBCF')
plt.xlabel('Number of neighbors')
plt.ylabel('MAE')
plt.legend()
plt.show()







# User-based collaborative filtering
rmse_scores_user = []
for k in range(1, 31):
    algo = KNNBasic(k=k, sim_options={'name': 'cosine', 'user_based': True})
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse_scores_user.append(accuracy.rmse(predictions))

# Item-based collaborative filtering
rmse_scores_item = []
for k in range(1, 31):
    algo = KNNBasic(k=k, sim_options={'name': 'cosine', 'user_based': False})
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse_scores_item.append(accuracy.rmse(predictions))

# Find the best K value for User-based collaborative filtering
best_k_user = np.argmin(rmse_scores_user) + 1
print(f"Best K value for User-based collaborative filtering: {best_k_user}")

# Find the best K value for Item-based collaborative filtering
best_k_item = np.argmin(rmse_scores_item) + 1
print(f"Best K value for Item-based collaborative filtering: {best_k_item}")



