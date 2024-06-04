import os
import json
import matplotlib.pyplot as plt

base_dir = "/home/kabongo/reco_nlp/data/process/All_Beauty/exps"
exps_paths = [
    "google_flan-t5-base_prompting_0_shot_10_reviews_2_sampling_1717407794",
    "google_flan-t5-base_prompting_0_shot_9_reviews_2_sampling_1717408347",
    "google_flan-t5-base_prompting_0_shot_8_reviews_2_sampling_1717408748",
    "google_flan-t5-base_prompting_0_shot_7_reviews_2_sampling_1717409167",
    "google_flan-t5-base_prompting_0_shot_6_reviews_2_sampling_1717409655",
    "google_flan-t5-base_prompting_0_shot_5_reviews_2_sampling_1717415391",
    "google_flan-t5-base_prompting_0_shot_4_reviews_2_sampling_1717416589",
    "google_flan-t5-base_prompting_0_shot_3_reviews_2_sampling_1717417483",
    "google_flan-t5-base_prompting_0_shot_2_reviews_2_sampling_1717418363",
    "google_flan-t5-base_prompting_0_shot_1_reviews_2_sampling_1717419363",
]
n_reviews = list(range(10, 0, -1))

n_non_numerical_list = []
rmse_list = []
mae_list = []
precision_list = []
recall_list = []
f1_list = []
auc_list = []


for exp in exps_paths:
    res_file = os.path.join(base_dir, exp, 'res.json')
    with open(res_file, 'r') as f:
        data = json.load(f)
        ratings = data.get('ratings', {})

        n_non_numerical_list = ratings.get('n_non_numerical_list', None)
        rmse = ratings.get('rmse', None)
        mae = ratings.get('mae', None)
        precision = ratings.get('precision', None)
        recall = ratings.get('recall', None)
        f1 = ratings.get('f1', None)
        auc = ratings.get('auc', None)

        rmse_list.append(rmse)
        mae_list.append(mae)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc)


plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(n_reviews, rmse_list, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Reviews')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of Reviews')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(n_reviews, mae_list, marker='o', linestyle='-', color='g')
plt.xlabel('Number of Reviews')
plt.ylabel('MAE')
plt.title('MAE vs Number of Reviews')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(n_reviews, precision_list, marker='o', linestyle='-', color='r')
plt.xlabel('Number of Reviews')
plt.ylabel('Precision')
plt.title('Precision vs Number of Reviews')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(n_reviews, recall_list, marker='o', linestyle='-', color='c')
plt.xlabel('Number of Reviews')
plt.ylabel('Recall')
plt.title('Recall vs Number of Reviews')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(n_reviews, f1_list, marker='o', linestyle='-', color='m')
plt.xlabel('Number of Reviews')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Number of Reviews')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(n_reviews, auc_list, marker='o', linestyle='-', color='y')
plt.xlabel('Number of Reviews')
plt.ylabel('AUC')
plt.title('AUC vs Number of Reviews')
plt.grid(True)

plt.tight_layout()
plt.savefig("user_only_res.png")

