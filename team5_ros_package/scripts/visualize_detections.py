# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file
# csv_file_path = '/workspaces/rss_workspace/data/object_detections_2.csv'
# df = pd.read_csv(csv_file_path)

# # Display the first few rows of the dataframe to understand its structure
# print(df.head())

# # Plot the x-y positions of different objects
# plt.figure(figsize=(12, 8))

# # Scatter plot for each unique class_id
# for class_id in df['class_id'].unique():
#     class_data = df[df['class_id'] == class_id]
#     plt.scatter(class_data['pose_x'], class_data['pose_y'], label=f'Class {class_id}')

# # Add titles and labels
# plt.title('X-Y Positions of Detected Objects')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.legend()
# plt.grid(True)
# plt.show()
# print("visualization finished")

############################

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import numpy as np

# # Load the CSV file
# csv_file_path = '/workspaces/rss_workspace/data/object_detections_20240703_115752.csv'
# df = pd.read_csv(csv_file_path)

# # Display the first few rows of the dataframe to understand its structure
# print(df.head())

# # Function to visualize the data
# def visualize_data(df):
#     plt.figure(figsize=(12, 8))
#     for class_id in df['class_id'].unique():
#         class_data = df[df['class_id'] == class_id]
#         plt.scatter(class_data['pose_x'], class_data['pose_y'], label=f'Class {class_id}')
#     plt.title('X-Y Positions of Detected Objects')
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Visualize the initial data
# visualize_data(df)

# # Prompt user to enter the number of objects per class
# num_objects_per_class = {}
# for class_id in df['class_id'].unique():
#     k = int(input(f"Enter the number of objects for class {class_id}: "))
#     num_objects_per_class[class_id] = k

# # Perform K-means clustering for each class and save the cluster centers
# cluster_centers = []

# for class_id, k in num_objects_per_class.items():
#     class_data = df[df['class_id'] == class_id]
#     if len(class_data) >= k:
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         class_data['cluster'] = kmeans.fit_predict(class_data[['pose_x', 'pose_y']])
        
#         for cluster_id in range(k):
#             cluster_data = class_data[class_data['cluster'] == cluster_id]
#             center_x = cluster_data['pose_x'].mean()
#             center_y = cluster_data['pose_y'].mean()
#             center_z = cluster_data['pose_z'].mean()
#             cluster_centers.append([class_id, center_x, center_y, center_z])
#     else:
#         print(f"Not enough data points for class {class_id} to form {k} clusters.")

# # Save the cluster centers to a CSV file
# cluster_centers_df = pd.DataFrame(cluster_centers, columns=['class_id', 'center_x', 'center_y', 'center_z'])
# output_csv_path = '/workspaces/rss_workspace/data/object_cluster_centers.csv'
# cluster_centers_df.to_csv(output_csv_path, index=False)

# print(f"Cluster centers saved to {output_csv_path}")

# # Visualize the clustered data
# plt.figure(figsize=(12, 8))
# for class_id in df['class_id'].unique():
#     class_data = df[df['class_id'] == class_id]
#     plt.scatter(class_data['pose_x'], class_data['pose_y'], label=f'Class {class_id}', alpha=0.5)
# for center in cluster_centers:
#     plt.scatter(center[1], center[2], label=f'Cluster Center {center[0]}', marker='x', s=100, color='red')
# plt.title('X-Y Positions of Detected Objects with Cluster Centers')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.legend()
# plt.grid(True)
# plt.show()

#####################################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load the CSV file
csv_file_path = '/workspaces/rss_workspace/data/object_detections_20240703_115752.csv'
df = pd.read_csv(csv_file_path)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Convert the confidence column to floats
df['confidence'] = df['confidence'].astype(float)

# Prompt user to enter the confidence level threshold
confidence_threshold = float(input("Enter the confidence level threshold (0-1): "))

# Filter the dataframe based on the confidence level
df = df[df['confidence'] >= confidence_threshold]

# Function to visualize the data
def visualize_data(df):
    plt.figure(figsize=(12, 8))
    for class_id in df['class_id'].unique():
        class_data = df[df['class_id'] == class_id]
        plt.scatter(class_data['pose_x'], class_data['pose_y'], label=f'Class {class_id}')
    plt.title('X-Y Positions of Detected Objects')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize the filtered data
visualize_data(df)

# Prompt user to enter the number of objects per class
num_objects_per_class = {}
for class_id in df['class_id'].unique():
    k = int(input(f"Enter the number of objects for class {class_id}: "))
    num_objects_per_class[class_id] = k

# Perform K-means clustering for each class and save the cluster centers
cluster_centers = []

for class_id, k in num_objects_per_class.items():
    class_data = df[df['class_id'] == class_id]
    if len(class_data) >= k:
        kmeans = KMeans(n_clusters=k, random_state=42)
        class_data['cluster'] = kmeans.fit_predict(class_data[['pose_x', 'pose_y']])
        
        for cluster_id in range(k):
            cluster_data = class_data[class_data['cluster'] == cluster_id]
            center_x = cluster_data['pose_x'].mean()
            center_y = cluster_data['pose_y'].mean()
            center_z = cluster_data['pose_z'].mean()
            cluster_centers.append([class_id, center_x, center_y, center_z])
    else:
        print(f"Not enough data points for class {class_id} to form {k} clusters.")

# Save the cluster centers to a CSV file
cluster_centers_df = pd.DataFrame(cluster_centers, columns=['class_id', 'center_x', 'center_y', 'center_z'])
output_csv_path = '/workspaces/rss_workspace/data/object_cluster_centers.csv'
cluster_centers_df.to_csv(output_csv_path, index=False)

print(f"Cluster centers saved to {output_csv_path}")

# Visualize the clustered data
plt.figure(figsize=(12, 8))
for class_id in df['class_id'].unique():
    class_data = df[df['class_id'] == class_id]
    plt.scatter(class_data['pose_x'], class_data['pose_y'], label=f'Class {class_id}', alpha=0.5)
for center in cluster_centers:
    plt.scatter(center[1], center[2], label=f'Cluster Center {center[0]}', marker='x', s=100, color='red')
plt.title('X-Y Positions of Detected Objects with Cluster Centers')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()
