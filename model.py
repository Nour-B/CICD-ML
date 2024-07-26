
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Step 4 - Detect the optimal number of clusters for k-means clustering
def plot_optimal_number_cluster(df : pd.DataFrame):
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
        inertia.append(kmeans.inertia_)    
    plt.plot(range(1, 10), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.savefig("inertia.png")
    plt.clf()
    

# Step 5 - Run the k-means clustering algorithm
# with the optimal number of clusters 

def run_model(df: pd.DataFrame, n_clusters=4):
    penguins_df = pd.read_csv('raw_data/penguins.csv')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(df)
    penguins_df['label'] = kmeans.labels_


    # and visualize the clusters (here for the 'culmen_length_mm' column)
    plt.scatter(penguins_df['label'], penguins_df['culmen_length_mm'], c=kmeans.labels_, cmap='viridis')
    plt.xlabel('Cluster')
    plt.ylabel('culmen_length_mm')
    plt.xticks(range(int(penguins_df['label'].min()), int(penguins_df['label'].max()) + 1))
    plt.title(f'K-means Clustering (K={n_clusters})')
    plt.savefig("kmeans.png")

    # Step - create final `stat_penguins` DataFrame
    numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm','label']
    stat_penguins = penguins_df[numeric_columns].groupby('label').mean()
    print(stat_penguins.head())


def main():
    
    df = pd.read_csv('processed_data/processed_penguins.csv')
    plot_optimal_number_cluster(df)
    run_model(df, n_clusters=4)


if __name__ == "__main__":
    main()
 