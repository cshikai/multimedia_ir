import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
import umap
import torch
import pandas as pd
import ast
import numpy as np
import enchant

df = pd.read_csv('data/articles_entity_linked.csv')
df = df[df.entity_names == 'Unknown']

print(df.info())

embeddings = [np.array(ast.literal_eval(row['embeddings'])).reshape(-1) for idx, row in df.iterrows()]
print(embeddings[0])

if len(embeddings) <= 300:
        umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=300,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            init="random",
        )

else:
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=300,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

embeddings = umap_model.fit_transform(embeddings)

# print("minimum cluster size for {} docs".format(len(embeddings)), 1)
hdb_model = hdbscan.HDBSCAN(
    min_cluster_size=2,
    metric="euclidean",
    cluster_selection_method="leaf",
    prediction_data=True,
)

hdb_model.fit(embeddings)

cluster = hdb_model.labels_.tolist()
print("cluster: ", cluster)

df["cluster"] = cluster
df = df.sort_values(["cluster"], ascending=True)
list_of_cluster_dfs = df.groupby('cluster')

avg_dist = []
ooc = []
diff = []

for group, cluster_df in list_of_cluster_dfs:
    if group == -1:
        mentions = cluster_df['mention'].tolist()
        for mention in mentions:
            avg_dist.append(1000)
            ooc.append('No Cluster')
            diff.append(0.0)
    else:
        mentions = cluster_df['mention'].tolist()
        total_edit_dist = 0
        mention_dist = []
        for mention in mentions:
            mention_total_edit_dist = 0
            for other_mention in mentions:
                mention_total_edit_dist += enchant.utils.levenshtein(mention, other_mention) - abs(len(mention)-len(other_mention))
            mention_avg_edit_dist = mention_total_edit_dist/len(mentions)
            mention_dist.append(mention_avg_edit_dist)
            total_edit_dist += mention_avg_edit_dist
        avg_edit_dist = total_edit_dist/len(mentions)

        for idx in range(0,len(mentions)):
            avg_dist.append(avg_edit_dist)
            if mention_dist[idx] >= avg_edit_dist and mention_dist[idx] >0 and abs(mention_dist[idx]-avg_edit_dist) > 0.2:
                ooc.append("Out of Cluster")
                diff.append(abs(mention_dist[idx]-avg_edit_dist))
                print(mentions)
                print(mentions[idx])
            else:
                ooc.append("")
                diff.append(0.0)

df["avg_edit_dist"] = avg_dist
df['ooc'] = ooc
df['diff_to_avg'] = diff

df = df.sort_values(["cluster"], ascending=True)

print(df.info())

df.to_csv("data/hdbscan_df.csv",index=False)
