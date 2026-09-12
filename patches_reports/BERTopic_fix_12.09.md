# BERTopic fix on 12.09 


## 1. BERTopic crashes witout noise docs
**location**: `topic_modeling_analysis.py` 

#### Description
HDBSCAN normally labels some documents as noise: topic -1. `reduce_outliers()` exists to sweep those up and push each one into its nearest real topic. if there is no noise (e.g. when the documents are very similar to each other) `reduce_outliers()` is asked to clean up an empty set, and BERTopic treats that as a mistake and raises instead of quietly doing nothing. 

#### Solution
Modify `topic_modeling_analysis.py` 

Add `hdbscan_min_samples` in line ~169
```python
hdbscan_min_cluster_size: int = 50
hdbscan_min_samples: Optional[int] = None
hdbscan_metric: str = 'euclidean'
hdbscan_cluster_selection_method: str = 'eom'
```

Add a swich for outlier reduction in line ~182
```python
reduce_outliers: bool = True
```

Replace lines ~1124-1136 with 

```python
n_outliers = sum(1 for t in topics if t == -1)
n_clusters = len({t for t in topics if t != -1})

print(f"Clustering: {n_clusters} topics, {n_outliers}/{len(topics)} outliers")

# if outlier reduction disabled
if not self.config.reduce_outliers:
    print("Outlier reduction disabled by config; skipping.")

# prevent the crash when no outliers are set
elif n_outliers == 0:
    print("No outliers found; skipping outlier reduction.")

# standard procedure
else:
    print(f"Reducing outliers ({n_outliers} documents)...")
    new_topics = self.model.reduce_outliers(
        docs,
        topics,
        strategy=self.config.outlier_strategy,
        threshold=self.config.outlier_threshold
    )
    self.model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer_model)
    # keep the returned labels in sync with the model's rebuilt state
    topics = new_topics

print(f"Model fitted with {len(self.model.get_topic_info())} topics")
```
Change the lines ~999-1004 to inlude new elements.

Add a swich for outlier reduction in line ~182
```python
return HDBSCAN(
    min_cluster_size=self.config.hdbscan_min_cluster_size,
    min_samples=self.config.hdbscan_min_samples,
    metric=self.config.hdbscan_metric,
    cluster_selection_method=self.config.hdbscan_cluster_selection_method,
    prediction_data=True
)
```
Fix the cuML branches in lines ~968–973

```python
return cuUMAP(
    n_neighbors=self.config.umap_n_neighbors,
    n_components=self.config.umap_n_components,
    min_dist=self.config.umap_min_dist,
    metric=self.config.umap_metric,
    random_state=self.config.umap_random_state
)
```

and ~992–995
```python
return cuHDBSCAN(
    min_cluster_size=self.config.hdbscan_min_cluster_size,
    min_samples=self.config.hdbscan_min_samples,
    cluster_selection_method=self.config.hdbscan_cluster_selection_method,
    prediction_data=True
)
```

Fix the `_setup_clustering` method  ~968–973

```python
def _setup_clustering(self):
    """Setup HDBSCAN clustering (GPU or CPU)."""
    if self.config.use_gpu:
        try:
            from cuml.cluster import HDBSCAN as cuHDBSCAN
            print("Using GPU-accelerated HDBSCAN")
            return cuHDBSCAN(
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                min_samples=self.config.hdbscan_min_samples,
                cluster_selection_method=self.config.hdbscan_cluster_selection_method,
                prediction_data=True
            )
        except ImportError:
            print("cuML not available, using CPU HDBSCAN")
    
    return HDBSCAN(
        min_cluster_size=self.config.hdbscan_min_cluster_size,
        min_samples=self.config.hdbscan_min_samples,
        metric=self.config.hdbscan_metric,
        cluster_selection_method=self.config.hdbscan_cluster_selection_method,
        prediction_data=True
    )
```

Add the consistency check just before line ~1144


```python
assert list(topics) == list(self.model.topics_), "returned topics diverge from model.topics_"

```