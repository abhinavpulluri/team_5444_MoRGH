# MoRGH-Inspired Movie Recommender System using GNNs on Heterogeneous Graphs

This repository contains an implementation of a hybrid movie recommender system inspired by the [MoRGH model](https://doi.org/10.21203/rs.3.rs-3860094/v1). The system combines collaborative filtering (user-movie ratings) with content-based filtering (movie plot and genre similarity) using Graph Neural Networks (GNNs) on a heterogeneous graph.

---

## Features

- **Heterogeneous Graph Construction:**  
  Builds a graph with users, movies, user-movie rating edges, and movie-movie similarity edges.
- **Content-Based Movie Similarity:**  
  Uses Sentence-BERT embeddings for Wikipedia plot synopses and multi-hot vectors for genres.
- **Flexible Movie-Movie Edge Construction:**  
  - By default, connects each movie to its top-k most similar movies (kNN) based on cosine similarity of plot+genre features.
  - Optionally, can use a similarity threshold (as in the MoRGH paper) to create edges between all sufficiently similar movies.
- **Graph Neural Network Model:**  
  Employs a heterogeneous GNN (HeteroConv with SAGEConv) as an encoder and an MLP decoder for rating prediction.
- **Hybrid Recommendation:**  
  Integrates collaborative and content-based signals for robust, cold-start-resistant recommendations.

---

## Data Sources

- [MovieLens](https://grouplens.org/datasets/movielens/) (ratings.csv, movies.csv)
- [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) (wiki_movie_plots_deduped.csv)

---

## Getting Started

### 1. **Install Dependencies**
```bash
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv sentence-transformers scikit-learn pandas numpy
```

### 2. **Prepare Data**

- Download the required datasets and place them in your working directory or update the paths in the code.

### 3. **Run the Pipeline**

The main steps are:

1. **Load and preprocess data:**  
   - Normalize movie titles and merge MovieLens with Wikipedia plots.
2. **Feature engineering:**  
   - Generate Sentence-BERT embeddings for plots and multi-hot vectors for genres.
3. **Build the heterogeneous graph:**  
   - Nodes: users (dummy or learnable features), movies (plot+genre features)
   - Edges: 
     - user-movie (ratings)
     - movie-movie (cosine similarity, see below)
4. **Movie-Movie Edge Construction:**  
   - **By default:** For each movie, connect to its top-k most similar movies (kNN) using cosine similarity of plot+genre features.
   - **Optionally:** Use a similarity threshold (e.g., sim > 0.3) to create edges between all sufficiently similar movies, as in the MoRGH paper.
5. **Model definition:**  
   - Heterogeneous GNN encoder and MLP decoder.
6. **Training and evaluation:**  
   - 80/10/10 train/val/test split on rating edges.
   - RMSE reported as evaluation metric.

See the provided notebook/script for a complete, ready-to-run pipeline.

---

## Example Usage

```python
# After setting up and preparing data, run:
python morgh_pipeline.py
```
Or run the provided Jupyter/Colab notebook cell by cell.

---

## Model Details

- **Encoder:** 2-layer HeteroConv (SAGEConv) GNN
- **Decoder:** MLP for rating prediction
- **User features:** Zero vectors or learnable embeddings
- **Movie features:** Sentence-BERT plot embeddings + multi-hot genres
- **Movie-movie edges:**  
  - **Default:** kNN (top-k cosine similarity)
  - **Optional:** Cosine similarity threshold (as in MoRGH paper)
- **Optimizer:** Adam, learning rate 0.01
- **Epochs:** 100

---

## Results

- **Test RMSE:** 0.98-1.07 (MovieLens Small Dataset, using Wikipedia plots and genres)

---

## References

- Ziaee, S. S., Rahmani, H., & Nazari, M. (2024). MoRGH: Movie Recommender System using GNNs on Heterogeneous Graphs. [DOI:10.21203/rs.3.rs-3860094/v1](https://doi.org/10.21203/rs.3.rs-3860094/v1)

---

## Acknowledgements

- [MovieLens](https://grouplens.org/datasets/movielens/)
- [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
- [MoRGH Paper](https://doi.org/10.21203/rs.3.rs-3860094/v1)

---
