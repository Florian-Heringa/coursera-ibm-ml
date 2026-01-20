import polars as pl
import polars.selectors as ps
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

AUDIO_FEATURE_COLS = ["danceability", "energy", "speechiness", "acoustiness", 
                      "instrumentalness", "liveness", "valence"]
RS = 42

def preprocess_data(path):
    data_orig = pl.read_csv(path)
    features = data_orig.select(AUDIO_FEATURE_COLS)
    mms = MinMaxScaler()
    features = pl.DataFrame(mms.fit_transform(features), schema=features.columns)
    return data_orig, features

def plot_genre_distribution(data_orig: pl.DataFrame):
    return (
        data_orig
        .group_by("genre")
        .agg(pl.len().alias("count"))
        .sort("Count", descending=True)
        .rename({"genre": "Genre"})
    ).to_pandas().plot.bar(
        x="Genre",
        y="Count",
        figsize=(15, 6),
        ylabel="Count",
        legend=None,
        title="Genre Counts"
    )
    
def plot_audio_features_heatmap(data: pl.DataFrame):
    ax = sns.heatmap(
        data
        .select(AUDIO_FEATURE_COLS)
        .corr(),
        cmap="coolwarm",
        annot=True,
        xticklabels=AUDIO_FEATURE_COLS,
        yticklabels=AUDIO_FEATURE_COLS,
    )
    ax.tick_params(labelrotation=45)
    return ax

def audio_features_pca(data: pl.DataFrame, threshold=0.95):
    pca = PCA(n_components=threshold, random_state=RS, solver="full").fit(data)
    print(f"PCA has kept {pca.components_.shape[0]/data.shape[1]} components for a variance threshold of {threshold}.")
    return pca

def audio_features_GMM_gridsearch(data):
    gmm_bic_score = lambda estimator, X: -estimator.named_steps['gmm'].bic(estimator.named_steps['scaler'].transform(X))
    param_grid = {
        "gmm__n_components": range(4, 10),
        "gmm__covariance_type": ["spherical", "tied", "diag", "full"]
    }
    
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gmm", GaussianMixture())
        ]
    )
    
    gm_gs = GridSearchCV(
        pipe, 
        param_grid=param_grid, 
        scoring=gmm_bic_score,
        verbose=2
    )
    gm_gs.fit(data)
    
    return gm_gs