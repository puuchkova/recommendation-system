import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def cluster_users(products, users, interactions, k=4):
    events = interactions.merge(products[['product_id','category','price']], on='product_id')
    f = events.groupby('user_id').agg(events=('rating','size'), mean_price=('price','mean'), mean_rating=('rating','mean'))
    shares = pd.crosstab(events.user_id, events.category, normalize='index').add_prefix('share_')
    f = f.join(shares).reindex(users.user_id).fillna(0)
    x = StandardScaler().fit_transform(f)
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(x)
    f['cluster'] = labels
    score = silhouette_score(x, labels, sample_size=min(1000,len(x)), random_state=42) if 1 < len(set(labels)) < len(x) else None
    return f.reset_index(), score
