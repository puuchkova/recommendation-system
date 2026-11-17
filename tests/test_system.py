import numpy as np
from src.data_loader import load_data
from src.recommender import HybridRecommender, topsis
from src.clustering import cluster_users

def test_constraints_and_cold_start():
    p,u,i=load_data(); m=HybridRecommender(p,i)
    for uid in [0,int(u.user_id.iloc[0])]:
        out=m.recommend(uid,30000,category='Наушники',method='hybrid+topsis')
        assert len(out)>0
        assert (out.price<=30000).all() and (out.category=='Наушники').all()
        assert not out.product_id.isin(i.loc[i.user_id==uid,'product_id']).any()
        assert np.isfinite(out.final).all()
    assert m.recommend(0,0).empty

def test_weight_endpoints_and_topsis():
    p,u,i=load_data(); m=HybridRecommender(p,i)
    a=m.scores(1,1); b=m.scores(1,0)
    assert np.allclose(a.hybrid,a.content) and np.allclose(b.hybrid,b.collaborative)
    sample=p.iloc[:2].copy(); sample['price']=[100,200]; sample['rating']=[5,3]; sample['cpu_score']=[90,50]
    assert topsis(sample)[0]>topsis(sample)[1]

def test_clustering_and_temporal_split():
    p,u,i=load_data(); c,s=cluster_users(p,u,i)
    assert len(c)==len(u) and c.cluster.nunique()==4 and -1<=s<=1
    test=i.sort_values(['timestamp','product_id']).groupby('user_id').tail(2); train=i.drop(test.index)
    assert set(train.index).isdisjoint(test.index)
    for uid,g in test.groupby('user_id'):
        assert train.loc[train.user_id==uid,'timestamp'].max()<=g.timestamp.min()
