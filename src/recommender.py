import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def scale(x):
    x = np.asarray(x, dtype=float)
    return (x-x.min())/(x.max()-x.min()) if np.ptp(x)>1e-12 else np.zeros_like(x)

def topsis(frame, weights=(.35,.35,.30)):
    # Цена — затраты, рейтинг и cpu_score — выгоды.
    a = frame[['price','rating','cpu_score']].fillna(0).to_numpy(float)
    w = np.asarray(weights,float)
    if (w<0).any() or w.sum()<=0: raise ValueError('Веса должны быть неотрицательны и иметь положительную сумму')
    v = a / np.maximum(np.linalg.norm(a,axis=0),1e-12) * (w/w.sum())
    best, worst = v.max(axis=0),v.min(axis=0)
    best[0],worst[0] = worst[0],best[0]
    dp,dm = np.linalg.norm(v-best,axis=1),np.linalg.norm(v-worst,axis=1)
    return np.divide(dm,dp+dm,out=np.full(len(a),.5),where=(dp+dm)>1e-12)

class HybridRecommender:
    def __init__(self, products, interactions):
        self.p = products.reset_index(drop=True).copy()
        self.i = interactions.copy()
        self.ids = self.p.product_id.to_numpy()
        text = self.p[['category','brand','description']].fillna('').agg(' '.join,axis=1)
        tfidf = TfidfVectorizer(ngram_range=(1,2)).fit_transform(text)
        numeric = MinMaxScaler().fit_transform(self.p[['ram','storage','screen_size','cpu_score','price']].fillna(0))
        self.content = .7*cosine_similarity(tfidf)+.3*cosine_similarity(numeric)
        self.matrix = interactions.pivot(index='user_id',columns='product_id',values='rating').reindex(columns=self.ids).fillna(0)
        self.item_sim = cosine_similarity(self.matrix.to_numpy().T)
        np.fill_diagonal(self.item_sim,0)
        stats = interactions.groupby('product_id').rating.agg(['mean','count']).reindex(self.ids)
        self.pop = ((stats['mean'].fillna(3.5)*stats['count'].fillna(0)+3.5*10)/(stats['count'].fillna(0)+10)).to_numpy()/5

    def scores(self,user_id,alpha=.5,query='', neural=None):
        history = self.i[self.i.user_id==user_id]
        if history.empty:
            content,cf = self.pop.copy(),self.pop.copy()
        else:
            positions = pd.Index(self.ids).get_indexer(history.product_id)
            values = history.rating.to_numpy()
            content = np.average(self.content[:,positions],axis=1,weights=values)
            sims = self.item_sim[:,positions]
            cf = np.divide(sims@values,sims.sum(axis=1),out=5*self.pop.copy(),where=sims.sum(axis=1)>0)/5
        if query.strip():
            texts = self.p[['category','brand','description','name']].fillna('').agg(' '.join,axis=1).tolist()
            tf = TfidfVectorizer(ngram_range=(1,2)).fit_transform(texts+[query])
            content = .4*content + .6*cosine_similarity(tf[:-1],tf[-1]).ravel()
        if neural is not None and str(user_id) in neural['users']:
            cf = neural['scores'][neural['users'].index(str(user_id))]
        return pd.DataFrame({'product_id':self.ids,'content':content,'collaborative':cf,'hybrid':alpha*content+(1-alpha)*cf})

    def recommend(self,user_id,budget=300000,category=None,query='',alpha=.5,n=10,method='hybrid',weights=(.35,.35,.30),neural=None):
        if not 0<=alpha<=1: raise ValueError('alpha вне [0,1]')
        out = self.p.merge(self.scores(user_id,alpha,query,neural),on='product_id')
        seen = self.i.loc[self.i.user_id==user_id,'product_id']
        out = out[(out.price<=budget)&~out.product_id.isin(seen)]
        if category: out = out[out.category==category]
        if out.empty: return out.assign(topsis=pd.Series(dtype=float),final=pd.Series(dtype=float))
        out['topsis'] = topsis(out,weights)
        w = np.asarray(weights)/sum(weights)
        out['utility'] = w[0]*(1-scale(out.price)) + w[1]*scale(out.rating) + w[2]*scale(out.cpu_score.fillna(0))
        # Pareto dominance on price (cost), rating and performance (benefits).
        a = out[['price','rating','cpu_score']].fillna(0).to_numpy(float)
        a[:,0] *= -1
        out['pareto'] = False
        out['final'] = .8*out.hybrid+.2*out.topsis if method=='hybrid+topsis' else out[method]
        selected = out.sort_values(['final','product_id'],ascending=[False,True]).head(n).copy()
        chosen = selected[['price','rating','cpu_score']].fillna(0).to_numpy(float)
        chosen[:,0] *= -1
        selected['pareto'] = [not np.any(np.all(a>=row,axis=1)&np.any(a>row,axis=1)) for row in chosen]
        return selected
