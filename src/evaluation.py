import json
import numpy as np
import pandas as pd
from src.data_loader import load_data, ROOT
from src.recommender import HybridRecommender

def evaluate(k=10):
    p,u,i=load_data()
    ordered=i.sort_values(['timestamp','product_id'])
    test=ordered.groupby('user_id').tail(2)
    train=i.drop(test.index)
    model=HybridRecommender(p,train)
    rows=[]
    for method in ['popular','content','collaborative','hybrid','topsis','hybrid+topsis']:
        precision=[]; recall=[]; ndcg=[]; recommended=set(); eligible=0
        # Relevant = rating >= 4; no budget/category filters in this benchmark.
        for uid,holdout in test.groupby('user_id'):
            relevant=set(holdout.loc[holdout.rating>=4,'product_id'])
            if not relevant: continue
            eligible+=1
            if method=='popular':
                seen=set(train.loc[train.user_id==uid,'product_id'])
                ranking=p.assign(score=model.pop).loc[~p.product_id.isin(seen)].sort_values(['score','product_id'],ascending=[False,True]).head(k).product_id.tolist()
            else:
                ranking=model.recommend(uid,budget=float('inf'),n=k,method=method).product_id.tolist()
            hits=np.array([int(x in relevant) for x in ranking])
            precision.append(hits.sum()/k); recall.append(hits.sum()/len(relevant))
            dcg=(hits/np.log2(np.arange(len(hits))+2)).sum()
            ideal=(1/np.log2(np.arange(min(k,len(relevant)))+2)).sum()
            ndcg.append(dcg/ideal)
            recommended.update(ranking)
        rows.append({'method':method,'Precision@10':np.mean(precision),'Recall@10':np.mean(recall),'NDCG@10':np.mean(ndcg),'coverage':len(recommended)/len(p),'eligible_users':eligible})
    result=pd.DataFrame(rows)
    (ROOT/'reports').mkdir(exist_ok=True)
    result.to_csv(ROOT/'reports/evaluation.csv',index=False)
    (ROOT/'reports/split.json').write_text(json.dumps({'train':len(train),'test':len(test),'strategy':'last two events per user; relevant rating >= 4; full catalog; no budget filters','seed':42},ensure_ascii=False,indent=2))
    print(result.to_string(index=False))
    return result
if __name__=='__main__': evaluate()
