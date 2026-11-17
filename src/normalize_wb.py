import argparse
import json
from pathlib import Path
import pandas as pd

def normalize(raw,out):
    raw,out=Path(raw),Path(out); out.mkdir(parents=True,exist_ok=True)
    cards=json.loads((raw/'cards.json').read_text()); prices=json.loads((raw/'prices.json').read_text()); reviews=json.loads((raw/'feedback.json').read_text())
    price_map={x['nmID']:min([s['discountedPrice'] for s in x.get('sizes',[]) if s.get('discountedPrice',0)>0],default=None) for x in prices if x.get('currencyIsoCode4217','RUB')=='RUB'}
    rf=pd.DataFrame(reviews,columns=['id','product_id','rating','text','pros','cons','timestamp']).drop_duplicates('id')
    stats=rf.groupby('product_id').rating.agg(['mean','count'])
    rows=[]
    for card in cards:
        pid=card['nmID']; price=price_map.get(pid)
        if price is None: continue
        chars=card.get('characteristics',[])
        # Preserve vendor characteristics verbatim. Do not invent CPU benchmarks.
        description=card.get('description','')+' '+json.dumps(chars,ensure_ascii=False)
        review_text=' '.join(rf.loc[rf.product_id==pid,'text'].fillna('').astype(str).head(10).tolist())
        description += ' '+review_text
        rows.append({'product_id':pid,'name':card.get('title',str(pid)),'category':card.get('subjectName','Другое'),'brand':card.get('brand',''),'price':price,'rating':float(stats.loc[pid,'mean']) if pid in stats.index else 0,'reviews_count':int(stats.loc[pid,'count']) if pid in stats.index else 0,'description':description,'ram':None,'storage':None,'screen_size':None,'cpu_score':None,'source':'WB seller API'})
    pd.DataFrame(rows).to_csv(out/'products.csv',index=False)
    rf.to_csv(out/'reviews.csv',index=False)
    print('Каталог и отзывы сохранены. Для персональной CF нужны отдельные обезличенные users.csv и interactions.csv: API отзывов не предоставляет историю покупок одного пользователя.')
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--raw',default='data/raw_wb'); parser.add_argument('--out',default='data/real_wb'); args=parser.parse_args(); normalize(args.raw,args.out)
