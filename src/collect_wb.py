"""Read-only collection of the authorized seller's catalog, prices and feedback."""
import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime,timezone
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def collect(destination, max_pages=10):
    token=os.environ.get('WB_API_TOKEN')
    if not token: raise ValueError('Установите WB_API_TOKEN в окружении. Токен не сохраняется в проекте.')
    session=requests.Session(); session.headers['Authorization']=token
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=2,status_forcelist=[429,500,502,503,504],allowed_methods=['GET','POST'],respect_retry_after_header=True)))
    def request(method,url,**kwargs):
        response=session.request(method,url,timeout=45,**kwargs)
        if not response.ok: raise RuntimeError(f'WB API: HTTP {response.status_code}. Проверьте права токена и лимиты; ответ сервера не выводится.')
        data=response.json()
        if data.get('error'): raise RuntimeError('WB API вернул error=true')
        time.sleep(1)
        return data
    cards=[]; cursor={'limit':100}; complete_cards=False
    for _ in range(max_pages):
        data=request('POST','https://content-api.wildberries.ru/content/v2/get/cards/list',json={'settings':{'cursor':cursor,'filter':{'withPhoto':-1}}})
        batch=data.get('cards',[]); cards.extend(batch)
        if len(batch)<100: complete_cards=True; break
        c=data['cursor']; cursor={'limit':100,'updatedAt':c['updatedAt'],'nmID':c['nmID']}
    prices=[]; complete_prices=False
    for page in range(max_pages):
        data=request('GET','https://discounts-prices-api.wildberries.ru/api/v2/list/goods/filter',params={'limit':1000,'offset':page*1000})
        batch=data.get('data',{}).get('listGoods',[]); prices.extend(batch)
        if len(batch)<1000: complete_prices=True; break
    feedback=[]; feedback_complete={}
    for answered in ['true','false']:
        complete=False
        for page in range(max_pages):
            data=request('GET','https://feedbacks-api.wildberries.ru/api/v1/feedbacks',params={'isAnswered':answered,'take':1000,'skip':page*1000,'order':'dateDesc'})
            batch=data.get('data',{}).get('feedbacks',[])
            # Do not retain buyer names, photos or other personal fields.
            feedback.extend([{'id':x['id'],'product_id':x.get('productDetails',{}).get('nmId'),'rating':x.get('productValuation'),'text':x.get('text',''),'pros':x.get('pros',''),'cons':x.get('cons',''),'timestamp':x.get('createdDate')} for x in batch])
            if len(batch)<1000: complete=True; break
        feedback_complete[answered]=complete
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    for name,data in [('cards',cards),('prices',prices),('feedback',feedback)]:
        (destination/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    (destination/'manifest.json').write_text(json.dumps({'source':'WB authorized seller API','collected_at':datetime.now(timezone.utc).isoformat(),'cards_complete':complete_cards,'prices_complete':complete_prices,'feedback_complete':feedback_complete,'includes_archive':False,'max_pages':max_pages},indent=2))
    print(f'Сохранено: {len(cards)} карточек, {len(prices)} цен, {len(feedback)} отзывов. Проверьте manifest.json на полноту.')
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default='data/raw_wb'); parser.add_argument('--max-pages',type=int,default=10)
    args=parser.parse_args()
    if args.max_pages<1: parser.error('--max-pages должен быть >= 1')
    collect(args.out,args.max_pages)
