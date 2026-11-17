import json
import pandas as pd
from src.normalize_wb import normalize
from src.collect_wb import collect
import pytest

def test_missing_token(monkeypatch,tmp_path):
    monkeypatch.delenv('WB_API_TOKEN',raising=False)
    with pytest.raises(ValueError): collect(tmp_path)

def test_normalize_fixture(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    payloads={'cards':[{'nmID':12,'title':'Тест','subjectName':'Ноутбуки','brand':'A','characteristics':[{'name':'RAM','value':['16 ГБ']}]}], 'prices':[{'nmID':12,'currencyIsoCode4217':'RUB','sizes':[{'discountedPrice':50000}]}], 'feedback':[{'id':'x','product_id':12,'rating':5,'text':'Хорошо','timestamp':'2026-01-01'}]}
    for key,value in payloads.items(): (raw/f'{key}.json').write_text(json.dumps(value))
    normalize(raw,tmp_path/'out')
    p=pd.read_csv(tmp_path/'out/products.csv')
    assert p.price.iloc[0]==50000 and p.rating.iloc[0]==5
    assert pd.isna(p.cpu_score.iloc[0]) and 'RAM' in p.description.iloc[0]
