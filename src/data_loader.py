from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def load_data(directory=None):
    directory = Path(directory or ROOT / 'data')
    p, u, i = [pd.read_csv(directory / f'{name}.csv') for name in ('products', 'users', 'interactions')]
    for df, col in ((p,'product_id'),(u,'user_id')):
        if df[col].duplicated().any():
            raise ValueError(f'Повторяющийся {col}')
        df[col] = df[col].astype(int)
    i[['user_id','product_id']] = i[['user_id','product_id']].astype(int)
    i['timestamp'] = pd.to_datetime(i.timestamp, errors='raise')
    if not i.product_id.isin(p.product_id).all() or not i.user_id.isin(u.user_id).all():
        raise ValueError('Взаимодействия с неизвестными ID')
    if (p.price <= 0).any() or not p.rating.between(0,5).all() or not i.rating.between(1,5).all():
        raise ValueError('Некорректная цена или рейтинг')
    if i.duplicated(['user_id','product_id']).any():
        raise ValueError('Повторные пары user_id/product_id; предварительно агрегируйте события')
    if 'name' not in p: p['name'] = p.brand + ' · ' + p.category + ' #' + p.product_id.astype(str)
    if 'description' not in p: p['description'] = ''
    return p,u,i
