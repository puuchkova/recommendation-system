import numpy as np
import streamlit as st
import plotly.express as px
from src.data_loader import load_data, ROOT
from src.clustering import cluster_users
from src.recommender import HybridRecommender
st.set_page_config(page_title='Выбор товара',layout='wide')
st.title('Выбор товара: рекомендательная система')
st.warning('Демонстрационные товары, цены, характеристики и истории сгенерированы. Это не актуальный каталог маркетплейса. cpu_score — условная учебная оценка.')
@st.cache_resource
def setup():
    p,u,i=load_data()
    return p,u,i,HybridRecommender(p,i),cluster_users(p,u,i)
p,u,i,model,(clusters,silhouette)=setup()
with st.sidebar:
    st.header('Потребности и бюджет')
    uid=st.selectbox('Пользователь (0 — новый)',[0]+u.user_id.tolist())
    default=int(u.loc[u.user_id==uid,'budget'].iloc[0]) if uid else 50000
    budget=st.number_input('Бюджет, ₽',min_value=0,value=default,step=1000)
    category=st.selectbox('Категория',['Все']+sorted(p.category.unique()))
    query=st.text_input('Что важно? Категория, бренд, описание','')
    alpha=st.slider('Вес контентной модели',0.,1.,.5)
    method=st.selectbox('Метод',['hybrid+topsis','hybrid','content','collaborative','topsis','utility'])
    n=st.slider('Количество рекомендаций',1,30,10)
    st.caption('Веса TOPSIS: цена / рейтинг / условная производительность')
    weights=[st.slider(label,0.,1.,v,key=label) for label,v in [('Цена',.35),('Рейтинг',.35),('Производительность',.30)]]
    backend=st.selectbox('Коллаборативная модель',['Item-based CF','TensorFlow Recommenders'])
if sum(weights)==0: st.error('Задайте хотя бы один положительный вес'); st.stop()
neural=None
if backend=='TensorFlow Recommenders':
    path=ROOT/'models/tfrs_scores.npz'
    if not path.exists(): st.error('Сначала выполните python -m src.tfrs_model'); st.stop()
    blob=np.load(path)
    if not np.array_equal(blob['products'],p.product_id.to_numpy()): st.error('Каталог изменён: переобучите TFRS'); st.stop()
    neural={'users':blob['users'].tolist(),'scores':blob['scores']}
tabs=st.tabs(['Рекомендации','Покупательские сегменты','Данные и качество'])
with tabs[0]:
    out=model.recommend(uid,budget,None if category=='Все' else category,query,alpha,n,method,weights,neural)
    if uid==0: st.info('Холодный старт: сглаженные оценки популярности + ваши потребности. Истории нового пользователя нет.')
    if out.empty: st.info('Товаров с такими ограничениями нет. Измените бюджет или категорию.')
    else:
        st.dataframe(out[['name','category','price','rating','cpu_score','content','collaborative','hybrid','topsis','utility','pareto','final']],hide_index=True)
        st.caption('Итог hybrid+topsis = 0,8 × hybrid + 0,2 × TOPSIS; hybrid = α × content + (1−α) × CF. Оценки — баллы ранжирования, не вероятности покупки.')
        st.plotly_chart(px.scatter(out,x='price',y='final',color='category',hover_name='name'),use_container_width=True)
        for row in out.head(3).itertuples(): st.write(f'{row.name}: цена {row.price:,.0f} ₽ ≤ бюджет {budget:,.0f} ₽; контент {row.content:.3f}; CF {row.collaborative:.3f}; TOPSIS {row.topsis:.3f}.')
        st.download_button('Скачать рекомендации CSV',out.to_csv(index=False).encode('utf-8-sig'),'recommendations.csv')
with tabs[1]:
    st.metric('Silhouette',f'{silhouette:.3f}' if silhouette is not None else 'н/д')
    st.plotly_chart(px.scatter(clusters,x='mean_price',y='events',color=clusters.cluster.astype(str),hover_data=['user_id']),use_container_width=True)
    st.dataframe(clusters.groupby('cluster').mean(numeric_only=True).drop(columns='user_id'),use_container_width=True)
    st.caption('K-means: частота взаимодействий, средняя цена просмотренных/оценённых товаров, средняя оценка и доли категорий. Это поведение, а не подтверждённые покупки.')
with tabs[2]:
    st.write(f'{len(p)} товаров · {len(u)} пользователей · {len(i)} оценок')
    st.plotly_chart(px.histogram(p,x='price',color='category'),use_container_width=True)
    path=ROOT/'reports/evaluation.csv'
    if path.exists():
        import pandas as pd
        st.dataframe(pd.read_csv(path),hide_index=True)
    else: st.info('Для расчёта метрик: python -m src.evaluation')
    st.caption('Оценка на временном отложенном наборе; подробности и ограничения — в отчёте.')
