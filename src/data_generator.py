import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataGenerator:
    """Генератор учебных данных для рекомендательной системы"""
    
    def __init__(self, n_products=500, n_users=1000, seed=42):
        self.n_products = n_products
        self.n_users = n_users
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_products(self):
        """Генерация данных о товарах"""
        categories = ['Ноутбуки', 'Смартфоны', 'Планшеты', 'Наушники', 'Умные часы']
        brands = {
            'Ноутбуки': ['ASUS', 'Lenovo', 'HP', 'Dell', 'Apple', 'Acer'],
            'Смартфоны': ['Samsung', 'Apple', 'Xiaomi', 'Huawei', 'Google', 'OnePlus'],
            'Планшеты': ['Apple', 'Samsung', 'Lenovo', 'Huawei', 'Xiaomi'],
            'Наушники': ['Sony', 'Bose', 'Apple', 'Sennheiser', 'JBL', 'Beats'],
            'Умные часы': ['Apple', 'Samsung', 'Garmin', 'Fitbit', 'Huawei']
        }
        
        products = []
        product_id = 1
        
        for category in categories:
            n_in_category = self.n_products // len(categories)
            
            for _ in range(n_in_category):
                brand = random.choice(brands[category])
                
                # Генерация характеристик в зависимости от категории
                if category == 'Ноутбуки':
                    price = np.random.normal(80000, 25000)
                    ram = random.choice([8, 16, 32])
                    storage = random.choice([256, 512, 1024, 2048])
                    screen_size = random.choice([13.3, 14, 15.6, 17])
                    cpu_score = np.random.randint(50, 100)
                    
                elif category == 'Смартфоны':
                    price = np.random.normal(50000, 20000)
                    ram = random.choice([4, 6, 8, 12])
                    storage = random.choice([64, 128, 256, 512])
                    screen_size = random.choice([5.5, 6.1, 6.5, 6.7])
                    cpu_score = np.random.randint(60, 100)
                    
                elif category == 'Планшеты':
                    price = np.random.normal(40000, 15000)
                    ram = random.choice([4, 6, 8])
                    storage = random.choice([64, 128, 256])
                    screen_size = random.choice([8, 10.2, 11, 12.9])
                    cpu_score = np.random.randint(50, 90)
                    
                elif category == 'Наушники':
                    price = np.random.normal(15000, 8000)
                    ram = None
                    storage = None
                    screen_size = None
                    cpu_score = np.random.randint(40, 100)
                    
                else:  # Умные часы
                    price = np.random.normal(25000, 10000)
                    ram = None
                    storage = random.choice([4, 8, 16])
                    screen_size = random.choice([1.2, 1.4, 1.5, 1.9])
                    cpu_score = np.random.randint(40, 80)
                
                price = max(5000, min(200000, price))  # Ограничение цены
                
                products.append({
                    'product_id': product_id,
                    'category': category,
                    'brand': brand,
                    'price': round(price, 2),
                    'ram': ram,
                    'storage': storage,
                    'screen_size': screen_size,
                    'cpu_score': cpu_score,
                    'rating': round(np.random.uniform(3.5, 5.0), 1),
                    'reviews_count': np.random.randint(10, 5000)
                })
                product_id += 1
        
        return pd.DataFrame(products)
    
    def generate_users(self):
        """Генерация данных о пользователях"""
        users = []
        
        for user_id in range(1, self.n_users + 1):
            age = np.random.randint(18, 65)
            budget = np.random.lognormal(mean=10.5, sigma=0.8)  # Логнормальное распределение
            budget = min(300000, max(10000, budget))
            
            # Предпочтения по категориям (веса)
            preferences = {
                'Ноутбуки': np.random.random(),
                'Смартфоны': np.random.random(),
                'Планшеты': np.random.random(),
                'Наушники': np.random.random(),
                'Умные часы': np.random.random()
            }
            
            # Нормализация предпочтений
            total = sum(preferences.values())
            preferences = {k: v/total for k, v in preferences.items()}
            
            users.append({
                'user_id': user_id,
                'age': age,
                'budget': round(budget, 2),
                'pref_notebooks': preferences['Ноутбуки'],
                'pref_smartphones': preferences['Смартфоны'],
                'pref_tablets': preferences['Планшеты'],
                'pref_headphones': preferences['Наушники'],
                'pref_watches': preferences['Умные часы']
            })
        
        return pd.DataFrame(users)
    
    def generate_interactions(self, products_df, users_df):
        """Генерация взаимодействий пользователей с товарами"""
        interactions = []
        
        # Маппинг русских категорий на английские названия колонок
        category_to_col = {
            'Ноутбуки': 'pref_notebooks',
            'Смартфоны': 'pref_smartphones',
            'Планшеты': 'pref_tablets',
            'Наушники': 'pref_headphones',
            'Умные часы': 'pref_watches'
        }
        
        for _, user in users_df.iterrows():
            n_interactions = np.random.randint(5, 21)
            
            available_products = products_df.copy()
            
            weights = []
            for _, product in available_products.iterrows():
                col_name = category_to_col[product["category"]]
                pref_weight = user[col_name]
                budget_factor = 1 if product['price'] <= user['budget'] else 0.3
                weights.append(pref_weight * budget_factor)
            
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            selected_products = np.random.choice(
                available_products['product_id'].values,
                size=min(n_interactions, len(available_products)),
                replace=False,
                p=weights
            )
            
            for product_id in selected_products:
                product = products_df[products_df['product_id'] == product_id].iloc[0]
                col_name = category_to_col[product["category"]]
                
                base_rating = product['rating']
                pref_match = user[col_name]
                budget_match = 1 if product['price'] <= user['budget'] else 0.7
                
                rating = base_rating * (0.7 + 0.3 * pref_match) * budget_match
                rating = min(5.0, max(1.0, rating + np.random.normal(0, 0.5)))
                
                interactions.append({
                    'user_id': user['user_id'],
                    'product_id': product_id,
                    'rating': round(rating, 1),
                    'timestamp': datetime(2026, 1, 1) - timedelta(days=np.random.randint(0, 365))
                })
        
        return pd.DataFrame(interactions)
    
    def generate_all_data(self):
        """Генерация всех данных"""
        products_df = self.generate_products()
        users_df = self.generate_users()
        interactions_df = self.generate_interactions(products_df, users_df)
        
        return products_df, users_df, interactions_df


def main():
    """Основная функция для генерации и сохранения данных"""
    print("Генерация данных...")
    generator = DataGenerator(n_products=500, n_users=1000)
    products_df, users_df, interactions_df = generator.generate_all_data()
    
    # Сохранение в CSV
    products_df.to_csv('data/products.csv', index=False)
    users_df.to_csv('data/users.csv', index=False)
    interactions_df.to_csv('data/interactions.csv', index=False)
    
    print(f"✓ Сгенерировано {len(products_df)} товаров")
    print(f"✓ Сгенерировано {len(users_df)} пользователей")
    print(f"✓ Сгенерировано {len(interactions_df)} взаимодействий")
    print("\nДанные сохранены в папку 'data/'")
    
    # Вывод статистики
    print("\nСтатистика товаров:")
    print(products_df['category'].value_counts())
    print(f"\nСредняя цена: {products_df['price'].mean():.2f} руб.")
    
    print("\nСтатистика пользователей:")
    print(f"Средний возраст: {users_df['age'].mean():.1f} лет")
    print(f"Средний бюджет: {users_df['budget'].mean():.2f} руб.")


if __name__ == "__main__":
    main()
