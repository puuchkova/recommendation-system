"""Train TFRS rating model and export scores for the hybrid dashboard."""
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from src.data_loader import load_data, ROOT

def main():
    tf.keras.utils.set_random_seed(42)
    p,u,i=load_data()
    # The same temporal holdout as the baseline evaluation; never train on test.
    holdout=i.sort_values(['timestamp','product_id']).groupby('user_id').tail(2)
    train=i.drop(holdout.index)
    users=u.user_id.astype(str).tolist(); products=p.product_id.astype(str).tolist()
    class RatingModel(tfrs.Model):
        def __init__(self):
            super().__init__()
            self.user=tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=users,mask_token=None),tf.keras.layers.Embedding(len(users)+1,24)])
            self.item=tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=products,mask_token=None),tf.keras.layers.Embedding(len(products)+1,24)])
            self.network=tf.keras.Sequential([tf.keras.layers.Dense(32,activation='relu'),tf.keras.layers.Dense(1)])
            self.task=tfrs.tasks.Ranking(loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.RootMeanSquaredError()])
        def call(self,features):
            return self.network(tf.concat([self.user(features['user_id']),self.item(features['product_id'])],axis=1))
        def compute_loss(self,features,training=False):
            return self.task(labels=features['rating'],predictions=tf.squeeze(self(features),axis=1))
    def dataset(frame):
        return tf.data.Dataset.from_tensor_slices({'user_id':frame.user_id.astype(str).to_numpy(),'product_id':frame.product_id.astype(str).to_numpy(),'rating':frame.rating.to_numpy(np.float32)})
    model=RatingModel(); model.compile(optimizer=tf.keras.optimizers.Adagrad(.1))
    model.fit(dataset(train).shuffle(len(train),seed=42).batch(512),epochs=15,verbose=2)
    metrics=model.evaluate(dataset(holdout).batch(512),return_dict=True,verbose=0)
    scores=np.stack([np.clip(model({'user_id':tf.constant([uid]*len(products)),'product_id':tf.constant(products)}).numpy().ravel(),1,5)/5 for uid in users])
    (ROOT/'models').mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'models/tfrs_scores.npz',users=np.array(users),products=p.product_id.to_numpy(),scores=scores)
    import json
    (ROOT/'reports/tfrs_metrics.json').write_text(json.dumps({key:float(value) for key,value in metrics.items()},indent=2))
    print(metrics)
if __name__=='__main__': main()
