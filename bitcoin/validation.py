from .db import get_all_data
from .log import logger
from .predition import Prediction
from .train import train, train_scaler
import pandas as pd


def test_computed(columns, df_test=None):
    df_test = df_test.reset_index()
    count_test = df_test['open'].count()
    logger.info('Test set of %d items !' % count_test)
    p = Prediction()
    df_computed = pd.DataFrame(columns=['real', 'predicted', 'diff'])
    for index, row in df_test.iterrows():
        close = row['close']

        X = []
        for c in columns:
            X.append(row[c])

        y_predict = p.predict(X, load_model=(index == 0))

        df_computed = df_computed.append({
            'real': close,
            'predicted': y_predict,
            'diff': close - y_predict
        }, ignore_index=True)

    regul = df_computed['diff'].median()
    df_computed['predicted_regul'] = df_computed['predicted'] + regul
    df_computed['diff_regul'] = df_computed['real'] - (df_computed['predicted'] + regul)

    logger.info('Done !')
    return df_computed, regul


def test_model():
    import numpy as np

    df = get_all_data()
    df_train, df_test = np.split(df.sample(frac=1), [int(.8 * len(df))])
    train_scaler(df=df)
    y = df_train[['close']].values.reshape(-1, 1)
    columns = ['open']
    history = train(df_train[columns].values, y)

    return test_computed(columns, df_test=df_test), history
