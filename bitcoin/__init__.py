def predict():
    pass


def train():
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split

    prices = pd.read_csv('prices.csv')
    X = np.arange(len(prices)).reshape(-1, 1)
    y = np.array(prices['close'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree_reg = DecisionTreeRegressor(max_depth=3)
    tree_reg.fit(X_train, y_train)

    # save the model to disk
    pickle.dump(tree_reg, open("model.sav", 'wb+'))
