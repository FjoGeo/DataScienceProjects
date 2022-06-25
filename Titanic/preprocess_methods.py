import pandas as pd
from sklearn.linear_model import LinearRegression

def preprocess_train():
    """
    Preprocess the training data.

    Returns:
        x_train: the training features.
        y_train: the training data's target.
    """

    training_df = pd.read_csv('./Data/train.csv')

    # replace NaN data with zeros
    training_df['Embarked'].fillna(value=0, inplace=True)
    
    # drop the columns we don't need
    training_df = training_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # convert categorical columns to one-hot encoded columns
    training_df['Sex'] = training_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    training_df['Embarked'] = training_df['Embarked'].map( {'S': 1, 'C': 2, 'Q':3, 0:0} ).astype(int)

    # use scikit to predict missing age data
    x = training_df[training_df['Age'] > 0]
    y = x['Age']
    
    clf = LinearRegression(positive=True)
    clf.fit(x.drop(['Age'], axis=1), y)
    x_test = training_df[training_df['Age'].isnull()]
    x_test['Age'] = clf.predict(x_test.drop(['Age'], axis=1))

    # concat x_test and x
    training_df = pd.concat([x, x_test])
    
    x_train = training_df.drop(['Survived'], axis=1)
    y_train = training_df['Survived']

    return x_train, y_train

def preprocess_test():
    """ Preprocess the test data.

    Returns:
        test_df: the test features.
    """
    test_df = pd.read_csv('./Data/test.csv')

    # replace NaN data with zeros
    test_df['Embarked'].fillna(value=0, inplace=True)
    test_df['Fare'].fillna(value=0, inplace=True)
    
    # drop the columns we don't need
    test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # convert categorical columns to one-hot encoded columns
    test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    test_df['Embarked'] = test_df['Embarked'].map( {'S': 1, 'C': 2, 'Q':3, 0:0} ).astype(int)

    # use scikit to predict missing age data
    x = test_df[test_df['Age'] > 0]
    y = x['Age']
    clf = LinearRegression(positive=True)
    clf.fit(x.drop(['Age'], axis=1), y)
    x_test = test_df[test_df['Age'].isnull()]
    x_test['Age'] = clf.predict(x_test.drop(['Age'], axis=1))

    # concat x_test and x
    test_df = pd.concat([x, x_test])

    return test_df

