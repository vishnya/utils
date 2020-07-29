# TRAIN TEST SPLIT


def split_df_train_test(df, label):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, ~df.columns.isin([label])],
                                                        df[label])
    return X_train, X_test, y_train, y_test


def split_df_train_test_dt(df, start_date, split_date, end_date):
    """
    e.g. '01-01-2020'
    Given a dataframe, a start date, a split date, and an end date, split the dataframe
    into test and training dataframes.
    """
    import pandas as pd

    def get_date(date_string):
        return pd.to_datetime(date_string).date()

    df.set_index("date", inplace=True)

    start_date, split_date, end_date = get_date(start_date), get_date(split_date), get_date(end_date)
    assert (start_date < split_date) & (split_date < end_date), "Incorrectly formatted dates"
    df_train = df.loc[(df.index > start_date) & (df.index <= split_date)].copy()
    df_test = df.loc[(df.index > split_date) & (df.index <= end_date)].copy()

    df_train.reset_index(inplace=True)
    df_test.reset_index(inplace=True)

    return df_train, df_test


# MODELING

def train_xgb_model(X_train, y_train, X_test, y_test,
                    n_estimators=1000, objective='reg:squarederror', early_stopping_rounds=50):
    """
    Trains an XGBoost model with the default settings
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    """
    import xgboost as xgb
    reg = xgb.XGBRegressor(n_estimators=n_estimators, objective=objective)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False)
    return reg


def get_pred_scores(model, X_test):
    return model.predict(X_test)


def train_log_reg_model(X_train, y_train, X_test):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    mdl = model.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    return model, y_pred


# PERFORMANCE


def performance_metrics_regression(y_test, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, \
        r2_score, median_absolute_error
    """
    Prints performance metrics.
    First, prints the R^2 score.
    """
    print("*" * 70)
    print("PERFORMANCE METRICS")
    r2score = round(r2_score(y_test, y_pred), 4)
    print(f"The R2 score is {r2score}")

    mae = round(median_absolute_error(y_test, y_pred), 4)
    print(f"The median absolute error is {mae}")
    meansqe = round(mean_squared_error(y_test, y_pred), 4)
    print(f"The mean squared error is {meansqe}")

    meanae = round(mean_absolute_error(y_test, y_pred), 4)
    print(f"The mean absolute error is {meanae}")


def performance_metrics_classification(y_test, y_pred_score, threshold=.5):
    from sklearn.metrics import classification_report, roc_auc_score
    y_pred = (y_pred_score > threshold).astype(int)

    print("*" * 70)
    print(f'CLASSIFICATION REPORT FOR THRESHOLD {threshold}')
    print(classification_report(y_test, y_pred))

    roc_auc = round(roc_auc_score(y_test, y_pred), 4)
    print(f"The ROC_AUC is: {roc_auc}")


def print_confusion_matrix(y_test, y_pred_score, threshold, class_names=[0,1]):
    import itertools
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import numpy as np
    y_pred = (y_pred_score > threshold).astype(int)
    matrix = confusion_matrix(y_test, y_pred)
    plt.clf()

    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    fmt = 'd'

    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label', size=14)
    plt.xlabel('Predicted label', size=14)
    plt.show()


def plot_roc_curve(y_test, y_pred):
    """
    y_pred: e.g. logreg.predict(X_test)
    """
    print("*" * 70)
    print(f'PLOTTING ROC CURVE')

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


# FEATURE IMPORTANCES


def plot_feature_importances_xgb(model):
    from xgboost import plot_importance
    _ = plot_importance(model, height=0.9)


def plot_feature_importances_logreg(model, X_train):
    import matplotlib.pyplot as plt
    import seaborn as sns
    f = plt.figure(figsize=(6, 6))
    gs = f.add_gridspec(2, 2)
    importance = model.coef_
    for i, v in zip(X_train.columns.values, importance[0]):
        print(f'Feature: {i}, Score: {v}')
    sns.barplot(y=abs(importance[0]), x=X_train.columns.values)
    plt.xticks(rotation=45)
    plt.title('Feature Importances (abs)')
    plt.show()
