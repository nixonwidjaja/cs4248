# !/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2 Template

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# TODO: Replace with your Student Number
_STUDENT_NUM = "A0236430N"


def build_model(model):
    print(model)
    f1_macro = make_scorer(f1_score, average="macro")

    if model == "NB":
        parameters = {
            "alpha": [0.01, 0.1, 1, 10],
            "fit_prior": [True, False],
        }
        return GridSearchCV(
            MultinomialNB(), parameters, cv=5, scoring=f1_macro, n_jobs=-1
        )
        # return MultinomialNB()
    elif model == "LR":
        param_grid = {
            "C": [1, 10, 100],
        }
        logreg = LogisticRegression(
            max_iter=300,
            solver="liblinear",
            penalty="l2",
            class_weight="balanced",
        )
        # return logreg
        return GridSearchCV(logreg, param_grid, cv=5, scoring=f1_macro, n_jobs=-1)
    elif model == "NN":
        param_grid = {"hidden_layer_sizes": [(100,), (50, 100, 50), (50, 50), (200,)]}
        classifier = MLPClassifier(
            max_iter=200,
            activation="relu",
            solver="adam",
            random_state=1,
            early_stopping=True,
        )
        return GridSearchCV(classifier, param_grid, cv=5, scoring=f1_macro, n_jobs=-1)
    elif model == "GBC":
        parameters = {
            "learning_rate": [1, 0.1, 0.01],
        }
        return GridSearchCV(
            GradientBoostingClassifier(),
            parameters,
            cv=5,
            scoring=f1_macro,
            n_jobs=-1,
        )


def preprocessing(text: str):
    # stops = set(stopwords.words("english"))
    wnl = WordNetLemmatizer()

    text = word_tokenize(text.lower())
    cleaned = [w for w in text if w.isalpha()]
    lemmatized = [wnl.lemmatize(w) for w in cleaned]
    return " ".join(lemmatized)


def train_model(model, X_train, y_train):
    """TODO: train your model based on the training data"""
    model.fit(X_train, y_train)


def predict(model, X_test):
    """TODO: make your prediction here"""
    return model.predict(X_test)


def generate_result(test, y_pred, filename):
    """generate csv file base on the y_pred"""
    test["Verdict"] = pd.Series(y_pred)
    test.drop(columns=["Text"], inplace=True)
    test.to_csv(filename, index=False)


def upsample(train):
    max_rows = train["Verdict"].value_counts().max()
    res = pd.DataFrame()
    for _ in range(max_rows // (train["Verdict"] == 1).sum()):
        res = pd.concat([res, train[train["Verdict"] == 1]])
    for _ in range(max_rows // (train["Verdict"] == 0).sum()):
        res = pd.concat([res, train[train["Verdict"] == 0]])
    return pd.concat([res, train])


def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def glove(X_train):
    glove_embeddings = load_embeddings("glove.6B/glove.6B.300d.txt")
    X_train_glove = []
    for text in X_train:
        words = preprocessing(text).split()
        sum = np.zeros((300,))
        for w in words:
            sum += glove_embeddings.get(
                w,
                np.zeros(
                    300,
                ),
            )
        sum /= len(words)
        X_train_glove.append(sum)
    return X_train_glove


def main(do_upsample=True, feature="glove", model_name="NN"):
    """load train, val, and test data"""
    print(do_upsample, feature, model_name)
    train = pd.read_csv("train.csv")

    # print((train["Verdict"] == 1).sum())  # 5413
    # print((train["Verdict"] == 0).sum())  # 2403
    # print((train["Verdict"] == -1).sum())  # 14685

    if do_upsample:
        train = upsample(train)
        # print((train["Verdict"] == 1).sum())  # 16239
        # print((train["Verdict"] == 0).sum())  # 16821
        # print((train["Verdict"] == -1).sum())  # 14685

    X_train = train["Text"]
    y_train = train["Verdict"]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    test = pd.read_csv("test.csv")
    X_test = test["Text"]

    model = build_model(model_name)  # TODO: Define your model here

    if feature == "cv":
        count_v = CountVectorizer(
            preprocessor=preprocessing, ngram_range=(1, 2), max_features=10000
        )
        X_train = count_v.fit_transform(X_train).toarray()
        print(X_train.shape)
        X_val = count_v.transform(X_val)
        X_test = count_v.transform(X_test)
    elif feature == "tfidf":
        tfidf = TfidfVectorizer(
            preprocessor=preprocessing,
            use_idf=True,
            smooth_idf=True,
            ngram_range=(1, 2),
        )
        X_train = tfidf.fit_transform(X_train)
        X_val = tfidf.transform(X_val)
        X_test = tfidf.transform(X_test)
    elif feature == "glove":
        X_train = glove(X_train)
        X_val = glove(X_val)
        X_test = glove(X_test)

    train_model(model, X_train, y_train)
    print(model.best_estimator_)
    print(model.best_params_)

    # test your model
    # Use f1-macro as the metric

    y_pred = predict(model, X_train)
    score = f1_score(y_train, y_pred, average="macro")
    print("score on training = {}".format(score))

    y_pred = predict(model, X_val)
    score = f1_score(y_val, y_pred, average="macro")
    print("score on validation = {}".format(score))

    # generate prediction on test data
    y_pred = predict(model, X_test)
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")


# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main(do_upsample=True, feature="cv", model_name="LR")
