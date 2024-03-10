# !/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2 Template

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# TODO: Replace with your Student Number
_STUDENT_NUM = "A0236430N"


def build_model(model):
    print(model)
    f1_macro = make_scorer(f1_score, average="macro")
    if model == "NB":
        parameters = {
            "alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
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
            max_iter=100,
            solver="liblinear",
            penalty="l2",
            class_weight="balanced",
        )
        # return logreg
        return GridSearchCV(logreg, param_grid, cv=5, scoring=f1_macro, n_jobs=-1)
    elif model == "NN":
        param_grid = {"hidden_layer_sizes": [(50, 50)]}
        classifier = MLPClassifier(
            max_iter=200,
            activation="relu",
            solver="adam",
            random_state=1,
            early_stopping=True,
        )
        return GridSearchCV(classifier, param_grid, cv=5, scoring=f1_macro, n_jobs=-1)


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


def main():
    """load train, val, and test data"""
    train = pd.read_csv("train.csv")
    X_train = train["Text"]
    y_train = train["Verdict"]

    test = pd.read_csv("test.csv")
    X_test = test["Text"]

    model = build_model("NN")  # TODO: Define your model here

    # count_v = CountVectorizer(preprocessor=preprocessing, ngram_range=(1, 2))
    # X_train = count_v.fit_transform(X_train).toarray()
    # X_test = count_v.transform(X_test)

    tfidf = TfidfVectorizer(
        preprocessor=preprocessing,
        use_idf=True,
        smooth_idf=True,
        ngram_range=(1, 2),
    )
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    # import spacy

    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(text)

    train_model(model, X_train, y_train)
    print(model.best_estimator_)
    print(model.best_params_)
    # test your model
    y_pred = predict(model, X_train)

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average="macro")
    print("score on validation = {}".format(score))

    # generate prediction on test data
    y_pred = predict(model, X_test)
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")


# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
