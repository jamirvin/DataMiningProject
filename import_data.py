import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
import seaborn as sb
from sklearn.metrics import classification_report, accuracy_score


class Categorize:
    def __init__(self):
        self.categories = {}
        self.counter = 0

    def enumerate(self, value):
        if value not in self.categories:
            self.categories[value] = self.counter
            self.counter += 1

        return self.categories[value]

    def num_to_value(self, num):
        for key, value in self.categories.items():
            if value == num:
                return key


cats = Categorize()


def age_in_years(age):
    age_string = str(age)
    if age_string == "nan":
        return 0

    age_num = int(age_string.split()[0])

    if age_string.find('year') > -1:
        return age_num
    if age_string.find('month') > -1:
        return age_num / 12.
    if age_string.find('week') > -1:
        return age_num / 52.
    if age_string.find('day') > -1:
        return age_num / 365.
    else:
        return 0


def get_month(time):
    dt = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
    month = dt.month
    return month


def get_weekday(time):
    dt = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
    weekday = dt.weekday()
    return weekday


def get_hour(time):
    dt = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
    hour = dt.hour
    return hour


def preprocess(data):
    data["AgeuponOutcome"] = data.AgeuponOutcome.apply(age_in_years)
    data["AnimalType"] = data.AnimalType.apply(cats.enumerate)
    data["SexuponOutcome"] = data.SexuponOutcome.apply(cats.enumerate)
    # data["Month"] = data.DateTime.apply(get_month)
    # data["Weekday"] = data.DateTime.apply(get_weekday)
    data["Hour"] = data.DateTime.apply(get_hour)
    data.drop(["Breed", "Color", "DateTime", "Name"], 1, inplace=1)

    return data


def import_training(filename: str):
    animals = pd.read_csv(filename, index_col=0)
    outcomes = animals.OutcomeType
    animals = preprocess(animals)

    animals.drop("OutcomeType", 1, inplace=True)

    print(animals)
    return animals, outcomes


def import_testing(filename: str):
    test_data = pd.read_csv(filename, index_col=0)
    test_data = preprocess(test_data)

    return test_data


def output(classifier, test_data, out_filename):
    # result = classifier.predict_proba(test_data)
    result = classifier.predict(test_data)
    # numbered_res = np.insert(result, 0, [int(i) for i in range(1, len(result)+1)], axis=1)
    np.set_printoptions(suppress=True, precision=None)
    print(result)
    # np.savetxt(out_filename, numbered_res, "%5.2f", delimiter=',', header="ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer")
    return result


def first_pass(animals, outcomes):  # -> RandomForestClassifier:
    animals.drop(["OutcomeSubtype"], 1, inplace=1)
    cats = Categorize()
    forest = RandomForestClassifier(n_jobs=-1, n_estimators=200)
    # extra = ExtraTreesClassifier(n_jobs=-1, criterion="entropy");

    forest.fit(animals, outcomes)

    results = forest.predict(animals)
    print(classification_report(outcomes, results))
    print(accuracy_score(outcomes, results))
    score = cross_val_score(forest, animals, outcomes, n_jobs=-1, cv=5)
    print("\n", np.mean(score))
    return forest


def naive(animals, outcomes):
    gnb = GaussianNB()
    gnb.fit(animals, outcomes)
    score = cross_val_score(gnb, animals, outcomes, n_jobs=-1)
    print("\n", np.mean(score))


def main(argv):
    if len(argv) != 2 and len(argv) != 4:
        print("Usage: {} <training_data.csv> [<testing_data.csv> <output_results>]".format(argv[0]))
        exit(2)

    animals, outcomes = import_training(argv[1])
    forest = first_pass(animals, outcomes)

    naive(animals, outcomes)

    if len(argv) == 4:
        test_data = import_testing(argv[2])
        result = output(forest, test_data, argv[3])
        # test_data['OutcomeType'] = pd.Series(result, index=test_data.index)
        result = pd.DataFrame({"result": result})
        test_data = test_data.join(result)
        test_data["SexuponOutcome"] = test_data.SexuponOutcome.apply(cats.num_to_value)

        print(test_data)

        sb.countplot(test_data.Hour, hue=test_data.result,
            hue_order=["Return_to_owner", "Euthanasia", "Adoption", "Transfer", "Died"])
        plt.show()


if __name__ == "__main__":
    main(sys.argv)
