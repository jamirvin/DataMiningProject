import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score


class Categorize:
    def __init__(self):
        self.categories = {}
        self.counter = 0

    def enumerate(self, value):
        if value not in self.categories:
            self.categories[value] = self.counter
            self.counter += 1

        return self.categories[value]


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


def import_array(filename: str):
    cats = Categorize()

    animals = pd.read_csv(filename, index_col=0)
    animals["AgeuponOutcome"] = animals.AgeuponOutcome.apply(age_in_years)
    animals["AnimalType"] = animals.AnimalType.apply(cats.enumerate)
    animals["SexuponOutcome"] = animals.SexuponOutcome.apply(cats.enumerate)
    outcomes = animals.OutcomeType

    animals.drop("OutcomeType", 1, inplace=True)

    print(animals)
    return animals, outcomes


def first_pass(animals, outcomes) -> RandomForestClassifier:
    animals.drop(["Breed", "Color", "DateTime", "Name", "OutcomeSubtype"], 1, inplace=1)
    forrest = RandomForestClassifier(n_jobs=-1, n_estimators=100)
    extra = ExtraTreesClassifier(n_jobs=-1, criterion="entropy")

    score = cross_val_score(extra, animals, outcomes, n_jobs=-1, cv=5)
    print("\n", np.mean(score))
    return forrest


def main(argv):
    if len(argv) != 2:
        print("Usage: {} <data.csv>".format(argv[0]))
        exit(2)

    animals, outcomes = import_array(argv[1])
    forrest = first_pass(animals, outcomes)


if __name__ == "__main__":
    main(sys.argv)
