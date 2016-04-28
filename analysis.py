import pandas as pd
import import_data as id
import matplotlib.pyplot as plt
import seaborn as sns

train_file = pd.read_csv("data/train.csv")

hours_df = train_file.DateTime.apply(id.get_hour)
hours_df.append(train_file.OutcomeType)

plt.figure()

# sns.countplot(, hue=train_file.OutcomeType)
sns.countplot(train_file.AnimalType, hue=train_file.OutcomeType)
plt.show()
