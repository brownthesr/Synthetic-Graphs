import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme()
types = ['Heirarchical','Triadic','Epsilon']
df = pd.read_csv("compiled_maxes/structured.csv")
g = sns.relplot(data=df, x="Mu", y="Acc", col="Transform Type", hue="Model", style="Structured", kind="line")

# Customize the titles: "{col_name}" will represent the value of the "Transform Type" column without the prefix
g.set_titles("{col_name}")
g.set_axis_labels("Feature Separation", "Accuracy")

plt.ylim(.5,1)
# plt.tight_layout()
plt.savefig("higher_order_plots.jpg")