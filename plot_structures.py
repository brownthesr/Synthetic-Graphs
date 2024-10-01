import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Read the compiled data
df = pd.read_csv("compiled_maxes/structured.csv", index_col=0)

g = sns.relplot(df,x="Mu",y="Acc",kind="line",col="Transform Type",hue= "Model", style="Structured",errorbar=None)

# Set x and y labels
g.set_axis_labels("Feature Separation", "Accuracy")

# Set titles for each subplot
g.set_titles(col_template="{col_name}")

plt.savefig("here.png")