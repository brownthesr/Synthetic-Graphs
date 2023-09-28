"""This plots all of the architectures on the given benchmark datasets
    """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd

# Set up Parameters and figure
type_ = ["features","edges","both"]
models = ["gcn","sage","gat"]
sns.set_theme()
palette = sns.color_palette("colorblind")

df = pd.DataFrame(pd.read_csv("data/GCN_scramble/scramble_all.txt"))
print(df)
plot = sns.relplot(data=df,x = "Original Accuracy", y = "Modified Accuracy",hue="Transform Type",col="Model")
plot.set_titles("{col_name}")
sns.lineplot(x=[0,1],y=[0,1],ax = plot.axes[0,0],color="grey",alpha=0.5,linestyle="--")
sns.lineplot(x=[0,1],y=[0,1],ax = plot.axes[0,1],color="grey",alpha=0.5,linestyle="--")
sns.lineplot(x=[0,1],y=[0,1],ax = plot.axes[0,2],color="grey",alpha=0.5,linestyle="--")
# plt.xlim(-0.01,1.01)
# plt.ylim(-.01,1.01)
plt.show()

