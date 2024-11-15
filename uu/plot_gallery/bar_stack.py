import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
bar_plot = pd.DataFrame({'Highest Mark': [78, 70, 74, 88, 75, 72],
  'Lowest Mark': [42, 46, 40, 43, 41, 48],
  'Average Mark': [55, 58, 52, 56, 53, 50]},
  index = ['ABC', 'PQR', 'XYZ', 'MNP', 'CBD', 'BCD'])
bar_plot.plot (kind = 'bar', stacked = True, color = ['Blue', 'yellow', 'green'])
plt.xlabel ('Stud_name')
plt.ylabel ('Student marks')
plt.title ('Subject mark of student')
plt.show ()


# import libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# load dataset
tips = sns.load_dataset("tips")

# set plot style: grey grid in the background:
sns.set_theme(style="darkgrid")

# set the figure size
plt.figure(figsize=(14, 14))

# top bar -> sum all values(smoker=No and smoker=Yes) to find y position of the bars
total = tips.groupby('day')['total_bill'].sum().reset_index()

# bar chart 1 -> top bars (group of 'smoker=No')
bar1 = sns.barplot(x="day",  y="total_bill", data=total, color='darkblue')

# bottom bar ->  take only smoker=Yes values from the data
smoker = tips[tips.smoker=='Yes']

# bar chart 2 -> bottom bars (group of 'smoker=Yes')
bar2 = sns.barplot(x="day", y="total_bill", data=smoker, estimator=sum, ci=None,  color='lightblue')

# add legend
top_bar = mpatches.Patch(color='darkblue', label='smoker = No')
bottom_bar = mpatches.Patch(color='lightblue', label='smoker = Yes')
plt.legend(handles=[top_bar, bottom_bar])

# show the graph
plt.show()


print('done')