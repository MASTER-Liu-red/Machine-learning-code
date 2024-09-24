import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


np.random.seed(1)

data=pd.read_excel("Pearson.xlsx")

pearson_corr = np.corrcoef(data, rowvar=False)  
print(pearson_corr)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.direction"] ="in"
plt.rcParams["ytick.direction"] ="in"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

w = 10
h = w / 4.85
plt.figure(figsize = (w,h),dpi = 300)
fig,ax=plt.subplots()
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

plt.figure(figsize=(8, 6))
plt.imshow(pearson_corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Pearson Correlation Coefficient')
plt.title('Pearson Correlation Matrix')
plt.xticks(np.arange(5), ['Speed (Hz)', 'Height (μm)', 'Concentration (M)', 'Wind T ($^{\circ}$C)', 'Wind S (m/s)'],rotation=-30)
plt.yticks(np.arange(5),['Speed (Hz)', 'Height (μm)', 'Concentration (M)', 'Wind T ($^{\circ}$C)', 'Wind S (m/s)'])

for i in range(pearson_corr.shape[0]):
    for j in range(pearson_corr.shape[1]):
        plt.text(j , i , '{:.2f}'.format(pearson_corr[i, j]), ha='center', va='center', color='black')

plt.show()

