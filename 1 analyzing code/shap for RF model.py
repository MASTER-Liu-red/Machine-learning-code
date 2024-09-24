from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(1)

data=pd.read_excel("database.xlsx")

cols = ['Speed (Hz)', 'Height (μm)', "Concentration (M)", "Wind T ($^{\circ}$C)", "Wind S (m/s)"]

model = RandomForestRegressor(
    n_estimators=429,
    min_samples_split= 2,
    min_samples_leaf= 1,
)
model.fit(data[cols], data['PCE (MAX)'].values)

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
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.25)

bar_color="orangered"
bar_alpha=0.7
plt.bar(range(len(cols)), model.feature_importances_,color=bar_color,alpha=bar_alpha)
plt.xticks(range(len(cols)), cols, rotation=-35, fontsize=14)
ax.set_ylabel('Feature importance', fontsize=14)
plt.show()
print(model.feature_importances_)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(data[cols])
print(shap_values.shape)


y_base = explainer.expected_value
print(y_base)

data['pred'] = model.predict(data[cols])
print(data['pred'].mean())


j = 10
player_explainer = pd.DataFrame()
player_explainer['feature'] = cols
player_explainer['feature_value'] = data[cols].iloc[j].values
player_explainer['shap_value'] = shap_values[j]
player_explainer

print('y_base + sum_of_shap_values: %.2f'%(y_base + player_explainer['shap_value'].sum()))
print('y_pred: %.2f'%(data['pred'].iloc[j]))

shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[j],
    data[cols].iloc[j],
    matplotlib=True
)

shap.summary_plot(shap_values, data[cols])
print(shap.summary_plot(shap_values, data[cols]))

shap.summary_plot(shap_values, data[cols], plot_type="bar",color="orangered",alpha=0.7)
print("----------------")


print()
shap.dependence_plot('Speed (Hz)',shap_values, data[cols], interaction_index=None, show=False)
shap.dependence_plot('Height (μm)',shap_values, data[cols], interaction_index=None, show=False)
shap.dependence_plot("Concentration (M)",shap_values, data[cols], interaction_index=None, show=False)
shap.dependence_plot("Wind T ($^{\circ}$C)",shap_values, data[cols], interaction_index=None, show=False)
print(shap.dependence_plot("Wind S (m/s)", shap_values, data[cols], interaction_index=None))
plt.show()
print(shap_values)



