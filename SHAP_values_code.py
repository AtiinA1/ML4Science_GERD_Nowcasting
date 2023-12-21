import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# X, y = ... # data loading here
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2)

# Train MLP model

# Summarize the training data
X_train_summary = shap.kmeans(X_train, 50)

# Define a function that the explainer can use
def model_predict(data):
    return model.predict(data).numpy()

# Create the SHAP KernelExplainer
explainer = shap.KernelExplainer(model_predict, X_train_summary)

# SHAP values computation
shap_values_train = explainer.shap_values(X_train)
shap_values_test = explainer.shap_values(X_test)

# SHAP Summary plot for training data
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_train, X_train, show=False)
plt.savefig("./shap_summary_train.pdf", bbox_inches='tight')

# SHAP Summary plot for test data
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_test, X_test, show=False)
plt.savefig("./shap_summary_test.pdf", bbox_inches='tight')


