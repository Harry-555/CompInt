import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import vis

# Data Collection
cols = [
    "sample_code", "clump_thickness", "uniform_cell_size", "uniform_cell_shape",
    "marginal_adhesion", "single_epithelial_size", "bare_nuclei",
    "bland_chromatin", "normal_nucleoli", "mitoses", "class"
]
data = pd.read_csv("Wisconsin_Breast_Cancer_Dataset/breast-cancer-wisconsin.data", header=None, names=cols)

#Data Exploration & Analysis
print(data.info())
print(data.describe().transpose())
non_numeric_counts = {}
for col in cols[1:]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    non_numeric = data[col].isna().sum()
    if non_numeric > 0:
        non_numeric_counts[col] = non_numeric
print(non_numeric_counts)

#Data Cleaning
data = data.drop_duplicates()
data = data.drop("sample_code", axis=1)
for col, count in non_numeric_counts.items():
    if count > 0:
        feature_mean = data[col].mean()
        data[col] = data[col].fillna(feature_mean)
        print(f"Filled {count} missing values in '{col}' with mean: {feature_mean:.2f}")
data['class'] = data['class'].map({2: 0, 4: 1}  )


#Data Splitting
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

#Data Preprocessing
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(pd.DataFrame(X_train, columns=X.columns).describe().transpose())

mlp = MLPClassifier(hidden_layer_sizes=(5,2),
                    max_iter=2000,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=50,
                    tol=1e-4
                    )

mlp.fit(X_train, y_train)

print(f"\nMLP finished training after {mlp.n_iter_} iterations (Max was 2000).")
if mlp.n_iter_ < 2000:
    print("Early stopping was applied.\n")

predictions = mlp.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print()

def visualise(mlp):
    n_neurons = [len(layer) for layer in mlp.coefs_]
    n_neurons.append(mlp.n_outputs_)

    y_range = [0, max(n_neurons)]
    x_range = [0, len(n_neurons)]
    loc_neurons = [[[l, (n + 1) * (y_range[1] / (layer + 1))] for n in range(layer)] for l, layer in
                   enumerate(n_neurons)]
    x_neurons = [x for layer in loc_neurons for x, y in layer]
    y_neurons = [y for layer in loc_neurons for x, y in layer]

    weight_range = [min([layer.min() for layer in mlp.coefs_]), max([layer.max() for layer in mlp.coefs_])]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x_neurons, y_neurons, s=100, zorder=5, color='black')

    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Positive Weight (Excitatory)'),
        Line2D([0], [0], color='blue', lw=2, label='Negative Weight (Inhibitory)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Neuron')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    for l, layer in enumerate(mlp.coefs_):
        for i, neuron in enumerate(layer):
            for j, w in enumerate(neuron):
                color = 'red' if w > 0 else 'blue'

                normalized_weight = (abs(w) - 0) / (max(abs(weight_range[0]), abs(weight_range[1])) - 0)
                linewidth = normalized_weight * 4 + 0.2

                ax.plot([loc_neurons[l][i][0], loc_neurons[l + 1][j][0]],
                        [loc_neurons[l][i][1], loc_neurons[l + 1][j][1]],
                        color=color, linewidth=linewidth, alpha=0.6)

    plt.title("ANN Structure Visualization (Red=Positive, Blue=Negative)")
    # Turn off axis numbers as they don't mean anything here
    ax.axis('off')
    plt.show()

visualise(mlp)

"""
Same dataset prediction
"""
# Scale the entire dataset
X_scaled_full = scaler.transform(X)

# Predict on the same dataset
all_predictions = mlp.predict(X_scaled_full)

# Create a side-by-side comparison
comparison_df = pd.DataFrame({
    'Actual Class': y,            # The original answers (0 or 1)
    'Predicted Class': all_predictions, # What the model guessed
    'Correct?': y == all_predictions    # Did it get it right?
})

# Show the first 20 rows
print("\n--- Full Dataset Predictions (First 20 Rows) ---")
print(comparison_df.head(20))

# Check overall accuracy on the full dataset
total_correct = comparison_df['Correct?'].sum()
total_rows = len(comparison_df)
print(f"\nModel Accuracy on Full Dataset: {total_correct}/{total_rows} ({total_correct/total_rows:.2%})")

#Final Classification Report
print("\nFull Dataset Classification Report Below:")
print(classification_report(y, all_predictions))

#Additional Visualisation
"""
Vis
"""
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
vis.vis_2D_projection(ax, mlp, X_train, y_train, X_test, y_test)
vis.vis_3D_projection(mlp, X_train, y_train, X_test, y_test)
plt.show()

vis.vis_feature_importance(mlp, X.columns)
vis.vis_confidence_distribution(mlp, X_test, y_test)
vis.plot_roc_curve(mlp, X_test, y_test)