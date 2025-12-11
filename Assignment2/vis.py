import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

def vis2d(ax, model, X_train, Y_train, X_test=[], Y_test=[]):
  # identify graph range
  x_range = [X_train[:,0].min()-0.5, X_train[:,0].max()+0.5]
  y_range = [X_train[:,1].min()-0.5, X_train[:,1].max()+0.5]
  if len(X_test) > 0:
    x_range = [min(x_range[0], X_test[:,0].min()-0.5), max(x_range[1], X_test[:,0].max()+0.5)]
    y_range = [min(y_range[0], X_test[:,1].min()-0.5), max(y_range[1], X_test[:,1].max()+0.5)]
  # create a meshgrid
  xx, yy = np.meshgrid(np.arange(x_range[0], x_range[1], .01), np.arange(y_range[0], y_range[1], .01))
  # identify the area of decision
  Z = model.predict([[x,y] for x,y in zip(xx.ravel(), yy.ravel())])
  Z = Z.reshape(xx.shape)
  # plot the decision areas
  ax.contourf(xx,yy,Z,alpha=.8)
  # plot the training and testing data
  ax.scatter([x[0] for x in X_train], [x[1] for x in X_train], c=Y_train, edgecolors='black')
  if len(X_test) > 0:
    ax.scatter([x[0] for x in X_test], [x[1] for x in X_test], c=Y_test, edgecolors='brown', alpha=.8)


def vis3d(fig, model, X_train, Y_train, X_test=[], Y_test=[]):
  possible_class = np.unique(Y_train)
  y_range = [0, 1]
  y_data_min = X_train.min(axis=0)
  y_data_max = X_train.max(axis=0)
  if len(X_test) > 0:
    y_data_min = np.amin([y_data_min, X_test.min(axis=0)], axis=0)
    y_data_max = np.amax([y_data_max, X_test.max(axis=0)], axis=0)
  single_y = np.arange(y_range[0], y_range[1], .1)
  single_y = single_y.reshape(len(single_y), 1)
  yy = []
  for i in range(X_train.shape[1]):
    if len(yy) == 0:
      yy = np.tile(single_y,1)
    else:
      old = np.tile(yy, (single_y.shape[0],1))
      new = np.repeat(single_y, yy.shape[0])
      new = new.reshape(len(new),1)
      yy = np.hstack([new, old])
  yy_data = [[yi*(y_data_max[i] - y_data_min[i])+y_data_min[i] for i,yi in enumerate(y)] for y in yy]
  zz = model.predict(yy_data)
  train_x = (X_train - y_data_min)/(y_data_max - y_data_min)
  axes = []
  for i in possible_class:
    ax = fig.add_subplot(len(possible_class), 1, i+1)
    ax.plot(yy[zz == i].transpose(), c=cm.Set2.colors[i%cm.Set2.N], alpha=0.5)
    ax.plot(train_x[Y_train == i].transpose(), c='black', lw=5, alpha=.8)
    ax.plot(train_x[Y_train == i].transpose(), c=cm.Dark2.colors[i%cm.Set2.N], lw=3, alpha=.8)
    ax.set_title("output = {}".format(i))
    ax.set_xticks([i for i in range(X_train.shape[1])])
    ax.set_ylim(y_range)
    axes.append(ax)
  return axes


def vis_2D_projection(ax, model, X_train, Y_train, X_test=[], Y_test=[]):
    # Initialize and fit PCA to the training data
    pca = PCA(n_components=2)
    pca.fit(X_train)

    # Transform the data to the 2D principal component space
    X_train_2d = pca.transform(X_train)

    if len(X_test) > 0:
        X_test_2d = pca.transform(X_test)
    else:
        X_test_2d = []

    # Identify graph range using the 2D projected data
    x_range = [X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5]
    y_range = [X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5]
    if len(X_test_2d) > 0:
        x_range = [min(x_range[0], X_test_2d[:, 0].min() - 0.5), max(x_range[1], X_test_2d[:, 0].max() + 0.5)]
        y_range = [min(y_range[0], X_test_2d[:, 1].min() - 0.5), max(y_range[1], X_test_2d[:, 1].max() + 0.5)]

    # Create a meshgrid in the 2D space
    xx, yy = np.meshgrid(np.arange(x_range[0], x_range[1], .05),
                         np.arange(y_range[0], y_range[1], .05))

    # Inverse transform the meshgrid points back to the original 9D space
    mesh_2d_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_9d_points = pca.inverse_transform(mesh_2d_points)

    # Identify the area of decision using the 9D points
    Z = model.predict(mesh_9d_points)
    Z = Z.reshape(xx.shape)

    # Plot the decision areas
    ax.contourf(xx, yy, Z, alpha=.8, cmap=cm.RdYlBu)

    # Plot the training and testing data in the 2D space
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=Y_train,
               edgecolors='black', label='Training Data', cmap=cm.RdYlBu, s=50)
    if len(X_test_2d) > 0:
        ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=Y_test,
                   edgecolors='brown', alpha=.8, marker='s',
                   label='Test Data', cmap=cm.RdYlBu, s=50)

    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Principal Component 2 (PC2)")
    ax.set_title("Decision Boundary Visualization via PCA")
    ax.legend()


def vis_3D_projection(model, X_train, Y_train, X_test=[], Y_test=[]):
    # Initialize and fit PCA to 3 components
    pca = PCA(n_components=3)
    pca.fit(X_train)

    # Transform the data to the 3D principal component space
    X_train_3d = pca.transform(X_train)

    if len(X_test) > 0:
        X_test_3d = pca.transform(X_test)
    else:
        X_test_3d = []

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Training Data
    scatter_train = ax.scatter(
        X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2],
        c=Y_train, marker='o', edgecolors='black',
        label='Training Data (Class 0/1)', s=50
    )

    # Plot Testing Data
    if len(X_test_3d) > 0:
        ax.scatter(
            X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2],
            c=Y_test, marker='s', edgecolors='brown',
            label='Test Data (Class 0/1)', alpha=0.8, s=50
        )

    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Principal Component 2 (PC2)")
    ax.set_zlabel("Principal Component 3 (PC3)")
    ax.set_title("3D Data Projection via PCA")

    # Add a color bar/legend for the classes
    legend1 = ax.legend(*scatter_train.legend_elements(),
                        title="Class", loc="upper left")
    ax.add_artist(legend1)

    plt.show()


def vis_feature_importance(model, feature_names):
    weights = model.coefs_[0]

    importance = np.mean(np.abs(weights), axis=1)

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
    plt.xlabel('Average Absolute Weight Magnitude')
    plt.title('Feature Importance (Based on First Layer Weights)')
    plt.gca().invert_yaxis()
    plt.show()


def vis_confidence_distribution(model, X_test, Y_test):
    # Get the predicted probabilities for Class 1
    probabilities = model.predict_proba(X_test)[:, 1]

    # Separate probabilities by the true class
    prob_class_0 = probabilities[Y_test == 0]
    prob_class_1 = probabilities[Y_test == 1]

    plt.figure(figsize=(10, 6))

    # Plot histogram for True Class 0 (should be near 0)
    plt.hist(prob_class_0, bins=20, alpha=0.6, label='True Class 0 (Benign)', color='blue')

    # Plot histogram for True Class 1 (should be near 1)
    plt.hist(prob_class_1, bins=20, alpha=0.6, label='True Class 1 (Malignant)', color='red')

    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')

    plt.title('Model Prediction Confidence Distribution (Test Set)')
    plt.xlabel('Predicted Probability of Class 1')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.show()


def plot_roc_curve(model, X_test, Y_test):
    # Get predicted probabilities for the positive class (Class 1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)

    # Calculate the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')

    # Plot the random chance line (AUC = 0.5)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance (AUC = 0.5)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
