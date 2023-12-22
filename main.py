# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Extract features and target variable
X = iris.data
y = iris.target

from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from scipy.stats import ortho_group

def einsum_outer_product_matrix(matrix1,matrix2):
    num_rows, num_cols = matrix1.shape
    return np.einsum('ij,ik->ijk', matrix1, matrix2).reshape(num_rows,-1)

def X_feature(X):
    # Standardize the features (normalize to mean=0 and variance=1)
    scaler = StandardScaler()
    XXT = einsum_outer_product_matrix(X,X)
    XXTXT = einsum_outer_product_matrix(XXT,X)
    X_normalized = np.hstack([np.ones((X.shape[0], 1)),X, XXT,XXTXT])

    X_normalized = scaler.fit_transform(X_normalized)
    return X_normalized

def type_2_one_hot(y):
    # Convert target variable 'y' to one-hot encoded vectors
    encoder = OneHotEncoder()
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))
    return y_onehot

def transform_matrix(X, y):
    # Create the matrix A (A = B * inv(X))
    A = np.linalg.pinv(X) @ y
    return A

def matrix_model(A,X):
    # Transform the original data using matrix A
    X_transformed = X @ A
    binary_matrix = np.zeros_like(X_transformed)
    max_indices = np.argmax(X_transformed, axis=1)
    binary_matrix[np.arange(binary_matrix.shape[0]), max_indices] = 1
    return X_transformed,binary_matrix






A = transform_matrix(X_feature(X), type_2_one_hot(y))
X_transformed,binary_matrix= matrix_model(A,X_feature(X))




from sklearn.metrics import accuracy_score

# Assuming y_true is the true labels and y_pred is the predicted labels
# You may replace these with your actual true and predicted labels
y_true = type_2_one_hot(y)
_,y_pred= matrix_model(A,X_feature(X))


# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Display the accuracy
print("Accuracy:", accuracy)



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming X_transformed is your transformed data and y is your labels

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the first three dimensions after PCA
scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=y, cmap='viridis', edgecolor='k', s=50)

# Set labels for each axis
ax.set_xlabel('dim 1')
ax.set_ylabel('dim 2')
ax.set_zlabel('dim 3')

# Set a title for the plot
ax.set_title("Transformed Data for Iris Data")

# Add legend
legend_labels = ['Setosa', 'Versicolour', 'Virginica']
ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Iris Types', loc='upper right')

# Add vectors of the identity matrix I3x3
origin = [0, 0, 0]
I3x3 = np.eye(3)

colors = ['red', 'green', 'blue']

for i in range(3):
    color = scatter.to_rgba(i)
    ax.quiver(origin[0], origin[1], origin[2], I3x3[i, 0], I3x3[i, 1], I3x3[i, 2], color=color, linewidth=2, arrow_length_ratio=0.1)

# Display the plot
plt.show()





import numpy as np
import matplotlib.pyplot as plt

def create_scatter(ax, x, y, color, xlabel, ylabel, title, aspect_equal=True):
    scatter = ax.scatter(x, y, c=color, cmap='viridis', edgecolor='k', s=50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if aspect_equal:
        ax.set_aspect('equal')
    return scatter

def plot_projections(ax, vectors, scatter, xlabel, ylabel, title, comb):
    for i in range(3):
        ax.arrow(0, 0, vectors[i, comb[0]], vectors[i, comb[1]], color=scatter.to_rgba(i), head_width=0.1, head_length=0.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def create_and_plot_scatter(ax, x, y, color, xlabel, ylabel, title, vectors, comb, aspect_equal=True):
    scatter = create_scatter(ax, x, y, color, xlabel, ylabel, title, aspect_equal)
    plot_projections(ax, vectors, scatter, xlabel, ylabel, title, comb)

# Assuming X_transformed is your transformed data and y is your labels
I3x3 = np.eye(3)

# Mapping for combinations
comb_mapping = {0: (0, 1), 1: (0, 2), 2: (1, 2)}

# Create a figure with subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterate over combinations and axes
for i, ax in enumerate(axs):
    create_and_plot_scatter(ax, X_transformed[:, comb_mapping[i][0]], X_transformed[:, comb_mapping[i][1]], y, f'dim {comb_mapping[i][0]}', f'dim {comb_mapping[i][1]}', f'Projection onto dim {comb_mapping[i][0]}-dim {comb_mapping[i][1]} plane', I3x3, comb_mapping[i])

plt.tight_layout()
plt.show()

