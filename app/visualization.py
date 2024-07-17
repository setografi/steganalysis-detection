import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_features(X, y):
    features = ['Energy', 'Mean', 'Std Dev', 'Skewness', 'Kurtosis']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Convert to DataFrame
    df = pd.DataFrame(X, columns=features)
    df['Class'] = ['Stego' if label == 1 else 'Normal' for label in y]

    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, hue='Class', kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('feature_distribution.png')
    plt.close()