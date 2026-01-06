"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Juan 
- Omar
- Adrian

Dataset: [Name of your dataset]
Predicting: [What you're predicting]
Features: [List your features]
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'FootballTransferValueTable.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    # Your code here
    data = pd.read_csv(filename)
    
    # TODO: Print the first 5 rows
    print("=== Market Price Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())
    # TODO: Print the shape of the dataset
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    # TODO: Print basic statistics for ALL columns
    print(f"\nBasic statistics:")
    print(data.describe())

    # TODO: Print the column names
    print(f"\nColumn names: {list(data.columns)}")
    
    # TODO: Return the dataframe
    return data
    


def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Your code here
    # Hint: Use subplots like in Part 2!
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # TODO: Add a main title: 'House Features vs Price'
    fig.suptitle('Player Features vs Price', fontsize=16, fontweight='bold')

    # TODO: Plot 1 (top left): SquareFeet vs Price
    #       - scatter plot, color='blue', alpha=0.6
    #       - labels and title
    #       - grid
    axes[0, 0].scatter(data['Age'], data['Price'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('Age vs Price')
    axes[0, 0].grid(True, alpha=0.3)
    # TODO: Plot 2 (top right): Bedrooms vs Price
    #       - scatter plot, color='green', alpha=0.6
    #       - labels and title
    #       - grid
    axes[0, 1].scatter(data['Goals'], data['Price'], color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Goals')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Goals vs Price')
    axes[0, 1].grid(True, alpha=0.3)
    # TODO: Plot 3 (bottom left): Bathrooms vs Price
    #       - scatter plot, color='red', alpha=0.6
    #       - labels and title
    #       - grid
    axes[1, 0].scatter(data['Assists'], data['Price'], color='red', alpha=0.6)
    axes[1, 0].set_xlabel('Assists')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Assists vs Price')
    axes[1, 0].grid(True, alpha=0.3)
    # TODO: Use plt.tight_layout() to make plots fit nicely
    plt.tight_layout()    
    # TODO: Save the figure as 'feature_plots.png' with dpi=300
    plt.savefig('feature_plots.png', dpi=300, bbox_inches='tight')
    
    # TODO: Show the plot
    print("\n✓ Feature plots saved as 'feature_plots.png'")
    plt.show()
    


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    # Your code here
    # TODO: Create a list of feature column names
    #       ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age']
    feature_columns = ['Age', 'Goals', 'Assists',]

    # TODO: Create X by selecting those columns from data
    X = data[feature_columns]

    # TODO: Create y by selecting the 'Price' column
    y = data['Price']
   
    # TODO: Print the shape of X and y
    print(f"\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")    
    # TODO: Print the feature column names
    print(f"\nFeature columns: {list(X.columns)}")
# TODO: Split into train (80%) and test (20%) with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # TODO: Print how many samples are in training and testing sets
    print(f"\n=== Data Split ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    # TODO: Return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test


    


def train_model(X_train, y_train):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
    else:
        # If X_train is a numpy array, just use generic names
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # Your code here
    # TODO: Create a LinearRegression model
    model = LinearRegression()
    # TODO: Train the model using fit()
    model.fit(X_train, y_train)
    # TODO: Print the intercept
    print(f"\n=== Model Training Complete ===")
    print(f"Intercept: ${model.intercept_:.2f}")
    # TODO: Print each coefficient with its feature name
    #       Hint: use zip(feature_names, model.coef_)
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")
    # TODO: Print the full equation in readable format
    print(f"\nEquation:")
    equation = f"Price = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} × {name}"
        else:
            equation += f" + ({coef:.2f}) × {name}"
    equation += f" + {model.intercept_:.2f}"
    print(equation)
    # TODO: Return the trained model
    return model

    


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)

    if hasattr(X_test, 'columns'):
        feature_names = X_test.columns
    else:
        feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
    
    # Your code here
    # TODO: Make predictions on X_test
    predictions = model.predict(X_test)

    # TODO: Calculate R² score
    r2 = r2_score(y_test, predictions)
    
    # TODO: Calculate MSE and RMSE
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    # TODO: Print R² score with interpretation
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of price variation")
    # TODO: Print RMSE with interpretation
    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")
    # TODO: Calculate and print feature importance
    #       Hint: Use np.abs(model.coef_) and sort by importance
    #       Show which features matter most
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    # TODO: Return predictions
    return predictions
    


def make_prediction(model):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    pass
    


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

