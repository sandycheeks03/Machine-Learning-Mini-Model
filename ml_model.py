import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """
    Load the housing dataset and perform initial exploration
    """
    print("=" * 50)
    print("HOUSING PRICE PREDICTION MODEL")
    print("=" * 50)
    
    # Load dataset (you can replace this with any CSV file)
    # For this example, we'll create a sample dataset
    data = {
        'size_sqft': [1500, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3500,
                      1200, 1600, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3300],
        'bedrooms': [3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5],
        'bathrooms': [2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        'age_years': [10, 5, 15, 8, 3, 12, 6, 2, 20, 4, 25, 8, 12, 7, 2, 18, 9, 4, 11, 6],
        'price_usd': [300000, 350000, 400000, 420000, 450000, 470000, 520000, 550000, 
                      480000, 580000, 250000, 320000, 380000, 410000, 440000, 460000, 
                      500000, 530000, 490000, 560000]
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV for demonstration
    df.to_csv('housing_data.csv', index=False)
    
    # Load from CSV to simulate real scenario
    df = pd.read_csv('housing_data.csv')
    
    print("\n1. DATASET OVERVIEW")
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset information:")
    print(df.info())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for machine learning
    """
    print("\n" + "=" * 50)
    print("2. DATA PREPROCESSING")
    print("=" * 50)
    
    # Check for missing values
    print("Missing values in each column:")
    print(df.isnull().sum())
    
    # Separate features and target
    X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
    y = df['price_usd']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Train the Linear Regression model
    """
    print("\n" + "=" * 50)
    print("3. MODEL TRAINING")
    print("=" * 50)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_:.2f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    """
    print("\n" + "=" * 50)
    print("4. MODEL EVALUATION")
    print("=" * 50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    print(f"- RMSE of ${rmse:.2f}: Average prediction error")
    print(f"- R² of {r2:.4f}: {r2*100:.2f}% of price variance explained by features")
    
    return y_pred

def visualize_results(df, model, y_test, y_pred):
    """
    Create visualizations for data and results
    """
    print("\n" + "=" * 50)
    print("5. DATA VISUALIZATION")
    print("=" * 50)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature correlation heatmap
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,0])
    axes[0,0].set_title('Feature Correlation Heatmap')
    
    # 2. Actual vs Predicted prices
    axes[0,1].scatter(y_test, y_pred, alpha=0.7)
    axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0,1].set_xlabel('Actual Prices')
    axes[0,1].set_ylabel('Predicted Prices')
    axes[0,1].set_title('Actual vs Predicted Prices')
    
    # 3. Residual plot
    residuals = y_test - y_pred
    axes[1,0].scatter(y_pred, residuals, alpha=0.7)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Predicted Prices')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title('Residual Plot')
    
    # 4. Feature importance (using coefficients)
    feature_names = ['Size (sqft)', 'Bedrooms', 'Bathrooms', 'Age (years)']
    coefficients = model.coef_
    axes[1,1].barh(feature_names, coefficients)
    axes[1,1].set_xlabel('Coefficient Value')
    axes[1,1].set_title('Feature Importance (Coefficients)')
    
    plt.tight_layout()
    plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'model_results.png'")

def make_prediction(model, scaler):
    """
    Make a prediction on new data
    """
    print("\n" + "=" * 50)
    print("6. SAMPLE PREDICTION")
    print("=" * 50)
    
    # Sample new house data
    new_house = np.array([[2000, 3, 2, 5]])  # size, bedrooms, bathrooms, age
    
    # Scale the features
    new_house_scaled = scaler.transform(new_house)
    
    # Make prediction
    predicted_price = model.predict(new_house_scaled)[0]
    
    print("Sample Prediction:")
    print(f"House features: 2000 sqft, 3 bedrooms, 2 bathrooms, 5 years old")
    print(f"Predicted price: ${predicted_price:,.2f}")

def main():
    """
    Main function to run the complete ML pipeline
    """
    try:
        # Step 1: Load and explore data
        df = load_and_explore_data()
        
        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        # Step 3: Train model
        model = train_model(X_train, y_train)
        
        # Step 4: Evaluate model
        y_pred = evaluate_model(model, X_test, y_test)
        
        # Step 5: Visualize results
        visualize_results(df, model, y_test, y_pred)
        
        # Step 6: Make sample prediction
        make_prediction(model, scaler)
        
        print("\n" + "=" * 50)
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()