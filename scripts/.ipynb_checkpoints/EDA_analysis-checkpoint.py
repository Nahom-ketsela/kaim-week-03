import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset in chunks to avoid memory issues
def load_data_in_chunks(file_path, chunk_size=50000):
    """
    Load CSV file in chunks to handle large files and avoid memory issues.
    """
    chunks = []
    try:
        # Read the file in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)
        # Concatenate all chunks into one DataFrame
        df = pd.concat(chunks, axis=0, ignore_index=True)
        print(f"Data successfully loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load the data with dtype specification to save memory (if you know the column types)
def load_data(file_path, dtype_dict=None):
    """
    Load CSV file into a pandas DataFrame with specified data types.
    """
    try:
        df = pd.read_csv(file_path, dtype=dtype_dict)
        print(f"Data successfully loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data Summarization
def data_summarization(df):
    """
    Perform data summarization such as descriptive statistics and data types.
    """
    print("\nDescriptive Statistics:")
    print(df.describe())  # Show summary statistics

    print("\nData Types:")
    print(df.info())  # Display data types of all columns

def review_data_structure(df):
    """
    Review the data structure (dtype) of each column and confirm if categorical variables,
    dates, and numerical features are correctly formatted.
    """
    print("Data Structure (Dtypes of each column):")
    print(df.dtypes)
    print("\n")
    
    # Check for columns that need type conversion
    print("Columns that may need conversion to correct types:")
    for col in df.columns:
        if df[col].dtype == 'object':  # Potential categorical or date columns
            if df[col].str.contains('-').any():  # Simple check for date-like values
                print(f"{col} might need to be converted to datetime.")
            else:
                print(f"{col} seems to be a categorical column.")

def descriptive_statistics(df):
    """
    Calculate the descriptive statistics for numerical columns such as TotalPremium, TotalClaims, etc.
    """
    # Numerical columns to consider for descriptive statistics
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # Get the descriptive statistics
    desc_stats = df[numerical_cols].describe().T  # Transpose for better readability
    
    # Calculate additional statistics such as variance and standard deviation
    desc_stats['variance'] = df[numerical_cols].var()
    desc_stats['std_dev'] = df[numerical_cols].std()
    
    # Display the result
    print("Descriptive Statistics (with variability metrics):")
    print(desc_stats)
    
    return desc_stats
    
# Data Quality Assessment
def data_quality_assessment(df):
    """
    Check for missing values in the dataset.
    """
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])  # Print columns with missing values


def handle_missing_values(df, threshold=0.2, default_date='1900-01-01', default_value='Unknown'):
    """
    Handle missing values in the DataFrame by:
    1. Dropping columns with too many missing values (threshold).
    2. Filling missing categorical values with the most frequent value.
    3. Filling missing numerical values with the median.
    4. Handling missing dates by filling with a default date.
    5. Handling missing values for specific columns like 'Bank'.
    
    Args:
    - df: DataFrame containing the data.
    - threshold: Proportion of missing values beyond which the column is dropped.
    - default_date: The default date to use for filling missing date values.
    - default_value: The value to use for filling missing categorical values.
    
    Returns:
    - df: DataFrame with missing values handled.
    """
    # Step 1: Drop columns with missing values above the threshold
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > threshold].index
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Step 2: Handle missing categorical columns (fill with mode)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])  # Fill with mode (most frequent value)
    
    # Step 3: Handle missing numerical columns (fill with median)
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())  # Fill with median
    
    # Step 4: Handle missing dates (fill with default date)
    date_columns = df.select_dtypes(include=['datetime']).columns
    for col in date_columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(pd.to_datetime(default_date))  # Fill with default date
    
    # Step 5: Handle missing values in specific columns ( 'Bank')
    if 'Bank' in df.columns:
        df['Bank'] = df['Bank'].fillna(default_value)  # Fill 'Bank' column missing values with 'Unknown'
    
    # Optional: Create missing flags for all columns
    for col in df.columns:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
    
    return df


def univariate_analysis(df):
    """
    Perform univariate analysis with histograms for numerical columns 
    and bar charts for categorical columns.
    """
    # Plot histograms for numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Number of numerical columns
    n = len(numerical_columns)
    
    # Dynamically calculate number of rows and columns for subplots
    ncols = 4  # Keep the number of columns fixed at 4
    nrows = np.ceil(n / ncols).astype(int)  # Calculate the number of rows needed

    # Plot histograms
    df[numerical_columns].hist(bins=15, figsize=(15, nrows * 4), layout=(nrows, ncols))
    plt.suptitle("Histograms of Numerical Variables")
    plt.tight_layout()
    plt.show()

    # Plot bar charts for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            df[col].value_counts().head(10).plot(kind='bar')
            plt.title(f"Top 10 Frequent Categories in {col.capitalize()}")
            plt.xlabel(col.capitalize())
            plt.ylabel("Frequency")
            plt.show()


def bivariate_analysis(df):
    """
    Perform bivariate or multivariate analysis to explore relationships.
    """
    # Select only numerical columns for correlation
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Ensure there are no missing values in the numerical columns
    df_numerical = df[numerical_columns].dropna()

    # Scatter plot for TotalPremium vs. TotalClaims (completed)
    if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
        plt.scatter(df['TotalPremium'], df['TotalClaims'])
        plt.title('Scatter Plot of TotalPremium vs. TotalClaims')
        plt.xlabel('TotalPremium')
        plt.ylabel('TotalClaims')
        plt.show()

# Data Comparison
def data_comparison(df):
    """
    Compare data trends over geographic locations like ZipCode or Province.
    """
    # Compare the average TotalPremium by ZipCode
    if 'ZipCode' in df.columns:
        plt.figure(figsize=(12, 6))
        df.groupby('ZipCode').agg({'TotalPremium': 'mean'}).plot(kind='bar')
        plt.title("Average TotalPremium by ZipCode")
        plt.xlabel('ZipCode')
        plt.ylabel('Average TotalPremium')
        plt.show()

    # Compare the distribution of TotalPremium by Province 
    if 'Province' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Province', y='TotalPremium', data=df)
        plt.title("Distribution of TotalPremium by Province")
        plt.show()


#outlier detetction function
def outlier_detection(df):
    """
    Use box plots to detect outliers in numerical data.
    Handles large datasets efficiently without limiting the number of rows.
    
    Parameters:
    - df (DataFrame): The input dataframe.
    """
    # Reduce memory usage
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    
    # Select numerical columns
    numerical_columns = df.select_dtypes(include=['float32', 'int32']).columns
    
    # Plot each column individually
    for column in numerical_columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=df[column])  # Removed palette to avoid warning
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        plt.tight_layout()
        plt.show()


# Visualization
def creative_visualizations(df):
    """
    Produce creative and beautiful plots that capture key insights from the EDA.
    """
    # Visualization 1: Histogram of TotalPremium with a twist
    plt.figure(figsize=(10, 6))
    sns.histplot(df['TotalPremium'], bins=30, kde=True, color='skyblue', linewidth=0)
    plt.title("Distribution of TotalPremium with KDE")
    plt.xlabel('TotalPremium')
    plt.ylabel('Frequency')
    plt.show()

    # Visualization 2: Scatter plot of TotalPremium vs. TotalClaims with color based on Vehicle Type
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='VehicleType', data=df, palette='coolwarm', s=100, edgecolor='black')
    plt.title("TotalPremium vs. TotalClaims by Vehicle Type")
    plt.xlabel('TotalPremium')
    plt.ylabel('TotalClaims')
    plt.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

