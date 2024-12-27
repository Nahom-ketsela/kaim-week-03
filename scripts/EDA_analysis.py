import pandas as pd
import seaborn as sns
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

# Data Quality Assessment
def data_quality_assessment(df):
    """
    Check for missing values in the dataset.
    """
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])  # Print columns with missing values

# Univariate Analysis
def univariate_analysis(df):
    """
    Perform univariate analysis with histograms for numerical columns 
    and bar charts for categorical columns.
    """
    # Plot histograms for numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns].hist(bins=15, figsize=(15, 10), layout=(3, 4))
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

# Bivariate or Multivariate Analysis
def bivariate_analysis(df):
    """
    Perform bivariate or multivariate analysis to explore relationships.
    """
    # Correlation heatmap for numerical variables
    plt.figure(figsize=(12, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # Scatter plot for TotalPremium vs. TotalClaims (completed)
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

# Outlier Detection
def outlier_detection(df):
    """
    Use box plots to detect outliers in numerical data.
    """
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[numerical_columns], orient='h', palette="Set2")
    plt.title("Boxplots of Numerical Variables")
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

    # Visualization 3: Heatmap of correlations with annotated values
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap with Annotations")
    plt.show()
