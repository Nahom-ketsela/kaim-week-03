import pandas as pd

def convert_txt_to_csv():
    """
    Converts a pipe-delimited text file into a CSV file.
    
    Reads a text file (data.txt) with a pipe '|' delimiter and 
    writes it to a CSV file (data.csv) with a comma ',' delimiter.
    """
    # Input and output file paths
    input_file = 'C:\\Users\\HP\\OneDrive\\Desktop\\ai2\\kaim-week-03\\data\\MachineLearningRating_v3.txt'  # Replace with your actual .txt file path
    output_file = 'C:\\Users\\HP\\OneDrive\\Desktop\\ai2\\kaim-week-03\\data\\MachineLearningRating_v3.csv'  # Desired output .csv file path

    try:
        # Read the text file using the pipe delimiter
        # Add low_memory=False or specify dtypes to handle mixed types
        df = pd.read_csv(input_file, delimiter='|', low_memory=False)

        # Save the DataFrame to a CSV file using the default comma delimiter
        df.to_csv(output_file, index=False)

        print(f"File successfully converted and saved as {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
