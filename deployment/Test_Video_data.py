import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to continuously read the CSV file and update the plot
@st.cache(allow_output_mutation=True)
def read_csv(file_name):
    return pd.read_csv(file_name)

# Streamlit app
def main():
    st.title('Real-time CSV Data Visualization')
    file_name = st.text_input('Enter CSV file path:')
    
    if file_name:
        st.write(f"Reading CSV file: {file_name}")
        
        # Initial read of the file
        data = read_csv(file_name)
        
        # Display line chart
        st.line_chart(data)

        # Continuously update the plot in real-time
        while True:
            # Read CSV file
            new_data = read_csv(file_name)
            
            # Check if new rows have been added
            if len(new_data) > len(data):
                data = new_data
                
                # Update line chart
                st.line_chart(data)
    
if __name__ == '__main__':
    main()
