import streamlit as st
from tools import *
from pydantic_models import output_parser
from model_call import get_llm_suggestions
from typing import List, Union

def apply_techniques(df: pd.DataFrame, suggestions) -> pd.DataFrame:
    for suggestion in suggestions:
        technique = suggestion.technique
        columns = suggestion.columns
        parameters = suggestion.parameters
        
        for column in columns:
            if technique == "Missing Value Imputation":
                method = parameters.get('method', 'mean')
                df = missing_value_imputation(df, column, method=method)
            elif technique == "One-Hot Encoding":
                print(column)
                df = one_hot_encoding(df, column)
            elif technique == "Normalization":
                method = parameters.get('method', 'min-max')
                df = normalization(df, column, method=method)
            elif technique == "Outlier Removal":
                method = parameters.get('method', 'iqr')
                threshold = parameters.get('threshold', 1.5)
                df = outlier_removal(df, column, method=method, threshold=threshold)
            elif technique == "Binning":
                bins = parameters.get('bins', 5)
                labels = parameters.get('labels')
                df = bin_numeric_column(df, column, bins=bins, labels=labels)
            elif technique == "Date Feature Extraction":
                df = create_date_features(df, column)
            elif technique == 'Binary Encoding':
                df = binary_encoding(df, column)
    
    return df

def main():
    st.title("Data Preprocessing App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Original Data:")
        st.write(df.head())

         # Use session state to store LLM output
        if 'llm_output' not in st.session_state:
            with st.spinner("Generating preprocessing suggestions..."):
                st.session_state.llm_output = get_llm_suggestions(df)

        st.write("Suggested Techniques:")
        selected_suggestions = []
        for suggestion in st.session_state.llm_output.suggestions:
            if st.checkbox(f"{suggestion.technique} - {', '.join(suggestion.columns)}"):
                selected_suggestions.append(suggestion)

        if st.button("Apply Selected Techniques"):
            if selected_suggestions:
                processed_df = apply_techniques(df, selected_suggestions)
                st.write("Processed Data:")
                st.write(processed_df.head())

                csv = processed_df.to_csv(index=False)
                st.download_button(
                    label="Download processed data as CSV",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No techniques selected. Please select at least one technique.")

if __name__ == "__main__":
    main()