
from langchain_core.prompts import PromptTemplate
import openai,os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from pydantic_models import output_parser
import os
from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(
    openai_api_base=os.getenv('OPENAI_API_BASE'), # https://api.openai.com/v1 or https://api.groq.com/openai/v1 
    openai_api_key=os.getenv('OPENAI_API_KEY'), # os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    model_name=os.getenv('LLM_MODEL') #  gpt-4-turbo-preview or mixtral-8x7b-32768 
)
# Prepare the prompt for the LLM
prompt = """
Dataset Analysis:

Description:
{describe}

Null value counts:
{null_counts}

Data types:
{data_types}

Sample entry:
{one_entry}

Based on this information, suggest data preprocessing and feature engineering techniques needed for ML training. 
For each suggestion, provide the technique name and the column(s) it should be applied to.

The technique name should be within this: [Missing Value Imputation,One-Hot Encoding,Normalization,Outlier Removal,Binning,Date Feature Extraction,Binary Encoding]
#format_instructions
{format_instructions}
"""


prompt = PromptTemplate(
    template=prompt,
    input_variables=['describe','null_counts','data_types','one_entry'],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

def get_llm_suggestions(df):
    # Get basic information
    describe = df.describe()
    null_counts = df.isnull().sum()
    data_types = df.dtypes
    one_entry = df.iloc[0]
    res = RunnablePassthrough() | prompt | llm | output_parser

    result = res.invoke({'describe':describe,'data_types':data_types,'null_counts':null_counts,'one_entry':one_entry})
    return result