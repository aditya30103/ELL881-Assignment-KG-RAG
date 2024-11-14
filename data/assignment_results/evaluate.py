import pandas as pd
import json

# Define file paths for the CSV files
file_path1 = 'data/assignment_results/gemini_1.5_flash_kg_rag_based_mcq_1.csv'

# Load the CSV file into a DataFrame
df1 = pd.read_csv(file_path1)

examples = 257
df1_subset = df1.head(examples)

# Define a function to check if the correct answer is present in the LLM answer
def contains_correct_answer(row):
    try: 
        return row['correct_answer'] == json.loads(row['llm_answer'].replace('```', '').replace('\n', '').replace('json', '').replace('{{', '{').replace('}}', '}').split('}')[0] + '}')['answer']
    except:
        return False

# Apply the function to each row of the subset DataFrame
df1_subset['is_correct'] = df1_subset.apply(contains_correct_answer, axis=1)

# Calculate the percentage of correct answers for the subset
correct_rate1 = df1_subset['is_correct'].mean() * 100
print(f"Correct Answer Rate for the first {examples} examples in {file_path1}: {correct_rate1:.2f}%")
