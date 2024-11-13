import pandas as pd
from openai import OpenAI
import re
from pathlib import Path
import logging
from tqdm import tqdm

MODEL = "gpt-4o-mini"  
startline = 0  
batch_size = 100  # Process 100 rows at a time
checkpoint_freq = 50  # Frequency of saving checkpoints

# Prompt template
gloss_creation_prompt = """
You will create ASL glosses following these rules: 
1. Use all CAPS
2. Omit articles (a, an, the) and other words typically not signed
3. Use hyphens for single signs needing multiple words (PICK-UP, DON'T KNOW)
4. Use plus(+) for compound signs (MOTHER+FATHER means parents)
5. Time indicators go at the beginning
6. Use ASL word order (typically Topic-Comment structure)
"""

def generate_gloss(client, sentence):
    """Generate ASL gloss using OpenAI API"""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": gloss_creation_prompt},
            {"role": "user", "content": sentence}
        ],
        model=MODEL
    )
    return chat_completion.choices[0].message.content.strip()

def save_checkpoint(df, startline, c, filename):
    """Save current progress to CSV"""
    df.iloc[startline:startline+c].to_csv(filename, index=False, sep='\t')
    print(f"Checkpoint saved at {startline + c}")

def process_csv_with_checkpoints(file_path: str, output_path: str):
    #
    api_key = " :) "
    client = OpenAI(api_key=api_key)
    
    
    df = pd.read_csv(file_path, delimiter="\t")
    total_rows = len(df)
    df['GLOSS'] = ""  # Add GLOSS column if doesn't exist
    
    # Process all rows
    print(f"Starting gloss generation for {total_rows} lines")
    
    c = 0  # Counter
    for idx in tqdm(range(startline, total_rows)):
        try:
            sentence = df.loc[idx, 'SENTENCE']
            gloss = generate_gloss(client, sentence)
            df.at[idx, 'GLOSS'] = gloss
            c += 1
            
            print(f"{c} / {total_rows} ({round((c/total_rows)*100, 2)} %) - Line {startline+c-1}")
            
            
            if c % checkpoint_freq == 0:
                print("Saving checkpoint...")
                save_checkpoint(df, startline, c, output_path)
                
        except Exception as e:
            print(f"Error processing line {idx}: {str(e)}")
            df.at[idx, 'GLOSS'] = "<error>"
            continue
    
    
    save_checkpoint(df, startline, c, output_path)
    print(f"Gloss generation completed and saved (lines {startline} to {startline+c-1}).")

file_path = "/scratch/enalisn1/acarbol1/mt_class/raw_data/how2sign_train.csv"
output_path = "/scratch/enalisn1/acarbol1/mt_class/processed/processed_file_with_glosses.csv"

if __name__ == "__main__":
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    process_csv_with_checkpoints(file_path, output_path)
