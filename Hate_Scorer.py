import re

import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline, set_seed
from openai import OpenAI
import pandas as pd
# Set the seed for reproducibility
set_seed(42)

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key="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"
)

def generate_text(
        prompt,
        max_tokens=500,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stream=True
):
    messages = [
        {
            "role": "user",
            "content": prompt + "\nOnly output the generated score, no other text."
        }
    ]

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        max_tokens=20,
        temperature=0.1,  # 控制随机性 (0-2)，越大越随机
        top_p=0.9,  # 控制输出的多样性
        frequency_penalty=0.3,  # 降低重复词的可能性
        stream=True
    )

    if stream:
        full_text = ""
        for chunk in stream:
            text_chunk = chunk.choices[0].delta.content
            if text_chunk:
                full_text += text_chunk
        return full_text
    else:
        return stream.choices[0].message.content

generator = lambda prompt, **kwargs: generate_text(prompt, **kwargs)






def get_hate_score(HATE_SPEECH_TEXT, generator):
    prompt = f"""Read the following hate speech definition and scoring criteria, then carefully evaluate the given text and provide a score between 0 and 100 that represents the level of prejudice and hate in the text.

<hate_speech_definition>
Hate speech refers to language that expresses prejudice against a person or group based on their race, ethnicity, national origin, religion, gender, sexual orientation, or other protected characteristics. It often involves the use of derogatory or dehumanizing language, stereotypes, and false claims about the abilities or worthiness of a particular group.
</hate_speech_definition>


<scoring_criteria>
0 - No hate speech detected. The text does not express any prejudice or hatred towards a group.
25 - Mild hate speech. The text contains some language that is mildly derogatory or stereotypical towards a group, but does not directly advocate for harm or discrimination.
50 - Moderate hate speech. The text contains language that is clearly prejudiced and dehumanizing towards a group, and may express views that could lead to discrimination or harm.
75 - Severe hate speech. The text contains language that is highly prejudiced, dehumanizing, and advocates for the mistreatment or harm of a protected group.
100 - Extreme hate speech. The text contains language that is overtly hateful, dehumanizing, and promotes violence or discrimination against a protected group.
</scoring_criteria>

<examples>
Example 1:
真恶心啊那个电影男主角。
Score:
0
Example 2:
只要有一个男人是坏的，所有男人都不是好东西？
Score:
0
Example 2:
男的真恶心
Score:
25
Example 2:
貌似江浙沪一起歧视安徽人……
Score:
30
Example 3:
你素质差能不让东北人背锅吗？就因为你这样的人东北人天天被骂。
Score:
50
Example 4:
现在广州大部分还是第一代原生态黑人，他们没有受过中华文明教育，长的也不美。
Score:
50
Example 4:
这种女人就是傻逼
Score:
60
Example 4:
我就不明白美国不能歧视黑人，为啥就能随便歧视亚裔
Score:
75
Example 5:
非洲黑人智商70，被学者归类为智障水平。美国黑人混血儿智商85，中非混血儿智商应该徘徊在90左右，妥妥的班里倒数。
Score:
100
Example 6:
只有资本家才能入犹太教当犹太人
Score:
100
</examples>

When evaluating the given text, consider the following:
- The specific language used and its level of prejudice or dehumanization
- The context in which the language is used and any underlying biases or beliefs it may reflect
- The potential impact the language could have on the targeted group or on society as a whole

After carefully evaluating the text, provide a score between 0 and 100 that represents the level of prejudice and hate expressed. Be sure to provide a brief explanation for your score.

<text>
{HATE_SPEECH_TEXT}
</text>

Score:"""

    try:
        result = generator(prompt)
        score = result.split("New sentence:")[-1].strip()
        return score
    except:
        return "Error"


def extract_last_column_as_list(file_path, output_file=None,start_row=None):
    """
    Reads a CSV file with variable-length rows, extracts the last column from each row
    using regex, and stores the results in a Python list. Optionally saves the result to a file.

    Parameters:
    - file_path (str): Path to the input CSV file.
    - output_file (str, optional): Path to save the output as a text file. Defaults to None.

    Returns:
    - list: A list containing the extracted last column values.
    """
    last_column_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Use regex to extract the last column value
            match = re.findall(r'([^,]+)$', line.strip())
            if match:
                last_column_list.append(match[0])

    # Save the list to a file if output_file is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in last_column_list:
                f.write(item + '\n')

    if start_row:
        last_column_list = last_column_list[start_row:]

    return last_column_list


def list_to_df(data_list):
    """
    Converts a list into a Pandas DataFrame with string data type.

    Parameters:
    - data_list (list): A list of strings to be converted into a DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame containing the list elements as strings.
    """
    # Create a DataFrame with a single column, ensuring data type is string
    df = pd.DataFrame(data_list, columns=['text'], dtype=str)
    return df


def load_data(input_file, start_row=1, end_row=None):
    rows=extract_last_column_as_list(input_file,start_row=start_row)
    df = list_to_df(rows)
    return df


def process_hate_speech(input_file, output_file, generator, start_row=None, end_row=None):
    # 加载数据
    df = load_data(input_file, start_row, end_row)
    tqdm.pandas(desc="处理进度")
    df['hate_score'] = df['text'].progress_apply(
        lambda x: get_hate_score(x, generator)
    )
    # 保存结果
    df[['text', 'hate_score']].to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")

    # 处理前10行数据


process_hate_speech(
    input_file="test.csv",
    output_file="hate_score_results.csv",
    generator=lambda prompt, **kwargs: generate_text(prompt, **kwargs),
    start_row=0,
    end_row=10
)
