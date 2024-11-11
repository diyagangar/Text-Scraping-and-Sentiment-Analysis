#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load the Excel file
file_path = 'Input.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Function to extract the title and article text
def extract_article_text(url_id, url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the title
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'No Title Found'

        # Extract the article text
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p')
        else:
            paragraphs = soup.find_all('p')

        article_text = '\n'.join([para.get_text(strip=True) for para in paragraphs])

        # Return url_id and concatenated title and article_text
        return url_id, f"{title}, {article_text}"

    except Exception as e:
        return url_id, f"Failed to extract article {url_id}: {e}"

# Function to handle parallel execution
def fetch_all_articles(df):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(extract_article_text, row['URL_ID'], row['URL']): row for index, row in df.iterrows()}
        for future in as_completed(future_to_url):
            url_id, result = future.result()
            results.append(result)
            print(f"Processed article {url_id}")
    return results

# Fetch all articles in parallel
article_contents = fetch_all_articles(df)

# Save all articles in a single text file
output_file = 'all_articles.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for content in article_contents:
        file.write(content + '\n')

print(f"All articles saved to {output_file}")


# In[ ]:





# In[13]:


import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load the Excel file
file_path = 'Input.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Function to extract the title and article text
def extract_article_text(url_id, url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the title
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'No Title Found'

        # Extract the article text
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p')
        else:
            paragraphs = soup.find_all('p')

        article_text = '\n'.join([para.get_text(strip=True) for para in paragraphs])

        return {'URL_ID': url_id, 'URL': url, 'Title': title, 'Text': article_text, 'Status': 'Success', 'Message': 'Article saved successfully'}
    except Exception as e:
        return {'URL_ID': url_id, 'URL': url, 'Title': '', 'Text': '', 'Status': 'Failed', 'Message': str(e)}

# Function to handle parallel execution
def fetch_all_articles(df):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(extract_article_text, row['URL_ID'], row['URL']): row for index, row in df.iterrows()}
        for future in as_completed(future_to_url):
            result = future.result()
            results.append(result)
            print(result)
    return results

# Fetch all articles in parallel
results = fetch_all_articles(df)

# Save the results to an Excel file
results_df = pd.DataFrame(results)
results_df.to_excel('article_extraction_results.xlsx', index=False)

print("Results saved to 'article_extraction_results.xlsx'")


# In[3]:


import pandas as pd
df=pd.read_excel("article_extraction_results1.xlsx")
df.head()


# In[4]:


sorted_df = df.sort_values(by='URL_ID', ascending=True)
sorted_df.head()


# In[17]:


import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

# Download the NLTK tokenizer
nltk.download('punkt')

# Function to load custom stopwords from multiple files
def load_stopwords(file_paths):
    stopwords = set()
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            stopwords.update(file.read().splitlines())
    return stopwords

# Load custom stopwords from multiple files
stopword_files = [
    'StopWords_Auditor.txt', 'StopWords_Currencies.txt', 'StopWords_DatesandNumbers.txt',
    'StopWords_Generic.txt', 'StopWords_GenericLong.txt', 'StopWords_Geographic.txt', 'StopWords_Names.txt'
]
custom_stop_words = load_stopwords(stopword_files)

# Function to remove stopwords
def remove_stopwords_and_numbers(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in custom_stop_words and not word.isdigit()]
    return ' '.join(filtered_text)
sorted_df['Title'] = sorted_df['Title'].astype(str).fillna('')
sorted_df['Text'] = sorted_df['Text'].astype(str).fillna('')
sorted_df['Title'] = sorted_df['Title'].apply(remove_stopwords)
sorted_df['Text'] = sorted_df['Text'].apply(remove_stopwords)


# In[18]:


sorted_df.head()


# In[19]:


# Function to load positive or negative words from a text file
def load_words_from_file(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return words

# Load positive words from text file
positive_words_file = 'positive-words.txt'
positive_words = load_words_from_file(positive_words_file)

# Load negative words from text file
negative_words_file = 'negative-words.txt'
negative_words = load_words_from_file(negative_words_file)

# Filter out stopwords from positive and negative words lists
filtered_positive_words = [word for word in positive_words if word.lower() not in custom_stop_words]
filtered_negative_words = [word for word in negative_words if word.lower() not in custom_stop_words]

# Tokenize filtered positive and negative words
tokenized_positive_words = [word_tokenize(word) for word in filtered_positive_words]
tokenized_negative_words = [word_tokenize(word) for word in filtered_negative_words]

# Print or use tokenized_positive_words and tokenized_negative_words as needed
print("Tokenized Positive Words:", tokenized_positive_words)
print("Tokenized Negative Words:", tokenized_negative_words)


# In[31]:


def calculate_positive_score(text):
    tokens = word_tokenize(text)
    return sum(1 for token in tokens if any(token in pos_words for pos_words in tokenized_positive_words))

# Function to calculate negative score for a text
def calculate_negative_score(text):
    tokens = word_tokenize(text)
    return sum(1 for token in tokens if any(token in neg_words for neg_words in tokenized_negative_words)) * -1

def calculate_polarity_score(pos_score, neg_score):
    return (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)

# Function to calculate subjectivity score
def calculate_subjectivity_score(pos_score, neg_score, total_words):
    return (pos_score + neg_score) / (total_words + 0.000001)

# Apply functions to create columns for scores
sorted_df['Positive_Score'] = sorted_df['Text'].apply(calculate_positive_score)
sorted_df['Negative_Score'] = sorted_df['Text'].apply(calculate_negative_score)
sorted_df['Polarity Score'] = sorted_df.apply(lambda row: calculate_polarity_score(row['Positive_Score'], row['Negative_Score']), axis=1)
sorted_df['Subjectivity Score'] = sorted_df.apply(lambda row: calculate_subjectivity_score(row['Positive_Score'], row['Negative_Score'], len(word_tokenize(row['Text']))), axis=1)


# In[32]:


sorted_df.head()


# In[38]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
def calculate_avg_sentence_length(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences)

def calculate_avg_sentence_length(text):
    sentences = sent_tokenize(str(text))  # Ensure text is converted to string
    words = word_tokenize(str(text))
    return len(words) / len(sentences) if len(sentences) > 0 else 0  # Handle case of empty text

# Function to determine if a word is complex (not in stop words)
stop_words = set(stopwords.words('english'))
def is_complex_word(word):
    return word.lower() not in stop_words and len(word) > 1

# Function to calculate percentage of complex words
def calculate_percentage_complex_words(text):
    words = word_tokenize(str(text))  # Ensure text is converted to string
    complex_words = [word for word in words if is_complex_word(word)]
    return len(complex_words) / len(words) * 100 if len(words) > 0 else 0  # Handle case of empty text

# Function to calculate Gunning Fog Index
def calculate_gunning_fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

# Apply functions to create columns for readability analysis
sorted_df['Avg_Sentence_Length'] = sorted_df['Text'].apply(calculate_avg_sentence_length)
sorted_df['Percentage_Complex_Words'] = sorted_df['Text'].apply(calculate_percentage_complex_words)
sorted_df['Gunning_Fog_Index'] = sorted_df.apply(lambda row: calculate_gunning_fog_index(row['Avg_Sentence_Length'], row['Percentage_Complex_Words']), axis=1)


# In[39]:


sorted_df.head()


# In[41]:


import re
def calculate_avg_words_per_sentence(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences) if len(sentences) > 0 else 0

# Function to count syllables in a word
def syllable_count(word):
    vowels = 'aeiouy'
    count = 0
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count

# Function to calculate number of complex words (>2 syllables)
def calculate_complex_word_count(text):
    words = word_tokenize(text.lower())
    complex_word_count = sum(1 for word in words if syllable_count(word) > 2)
    return complex_word_count

# Function to calculate word count (excluding stopwords and punctuations)
def calculate_word_count(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in words if word.isalnum() and word not in stop_words]
    return len(cleaned_words)

# Function to calculate personal pronoun count
def calculate_personal_pronoun_count(text):
    pronouns = ['i', 'we', 'my', 'ours', 'us']
    regex_pattern = r'\b(?:{})\b'.format('|'.join(pronouns))
    matches = re.findall(regex_pattern, text.lower())
    return len(matches)

# Function to calculate average word length
def calculate_average_word_length(text):
    words = word_tokenize(text.lower())
    total_characters = sum(len(word) for word in words if word.isalnum())
    return total_characters / len(words) if len(words) > 0 else 0
def calculate_syllable_count(text):
    words = word_tokenize(text.lower())
    syllables = [syllable_count(word) for word in words]
    return sum(syllables)

# Apply function to create a new column 'Syllable_Count'
sorted_df['Syllable_Count'] = sorted_df['Text'].apply(calculate_syllable_count)


# Apply functions to create columns
sorted_df['Avg_Words_Per_Sentence'] = sorted_df['Text'].apply(calculate_avg_words_per_sentence)
sorted_df['Complex_Word_Count'] = sorted_df['Text'].apply(calculate_complex_word_count)
sorted_df['Word_Count'] = sorted_df['Text'].apply(calculate_word_count)
sorted_df['Personal_Pronoun_Count'] = sorted_df['Text'].apply(calculate_personal_pronoun_count)
sorted_df['Average_Word_Length'] = sorted_df['Text'].apply(calculate_average_word_length)


# In[43]:


sorted_df.head()


# In[45]:


excel_file_path = 'output_data.xlsx'

# Export DataFrame to Excel
sorted_df.to_excel(excel_file_path, index=False)

