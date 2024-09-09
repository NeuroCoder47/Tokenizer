# Tokenizer and Dataset Preprocessing

This project focuses on implementing a custom tokenizer and processing a large dataset for use in Natural Language Processing (NLP) models. The dataset undergoes several stages of cleaning, splitting, and preparation for training a model.

![Tokenizer and Data Pipeline](./path_to_your_image.jpg)

## Dataset

The dataset, named `cosmopedia_100k_train.csv`, contains various text fields. It is split into training and testing subsets and undergoes cleaning to remove unnecessary characters and columns.

### Dataset Structure
- The dataset consists of text data with the following columns after cleaning: `['text']`.
- Columns such as `seed_data`, `format`, `audience`, `text_token_length`, and `prompt` are dropped.

## Key Steps

1. **Loading the Dataset**: The dataset is loaded using Pandas.
    ```python
    df = pd.read_csv(r"C:\\Users\\cosmopedia_100k_train.csv")
    ```
   
2. **Cleaning the Text**: The text data is cleaned by removing special characters and unnecessary punctuations.
    ```python
    def clean_text(df):
        df['text'] = df['text'].apply(lambda text: re.sub(r'\\|\\n|;', ' ', text.replace('"', ' ').replace('\\n', ' ')).lower())
        return df
    ```

3. **Splitting the Data**: The data is split into training and testing sets using `train_test_split` with 80/20 split.
    ```python
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    ```

4. **Saving the Preprocessed Data**: The cleaned and split data is saved into CSV files.
    ```python
    train_df.to_csv(train_csv_path, index=False, encoding="utf-8")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8")
    ```

5. **Combining the Dataset**: A combined dataset is created from the training and testing data for further analysis or usage.
    ```python
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df.to_csv(combined_csv_path, index=False, encoding="utf-8")
    ```

## Tokenizer

A tokenizer is used to split the text data into tokens for further processing in NLP models. It is essential for transforming text into a format that can be understood by machine learning algorithms.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tokenizer-project.git
