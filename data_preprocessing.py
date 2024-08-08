import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['cleaned_text'] = data['text'].str.replace('[^\w\s]', '')
    return data

def preprocess_data(data):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(data['cleaned_text'])
    padded_sequences = pad_sequences(sequences, maxlen=100)
    return padded_sequences, data['label'], tokenizer

file_path = 'sorted_data_acl/dataset.csv'
data = load_data(file_path)
X, y, tokenizer = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save the tokenizer
with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Data preprocessing completed successfully.")
