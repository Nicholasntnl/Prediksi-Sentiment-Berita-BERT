import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import re
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st
import os

# Load model
# MODEL_PATH = "quantized_model.pth"
# Judul aplikasi
st.markdown("<h1 style='text-align: center;'>PREDIKSI SENTIMEN BERITA KEUANGAN MEDIA DI INDONESIA</h1>", unsafe_allow_html=True)

# Inisialisasi NLTK
if 'punkt_tab' not in st.session_state:
    nltk.download('punkt_tab')
    st.session_state['punkt_tab'] = True

if 'stopwords' not in st.session_state:
    nltk.download('stopwords')
    st.session_state['stopwords'] = True

if 'wordnet' not in st.session_state:
    nltk.download('wordnet')
    st.session_state['wordnet'] = True

try:
    stop_words = set(stopwords.words('indonesian'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

lemmatizer = WordNetLemmatizer()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

class BertClassifierTuneGeLU(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(BertClassifierTuneGeLU, self).__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(768, 256)
        self.gelu = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(256, 4)  # 4 kelas: Negative, Neutral, Positive, Dual

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        _, pooled_output = outputs
        dropout_output = self.dropout(pooled_output)
        fc1_output = self.fc1(dropout_output)
        gelu_output = self.gelu(fc1_output)
        output = self.fc2(gelu_output)
        return output

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

# Load model
MODEL_PATH = "quantized_model.pth"
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
model.eval()

# Simpan model dan tokenizer ke dalam session state
if 'model' not in st.session_state:
    st.session_state['model'] = model

if 'tokenizer' not in st.session_state:
    st.session_state['tokenizer'] = tokenizer

# Preprocessing function
@st.cache_data
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\.com|\.id|\.co', '', text)
    text = re.sub(r'https?\S+|www\.\S+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[{}]+'.format(string.punctuation), '', text)
    additional_symbols = r'[©â€“œ]'
    text = re.sub(additional_symbols, '', text)
    text = re.sub(r'\s+', ' ', text)
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Fungsi prediksi
@st.cache_data
def predict(text, _model, _tokenizer):
    cleaned_text = clean_text(text)  # Preprocess teks inputan
    inputs = tokenizer.encode_plus( #tokenization atau vectorization
        cleaned_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1)
    return prediction.item()

# Mengurangi jarak secara maksimal antara label dan text box
input_text = st.text_area("Masukkan Teks Berita:", height=300)

# CSS untuk merapikan tombol
st.markdown("""
    <style>
        .stButton>button {
            background-color: #74574F;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            margin-left: 545px;
            margin-top: -10px;
            padding: 10px 18px;
            font-size: 30px;
            cursor: pointer;
        }
        .result {
            font-family: 'Times New Roman';
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Jika tombol ditekan, lakukan prediksi
if st.button("Prediksi Sentimen"):
    if input_text:
        model = st.session_state['model']
        tokenizer = st.session_state['tokenizer']
        hasil_sentimen = predict(input_text, model, tokenizer)
        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', 3: 'Dual'}

        # Menentukan warna berdasarkan hasil prediksi
        if hasil_sentimen == 0:  # Negative
            color = "#9b373d"  # Warna merah
        elif hasil_sentimen == 1:  # Neutral
            color = "#004278"  # Warna biru
        elif hasil_sentimen == 2:  # Positive
            color = "#006c4f"  # Warna hijau
        elif hasil_sentimen == 3:  # Dual
            color = "#4d4e56"  # Warna abu

        # Menampilkan hasil dengan warna yang berbeda
        st.markdown(f"""
        <div style="border-radius: 0px; padding : 10px; background-color: {color}; color: white;" class="result">
            Hasil Prediksi : {label_map.get(hasil_sentimen, 'Unknown')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write("Silakan masukkan teks berita untuk diprediksi.")
