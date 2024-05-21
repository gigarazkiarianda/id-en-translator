import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import warnings


warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


model_id_to_en = 'Helsinki-NLP/opus-mt-id-en'
tokenizer_id_to_en = MarianTokenizer.from_pretrained(model_id_to_en)
id_to_en = MarianMTModel.from_pretrained(model_id_to_en)


model_en_to_id = 'Helsinki-NLP/opus-mt-en-id'
tokenizer_en_to_id = MarianTokenizer.from_pretrained(model_en_to_id)
en_to_id = MarianMTModel.from_pretrained(model_en_to_id)

def translate(text, src_lang='id', tgt_lang='en'):
    
    if src_lang == 'id' and tgt_lang == 'en':
        tokenizer = tokenizer_id_to_en
        model = id_to_en
    elif src_lang == 'en' and tgt_lang == 'id':
        tokenizer = tokenizer_en_to_id
        model = en_to_id
    else:
        raise ValueError("Unsupported language pair")

    
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    translated_tokens = model.generate(**tokens)
    
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    
    return translated_text[0]


st.title('Indonesia-English Translator')


text = st.text_area('Enter text to translate')

src_lang = st.selectbox('Select source language', ['id', 'en'])

tgt_lang = st.selectbox('Select target language', ['en', 'id'])


if st.button('Translate'):
    if text:
        translated_text = translate(text, src_lang, tgt_lang)
        st.write('Translated text:')
        st.write(translated_text)
    else:
        st.write('Please enter text to translate')
