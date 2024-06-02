from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib
import re

app = Flask(__name__)

SENT_MODEL_PATH = 'logReg.pkl'

SENT_TOKENIZER_PATH ='tfidfvectorizer.pkl'
# loading vectorizer
sentiment_vectorizer = joblib.load(SENT_TOKENIZER_PATH)
# loading model
sentiment_model = joblib.load(SENT_MODEL_PATH)

#getting stopwords
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Load the pre-trained BART model and tokenizer
summ_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
summ_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')



# Function to summarize text
def summarize_text(text):
    
    inputs = summ_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = summ_model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
    summary = summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        clean_text = cleanText(text)
        clean_lst = [clean_text]

        tfidf_vect = sentiment_vectorizer.transform(clean_lst)
        prediction_proba = sentiment_model.predict_proba(tfidf_vect)
        prediction_probs = round(prediction_proba[0, 1], 2)
        prob_percent = prediction_probs*100
        print("percent",prob_percent)
        if (prob_percent>=40) and (prob_percent<60):
            sentiment= "😐" + "(Neutral)"
        elif prob_percent <40:
            sentiment= "😊" + "(Positive)"
        else:
            sentiment="😠" + "(Negative)"

        summary = summarize_text(text)
        return render_template('predict.html',sentiment=sentiment, summary=summary,rawtext=text)


def cleanText(raw_review):
    
    # 2. Make a space
    raw_review = str(raw_review)
    letters_only = re.sub('[^a-zA-Z]', ' ', raw_review)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))

if __name__ == '__main__':
    app.run(debug=True)
