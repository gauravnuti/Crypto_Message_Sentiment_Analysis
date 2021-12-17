import json
import spacy
import pickle
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from tqdm import tqdm
from transformers import pipeline


def get_lang_detector(nlp, name):
    """Wrapper Function For Spacy add_pipe"""
    return LanguageDetector()

# Loading pipeline for language detection
nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

# Reading from the json file
f = open('result.json')
data = json.load(f)

valid_sentences = []  # Storing messages we need to procedd
days = []  # Stroring day for each message
coin = []  # Storing Crypto type for each message

for index in tqdm(range(len(data['messages']))):
    curr_message = data['messages'][index]
    day = int(curr_message['date'][8:10])
    sentence = ''
    if(type(curr_message['text']) == list):
        # Some messages have links in them.
        # Therfore need to extract str for each.
        for j in curr_message['text']:
            if(type(j) == str):
                sentence += j
            else:
                sentence = sentence + ' ' + j['text']
    else:
        sentence = curr_message['text']

    sentence_lang = nlp(sentence)
    if (sentence_lang._.language['language'] != 'en'):
        # If the language is not english continue
        continue

    if('shib' in sentence.lower() or 'doge' in sentence.lower()):
        # Since people text while using Doge/DOGE/doge,
        # we turn everything to lowercase and check.
        # If true we append the sentence, crypto, and day.
        valid_sentences.append(sentence)
        days.append(day)
        if('shib' in sentence.lower()):
            coin.append(0)
        else:
            coin.append(1)

# Using BERT Transformer for sentiment-analysis
sentiment_analysis = pipeline('sentiment-analysis')
res = sentiment_analysis(valid_sentences)

# Storing [num_pos, num_neg, avg_score] for each day,
# for each Crypto. Assigned 1/n*score for positive and
# -1/n*score to calculate average. Where score is given
# by the transformer and n is the number of tokens on a
# particular day.
doge_scores = [[0, 0, 0] for i in range(15)]
shiba_scores = [[0, 0, 0] for i in range(15)]

for i in range(len(res)):
    if(res[i]['label'] == 'POSITIVE'):
        if(coin[i] == 0):
            shiba_scores[days[i] - 1][0] += 1
            shiba_scores[days[i] - 1][2] += res[i]['score']
        else:
            doge_scores[days[i] - 1][0] += 1
            doge_scores[days[i] - 1][2] += res[i]['score']
    else:
        if(coin[i] == 0):
            shiba_scores[days[i] - 1][1] += 1
            shiba_scores[days[i] - 1][2] -= res[i]['score']
        else:
            doge_scores[days[i] - 1][1] += 1
            doge_scores[days[i] - 1][2] -= res[i]['score']

for i in range(15):
    # Computing avg_score by dividing by number of messages
    # on a day.
    if((doge_scores[i][0]+doge_scores[i][1]) > 0):
        doge_scores[i][2] /= (doge_scores[i][0]+doge_scores[i][1])
    if((shiba_scores[i][0]+shiba_scores[i][1]) > 0):
        shiba_scores[i][2] /= (shiba_scores[i][0]+shiba_scores[i][1])

# Storing the results to use them later for plotting
file_name = "doge_sentiment_summary.pkl"
open_file = open(file_name, "wb")
pickle.dump(doge_scores, open_file)
open_file.close()

file_name = "shiba_sentiment_summary.pkl"
open_file = open(file_name, "wb")
pickle.dump(shiba_scores, open_file)
open_file.close()
