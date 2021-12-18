# Crypto Message Sentiment Analysis

## Implementation

High level Code flow:
1. Read the data from `result.json`.
2. Then iterate over each message and use spacy to detect the language.
3. If language is english then check wether 'shib' or 'doge' is in the message(to make the DOGE/SHIB case invariant each sentence is transformes to lower case).
4. If yes then collect the sentences in `valid_sentences`.
5. Use Huggingface's Transformers Library to perform sentiment classification and the results are stored in intermediate pickle files.
6. Then in `plot_sentiment.py` use plotly to plot. 

## Sentiment Analysis Method

I used Huggingface's Transformers library to perform sentiment analysis. It uses DistilBERT which is a distilled version of BERT transformer and has lesser amount of variables. I used the sentiment-analysis pipeline avaialable in the library. The pipeline has two components, a tokenizer and a model. The tokenizer(DistilBert uses wordpeice algorithm to tokenize) splits the sentence into words(or group of characters) and performs look up on it, which then transforms the sentence into tokens, on which the model acts. Consider the following example:

```python
>>> from transformers import pipeline
>>> from transformers import AutoTokenizer

#Tokenization for DistilBert
>>> model_name = "distilbert-base-uncased-finetuned-sst-2-english"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
>>> tokenizer("Shib gives good profits.")
{'input_ids': [101, 11895, 2497, 3957, 2204, 11372, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

# Allocate a pipeline for sentiment-analysis
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('Shib gives good profits.')
[{'label': 'POSITIVE', 'score': 0.9997856020927429}]
```

I experimented with other transformer models which were trained on other datsets like english tweets, however the results were of a similar nature. Additionally to get better results perhaps finetuning on a similar dataset is required, as these models were not able to pick on common terminology such as the one present in "Doge coin to the moon"(for which DistilBert gave low confidence Positive). Also there were many complaints by users how they were facing technical difficulty in trading via app which were classified as negative with high confidence, which ideally is a neutral statement.
## Result Summary

I generated three plots using plotly. Code for the same  
![](https://github.com/gauravnuti/Crypto_Message_Sentiment_Analysis/blob/master/Total_Messages.png?raw=True)
The above plot shows the number of messages on each day and also what fraction of them were negative. As we can see negative messages are far more than the positive messages for both of them combined, but as has been noted in the previous section such negativity cannot all be attributed to the reaction towards the specific crypto. Additionally there seems to be some heightened activity during 9th, 10th and 11th.
![](https://github.com/gauravnuti/Crypto_Message_Sentiment_Analysis/blob/master/Total_Avg_Score.png?raw=True)
The above plot shows that the average score is negative, which then again corresponds with the first plot. 
![](https://github.com/gauravnuti/Crypto_Message_Sentiment_Analysis/blob/master/Doge_vs_shiba_messages.png?raw=True)
In the above plot I separated the sentiments for shiba and doge and then it can be seen that the above trends of greater number of negative messages is present in both of them.

## Running the Code

To install:
```sh
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

To generate the intermediate files containing results of sentiment analysis:
```sh
python3 crypto_sentiment.py
```

To compute plots:
```sh
python3 plot_sentiment.py  
```
