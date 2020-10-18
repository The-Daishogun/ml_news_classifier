import pandas as pd
import hazm
import matplotlib.pyplot as plt
from stopwords import all_stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


normalizer = hazm.Normalizer(persian_numbers=True)
tokenizer = hazm.WordTokenizer(replace_numbers=True, replace_hashtags=True)
lemmatizer = hazm.Lemmatizer()
tagger = hazm.POSTagger(model='resources/postagger.model')

def clean_text(sentence):
    sentence = normalizer.normalize(sentence)
    sentence = tokenizer.tokenize(sentence)
    sentence = tagger.tag(sentence)
    sentence = [lemmatizer.lemmatize(x[0], pos=x[1]) for x in sentence]
    return " ".join(sentence)

data = pd.read_csv('Corpus/persica_org_english_cat.csv')
data = data[['title', 'text', 'category2']].dropna()
data['cleaned_text'] = data["title"] + " " + data['text']

doc = ""
for item in data['cleaned_text']:
    doc += item + " "
doc = doc.split()

print("Number of Entries: {:,}\nNumber of tokens: {:,}\nVocabulary size: {:,}\nNumber of Classes: {:,}\n".format(
    data.shape[0], 
    len(doc),
    len(set(doc)),
    len(set(data['category2']))
))

le = LabelEncoder()
data['num_cat'] = le.fit_transform(data['category2'])

# Shuffling the Data to get a more realistic evaluation
data = data.sample(frac=1)

# Defining Features and Labels
X = data['cleaned_text']
y = data['num_cat']

# Spliting the Data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

# The PipeLine Containing the Vectorizer and The Model With Text PreProcessing
sgd = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(2, 3), lowercase=False, max_df=0.1, preprocessor=clean_text,
                              stop_words=all_stopwords)),
    ('clf', SGDClassifier(loss='modified_huber', max_iter=1300, n_jobs=-1, n_iter_no_change=80)),
])

# Fitting the Model with Training Data
sgd.fit(X_train, y_train)

# Making Predictions Using the Model on Test Data
pred = sgd.predict(X_test)

print(classification_report(y_test, pred, target_names=list(le.classes_)))