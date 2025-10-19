# Import packages
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import NMF
import gzip
from sklearn.preprocessing import Normalizer
from flask import Flask, request, render_template_string
# Read in 984 most popular Wiki names of July 2025.
base_dir = os.path.dirname(__file__)
input_file = os.path.join(base_dir, "topviews-2025_07.csv")
most_pop_wikis = pd.read_csv(input_file)
# Add in column Page_With_Underscore that replaces spaces with underscores in Wiki names.
most_pop_wikis["Page_With_Underscore"] = most_pop_wikis["Page"].str.replace(" ", "_")
# In the next lines of code, calculate or read in extracted text from the 984 most popular
# Wiki articles from July 2025. Save the result to the variable most_pop_wikis_text_list.
input_file = os.path.join(base_dir, "most-pop-wikis-text-list.pkl")
working_file = os.path.join(base_dir, "working", "most-pop-wikis-text-list.pkl")
if os.path.exists(input_file) == True:
  with open(input_file, "rb") as f:
    most_pop_wikis_text_list = pickle.load(f)
else:
  most_pop_wikis_text_list = []
  headers = {'User-Agent': 'RamiBot/0.0 (ramizmshawad@gmail.com)'}
  for page in most_pop_wikis["Page_With_Underscore"]:
    url = f"https://en.wikipedia.org/wiki/{page}"
    response = requests.get(url, headers = headers)
    response_text = response.text
    response_bs = BeautifulSoup(response_text)
    response_bs_text = response_bs.get_text()
    most_pop_wikis_text_list.append(response_bs_text)
  os.makedirs(os.path.dirname(working_file), exist_ok = True)
  with open(working_file, "wb") as f:
    pickle.dump(most_pop_wikis_text_list, f)
# Perform tf-idf (term frequency-inverse document frequency) on the extracted text
# of the 984 most popular Wikipedia articles of July 2025. Save the result to the
# tfidf_array variable.
tfidf = TfidfVectorizer()
tfidf_array = tfidf.fit_transform(most_pop_wikis_text_list)
# Perform calculation or reading in of the extraction of the champion 17 (found through 
# grid search) NMF (non-negative matrix factorization) topics from the tf-idf vectorized 
# text of the 984 most popular Wikipedia articles of July 2025, stored in the tfidf_array
# variable. Save the results in the nmf_features variable.
nmf_features_file_working = os.path.join(base_dir, "working", "nmf_features_17.pkl")
nmf_features_file_input = os.path.join(base_dir, "nmf_features_17.pkl")
if (os.path.exists(nmf_features_file_input) == True):
    with open(nmf_features_file_input, "rb") as f:
        nmf_features = pickle.load(f)
else:
    nmf = NMF(n_components = 17, init = "nndsvd", random_state = 42, max_iter = 200, verbose = 1)
    nmf_features = nmf.fit_transform(tfidf_array)
    os.makedirs(os.path.dirname(nmf_features_file_working), exist_ok = True)
    with gzip.open(nmf_features_file_working, "wb") as f:
        pickle.dump(nmf_features, f)
# Perform normalization on the 17 NMF topics that were extracted from the text of the
# 984 most popular Wikipedia articles of July 2025, stored in the variable nmf_features.
# Save the results to norm_features
normalizer = Normalizer()
# For the champion 17 NMF features/topics to be extracted, perform the fit_transform 
# method on the Normalizer instance normalizer to normalize the corresponding NMF 
# calculation results matrix, nmf_features, and store it in the variable, norm_features. 
# Convert the numpy array, norm_features, to a dataframe, norm_features_df.
norm_features = normalizer.fit_transform(nmf_features)
norm_features_df = pd.DataFrame(data = norm_features, index = most_pop_wikis["Page"], columns = [f"Topic {i + 1}" for i in range(norm_features.shape[1])])
# Create the Flask app and define a wikipedia_recommender_system function within
# the app to allow for the wikipedia recommender system to run. As a result, when 
# the user goes to the app link, the user inputs the Wikipedia article name they want 
# to get recommended articles for and also the user inputs the number of articles they want.
# Then the user clicks the "Get Recommended Articles" button and the recommended wikipedia articles
# are returned with their names, cosine similarity scores, and clickable hyperlinks.
app = Flask(__name__)
@app.route("/", methods = ["GET", "POST"])
def wikipedia_recommender_system():
    if request.method == "POST":
        article_name = request.form.get("article_name")
        possible_matches = [name for name in norm_features_df.index if name.lower() == article_name.lower()]
        if possible_matches:
            article_name = possible_matches[0]
        n_articles = request.form.get("n_articles")
        n_articles = int(n_articles)
        article = norm_features_df.loc[article_name, :]
        top_similar_articles_ser = norm_features_df.dot(article).nlargest(n = n_articles)
        top_similar_articles_df = pd.DataFrame(top_similar_articles_ser)
        top_similar_articles_df.columns = ["Cosine Similarity Score"]
        article_without_underscores = pd.Series(top_similar_articles_df.index)
        article_with_underscores = article_without_underscores.str.replace(" ", "_")
        url_list = []
        for i, j in zip(article_with_underscores, article_without_underscores):
            url = f'<a href="https://en.wikipedia.org/wiki/{i}" target="_blank">{j}</a>'
            url_list.append(url)
        top_similar_articles_df["Clickable Hyperlink"] = url_list
        top_similar_articles_df_html = top_similar_articles_df.to_html(escape = False)
        return render_template_string(f"<pre>\n"
                                      f"<h3>{n_articles} Most Similar Articles to {article_name} Recommended to Read Next</h3><br><br>\n"
                                      f"{top_similar_articles_df_html}<br><br>\n"
                                      f"<a href='/'>Go back to pick more recommended Wikipedia articles to read.</a>")
    return render_template_string("<h3> Wikipedia Recommender System</h3><br><br>\n"
        "<a href='https://pageviews.wmcloud.org/topviews/?project=en.wikipedia.org&platform=all-access&date=2025-07&excludes=' target='_blank'>Here is the url to the most popular Wikipedia articles of July 2025.</a><br><br>\n"
                                  "<pre>\n"
                                  "Please enter the Wikipedia article name where you want to get recommended\n"
                                  "similar articles to read. The url for options that you can pick is above.\n"
                                  "Please type in the exact article name (case sensitive) that you see in the\n"
                                  "Page column of that url. (Note: Type lowercase 'l' instead of capital 'L'\n"
                                  "we'll auto-fix it.). Please also enter the number of recommended similar\n" 
                                  "articles you want to read. Input a number like 3 or 5 and not a word like\n"
                                  "three or five.<br><br>\n"
                                  "</pre>\n"
                                  "<form method='post'>\n"
                                  "<label for='article_name'>Article Name:</label><br>\n"
                                  "<input type='text' id='article_name' name='article_name' required><br><br>\n"
                                  "<label for='n_articles'>Number of Recommended Articles To Read:</label><br>\n"
                                  "<input type='number' id='n_articles' name='n_articles' required><br><br>\n"
                                  "<input type='submit' value='Get Recommended Articles'>\n"
                                  "</form>")
# Run the flask app at the link http://127.0.0.1:5000/ locally or 
# https://wikipedia-recommender-system.onrender.com publicly, for 
# the Wikipedia recommender system to run. Set debug to True to be 
# in debug mode and use_reloader to False so the code works in Jupyter 
# Notebook or regular Python file.
if __name__ == "__main__":
    app.run(debug = True, use_reloader=False)



