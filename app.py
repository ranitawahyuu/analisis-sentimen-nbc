from glob import glob
import re
from flask import Flask, app, render_template, request, url_for, flash
import flask
from nltk.util import pr
import tweepy
import csv
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')
import googletrans
import pandas as pd
from googletrans import Translator
from textblob import TextBlob
from werkzeug.utils import redirect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time


app = Flask(__name__, static_folder="templates/assets")

hasil_scrapping =[]
hasiL_preprocessing =[]
hasil_labelling=[]

app.config['SECRET_KEY'] = 'ranita'


ALLOWED_EXTENSION = set(['csv'])


def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION
def prepropecossing_twitter():
    # Membuat File CSV
    file = open('templates/assets/files/Data Preprocessing Ranita.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)

    hasiL_preprocessing.clear()

    with open("templates/assets/files/Data Scrapping Ranita.csv", "r",encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter =',')
        hasil_labelling.clear()
        for row in readCSV:
            # proses clean
            clean = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", row[2]).split())
            clean = re.sub("\d+", "", clean)
            

            # proses casefold
            casefold = clean.casefold()

            # proses tokenize
            tokenizing = nltk.tokenize.word_tokenize(casefold)


            # proses stop removal
            # mengambil data stop word dari library
            stop_factory = StopWordRemoverFactory().get_stop_words()
            # menambah stopword sendiri
            more_stop_word = [ "apa", "yg"]
            # menggabungkan stopword library + milik sendiri
            data = stop_factory + more_stop_word
            
            dictionary = ArrayDictionary(data)
            str = StopWordRemover(dictionary)
            stop_wr = nltk.tokenize.word_tokenize(str.remove(casefold))

            # proses stemming
            kalimat = ' '.join(stop_wr)
            factory = StemmerFactory()
            # mamanggil fungsi stemming
            stemmer = factory.create_stemmer()
            stemming = stemmer.stem(kalimat)


            tweets =[row[0], row[1], row[2], clean, casefold, tokenizing, stop_wr, stemming]
            hasiL_preprocessing.append(tweets)

            writer.writerow(tweets)
            flash('Preprocessing Berhasil', 'preprocessing_data')



df= None
df2 = None
akurasi = 0
def klasifikasi_data():
    global df
    global df2
    global akurasi
    # membca csv
    data = pd.read_csv("templates/assets/files/Data Labelling Ranita.csv")
    tweet = data.iloc[:, 1]
    y =  data.iloc[:, 2]


    # kalimat ke angka
    vec = CountVectorizer()
    x = vec.fit_transform(tweet)
    # tfidf
    tf_transform = TfidfTransformer().fit(x)
    x = tf_transform.transform(x)


    xtoarray = x.toarray()
    xshape = x.shape
    xnnz = x.nnz


    # split data training dan testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # naive bayes
    # naove bayes
    clf = MultinomialNB()
    clf.fit(x_train, y_train)

    predict = clf.predict(x_test)
    report = classification_report(y_test, predict, output_dict=True)
    # simpan ke csv
    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv('templates/assets/files/Data Hasil Klasifikasi.csv', index= True)




    unique_label = np.unique([y_test, predict])
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, predict, labels=unique_label), 
        index=['true:{:}'.format(x) for x in unique_label], 
        columns=['pred:{:}'.format(x) for x in unique_label]
    )


    
    cmtx.to_csv('templates/assets/files/Data Confusion Matrix.csv', index= True)

    df = pd.read_csv('templates/assets/files/Data Confusion Matrix.csv', sep=",")
    df.rename( columns={'Unnamed: 0':''}, inplace=True )

    df2 = pd.read_csv('templates/assets/files/Data Hasil Klasifikasi.csv', sep=",")
    df2.rename( columns={'Unnamed: 0':''}, inplace=True )

    akurasi = round(accuracy_score(y_test, predict)  * 100, 2)
    

    


            
def crawling_twitter_query(query, jumlah):
    api_key = "XaYBZDIN7j6xVHdPRIfOsu6mJ"
    api_secret_key = "CQKtU7Bpi8xmzhfLgqFguABTBOvBUwS8KQMdQk5A1HAttLKLSx"
    access_token = "1280139539361566721-e7s1tA20RyOTUAqAPT5dxiMawuUOWe"
    access_token_secret = "7PhFWlLR6eHVNbmtLqKjBaGJnQjQRUAp9fkcL2CUpwb4B"


    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    filter = " -filter:retweets"
    # Membuat File CSV
     #membuat file scrapping csv
    file = open('templates/assets/files/Data Scrapping Ranita.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)


    
    hasil_scrapping.clear()

    for tweet in tweepy.Cursor(api.search, q=query + filter, lang='id', tweet_mode="extended").items(int(jumlah)):
        tweet_properties = {}
        tweet_properties["tanggal_tweet"] = tweet.created_at
        tweet_properties["username"] = tweet.user.screen_name
        tweet_properties["tweet"] =  tweet.full_text.replace('\n', '')
        
        
 
        # Menuliskan data ke csv
        tweets =[tweet.created_at, tweet.user.screen_name, tweet.full_text.replace('\n', '')]
        if tweet.retweet_count > 0:
            if tweet_properties not in hasil_scrapping:
                hasil_scrapping.append(tweets)
        else:
            hasil_scrapping.append(tweets)

        writer.writerow(tweets)


def crawling_twitter_tanggal(query, sejak, sampai):
    api_key = "XaYBZDIN7j6xVHdPRIfOsu6mJ"
    api_secret_key = "CQKtU7Bpi8xmzhfLgqFguABTBOvBUwS8KQMdQk5A1HAttLKLSx"
    access_token = "1280139539361566721-e7s1tA20RyOTUAqAPT5dxiMawuUOWe"
    access_token_secret = "7PhFWlLR6eHVNbmtLqKjBaGJnQjQRUAp9fkcL2CUpwb4B"



    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    filter = " -filter:retweets"
    # Membuat File CSV
     #membuat file scrapping csv
    file = open('templates/assets/files/Data Scrapping Ranita.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)  

    
    hasil_scrapping.clear()

    for tweet in tweepy.Cursor(api.search, q=query + filter, lang='id', since=sejak, until=sampai,  tweet_mode="extended").items():
        tweet_properties = {}
        tweet_properties["tanggal_tweet"] = tweet.created_at
        tweet_properties["username"] = tweet.user.screen_name
        tweet_properties["tweet"] =  tweet.full_text.replace('\n', '')
        
        
 
        # Menuliskan data ke csv
        tweets =[tweet.created_at, tweet.user.screen_name, tweet.full_text.replace('\n', '')]
        if tweet.retweet_count > 0:
            if tweet_properties not in hasil_scrapping:
                hasil_scrapping.append(tweets)
        else:
            hasil_scrapping.append(tweets)

        writer.writerow(tweets)

def labelling_process():
    # Membuat File CSV
    file = open('templates/assets/files/Data Labelling Ranita.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    translator = Translator()

    with open("templates/assets/files/Data Preprocessing Ranita.csv", "r",encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter =',')
        hasil_labelling.clear()
        for row in readCSV:
            tweet = {}
            value = translator.translate(row[6], dest='en')
            terjemahan = value.text
            data_label = TextBlob(terjemahan)


            if data_label.sentiment.polarity > 0.0 :
                tweet['sentiment'] = "Positif"
            elif data_label.sentiment.polarity == 0.0 :
                tweet['sentiment'] = "Netral"
            else : 
                tweet['sentiment'] = "Negatif"

            labelling = tweet['sentiment']
            tweets =[row[2], row[7], labelling]
            hasil_labelling.append(tweets)

            writer.writerow(tweets)
            flash('Labelling Berhasil', 'labelling_data')
            
        

@app.route('/')
def index():
    return render_template('index.html')

#menjalankan menu scrapping.html
@app.route('/scrapping',  methods=['GET', 'POST'])
def scrapping():
    if request.method == 'POST':
        if request.form.get('lanjutkan') == 'Lanjutkan':
            prepropecossing_twitter()
            return redirect(url_for('preprocessing'))
        if request.form.get('scrapping') == 'Scrapping':
            query = request.form.get('query')
            jumlah = request.form.get('jumlah')
            hasil_scrapping.clear()
            crawling_twitter_query(query, jumlah)
            return render_template('scrapping.html', value=hasil_scrapping)
        
        elif request.form.get('scrapping-tanggal') == 'Scrapping':

            since = request.form.get('since')
            kata_kunci = request.form.get('kata_kunci')
            
            until = request.form.get('until')
            hasil_scrapping.clear()
            
            crawling_twitter_tanggal(kata_kunci, since, until)
            return render_template('scrapping.html', value=hasil_scrapping)


    return render_template('scrapping.html', value=hasil_scrapping)



    
@app.route('/preprocessing', methods = ['POST', 'GET'])
def preprocessing():
    if request.method == 'POST':
        if request.form.get('upload_file') == 'Upload':
            file = request.files['file']
            if not allowed_files(file.filename):
                flash('Format Salah', 'upload_category')
                return render_template('preprocessing.html', value=hasiL_preprocessing)
            if file and allowed_files(file.filename):
                flash('Upload Berhasil', 'upload_category')
                file.save("templates/assets/files/Data Scrapping Ranita.csv")
                return render_template('preprocessing.html', value=hasiL_preprocessing)

        hasiL_preprocessing.clear()
        if request.form.get('preprocessing_process') == 'Preprocessing Data':
            prepropecossing_twitter()
            
            return render_template('preprocessing.html', value=hasiL_preprocessing)
        if request.form.get('lanjutkan') == 'Lanjutkan':
            labelling_process()
            return redirect(url_for('labelling'))
            
    return render_template('preprocessing.html', value=hasiL_preprocessing)

@app.route('/klasifikasi',  methods= ['POST', 'GET'])
def klasifikasi():
    # if request.method == 'POST':
    #     if request.form.get('matriks') == 'matriks':

            

    return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-striped', index=False)], titles=df.columns.values, tables2=[df2.to_html(classes='table table-striped', index=False)], titles2=df2.columns.values)
            

@app.route('/labelling', methods= ['POST', 'GET'])
def labelling():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload':
            file = request.files['filecsv']
            if not allowed_files(file.filename):
                flash('Format Salah', 'upload_category')
                return render_template('Labelling.html', value=hasil_labelling)
            if file and allowed_files(file.filename):
                flash('Upload Berhasil', 'upload_category')
                file.save("templates/assets/files/Data Preprocessing Ranita.csv")
                return render_template('Labelling.html', value=hasil_labelling)

        hasil_labelling.clear()
        if request.form.get('labelling_data') == 'Labelling Data':
            labelling_process()
            
            return render_template('Labelling.html', value=hasil_labelling)
        if request.form.get('lanjutkan') == 'Lanjutkan':
            klasifikasi_data()
            return redirect(url_for('klasifikasi'))


    return render_template('Labelling.html', value=hasil_labelling)



if __name__ == '__main__':
    app.run(debug=True)

