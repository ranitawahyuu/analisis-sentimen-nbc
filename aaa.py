import os
from flask import Flask, render_template, request,redirect, url_for
import tweepy
from werkzeug.utils import secure_filename
import re
import csv
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')
import googletrans
from googletrans import Translator
from textblob import TextBlob

app = Flask(__name__, static_folder="templates/assets")
  
  
hasil_analisis = []

hasil_labelling = []

#scrapping data menggunakan kata kunci
def scrapping_data_query(query, jumlah):
    apikey = "KqhjM6vhpOa94wHKf8q0ndsxu"
    apikeysecret = "SjNZLqHxKWClGsgqfVE1vJivZvX6NJqd1U0BCaYWc8mXcOTG33"
    accesstoken = "1280139539361566721-Nr0kZcpHBoWvUOxrfuQEFVdOeSr94j"
    accesstokensecret = "EIYWvjHXnKeX0vM37B06OM9uSEPi3yyZnpDrU4gizGq04"

    auth = tweepy.OAuthHandler(apikey, apikeysecret)
    auth.set_access_token(accesstoken, accesstokensecret)
    api = tweepy.API(auth,wait_on_rate_limit=True)

    #membuat file scrapping csv
    file = open('templates/assets/files/Scrapping Ranita.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)  

    #membuat file prepocessing csv
    filePre = open('templates/assets/files/Preprocessing Ranita.csv', 'w', newline='', encoding='utf-8')
    writerPre = csv.writer(filePre)  

    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="id").items(int(jumlah)):
        # proses scrapping kata kunci
        tweet_properties = {}
        tweet_properties["tanggal_tweet"] = tweet.created_at
        tweet_properties["username"] = tweet.user.screen_name
        tweet_properties["tweet"] = tweet.text
       
        # proses clean
        clean = re.sub(r'(\\x(.){2})', '', tweet_properties["tweet"])
        clean = ' '.join(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", clean).split())
        #clean = clean[1:]
        #clean = clean.replace('RT', '')
        #clean = clean.replace('RT', '')
        
        clean = clean.strip()

        # proses casefold
        casefold = clean.casefold()

        # proses tokenize
        tokenizing = nltk.tokenize.word_tokenize(casefold)


        # proses stop removal
        # mengambil data stop word dari library
        stop_factory = StopWordRemoverFactory().get_stop_words()
        # menambah stopword sendiri
        more_stop_word = [ "apa", "yg", "si"]
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

        if tweet.retweet_count > 0:
            if tweet_properties not in hasil_analisis:
                hasil_analisis.append([tweet_properties['tanggal_tweet'], tweet_properties['username'], tweet.text.encode('utf-8'), clean, casefold, tokenizing, stop_wr, stemming])
        else:
            hasil_analisis.append([tweet_properties['tanggal_tweet'], tweet_properties['username'], tweet.text.encode('utf-8'), clean, casefold, tokenizing, stop_wr, stemming ])

        tweets =[tweet.created_at, tweet.id, tweet.user.screen_name, tweet.text.encode('utf-8')]
        tweetsPre =[clean, casefold, tokenizing, stop_wr, stemming]
        
        #menuliskan data ke csv
        writer.writerow(tweets)
        writerPre.writerow(tweetsPre)

#scrapping menggunakan tanggal
def scrapping_data_tanggal(query, dari, sampai):
    apikey = "KqhjM6vhpOa94wHKf8q0ndsxu"
    apikeysecret = "SjNZLqHxKWClGsgqfVE1vJivZvX6NJqd1U0BCaYWc8mXcOTG33"
    accesstoken = "1280139539361566721-Nr0kZcpHBoWvUOxrfuQEFVdOeSr94j"
    accesstokensecret = "EIYWvjHXnKeX0vM37B06OM9uSEPi3yyZnpDrU4gizGq04"

    auth = tweepy.OAuthHandler(apikey, apikeysecret)
    auth.set_access_token(accesstoken, accesstokensecret)
    api = tweepy.API(auth,wait_on_rate_limit=True)

    #membuat file csv
    file = open('templates/assets/files/Scrapping Ranita.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)  

    #membuat file prepocessing csv
    filePre = open('templates/assets/files/Preprocessing Ranita.csv', 'w', newline='', encoding='utf-8')
    writerPre = csv.writer(filePre)  

    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="id", since=dari, until=sampai).items():
        tweet_properties = {}
        tweet_properties["tanggal_tweet"] = tweet.created_at
        tweet_properties["username"] = tweet.user.screen_name
        tweet_properties["tweet"] = tweet.text
       
        # proses clean
        clean = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", tweet_properties["tweet"]).split())

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

        if tweet.retweet_count > 0:
            if tweet_properties not in hasil_analisis:
                hasil_analisis.append([tweet_properties['tanggal_tweet'], tweet_properties['username'], tweet.text.encode('utf-8'), clean, casefold, tokenizing, stop_wr, stemming])
        else:
            hasil_analisis.append([tweet_properties['tanggal_tweet'], tweet_properties['username'], tweet.text.encode('utf-8'), clean, casefold, tokenizing, stop_wr, stemming ])

        tweets =[tweet.created_at, tweet.id, tweet.user.screen_name, tweet.text.encode('utf-8')]
        tweetsPre =[clean, casefold, tokenizing, stop_wr, stemming]
        
        #menuliskan data ke csv
        writer.writerow(tweets)
        writerPre.writerow(tweetsPre)

#proses prepocessing
def open_file_preprocessing(name):
    #membuat file csv
    file = open('templates/assets/files/Preprocessing Ranita.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)  
    with open('templates/assets/files/' + name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter =',')
        for row in readCSV:
            # proses clean
            clean = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", row[3]).split())

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
            tweets =[row[0], row[2], row[3], clean, casefold, tokenizing, stop_wr, stemming]
            #menuliskan data ke csv
            writer.writerow(tweets)
            hasil_analisis.append(tweets)

#proses labelling menggunakan text bloob
def labelling_process():
    # Membuat File CSV
    file = open('templates/assets/files/Labelling Ranita.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    translator = Translator()
 
    with open("templates/assets/files/Preprocessing Ranita.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter =',')
        hasil_labelling.clear()
        for row in readCSV:
            tweet = {}
            value = translator.translate(row[4], dest='en')
            terjemahan = value.text
            data_label = TextBlob(terjemahan)

            if data_label.sentiment.polarity > 0.0 :
                tweet['sentiment'] = "Positif"
            elif data_label.sentiment.polarity == 0.0 :
                tweet['sentiment'] = "Netral"
            else : 
                tweet['sentiment'] = "Negatif"

            labelling = tweet['sentiment']
            tweets =[row[0], row[4], labelling]
            hasil_labelling.append(tweets)

            writer.writerow(tweets)


           
#menjalankan menu index.html
@app.route('/')
def index():
    return render_template('index.html')

#menjalankan menu scrapping.html
@app.route('/scrapping',  methods=['GET', 'POST'])
def scrapping():
    if request.method == 'POST':
        if request.form.get('scrapping') == 'Scrapping':
            query = request.form.get('query')
            jumlah = request.form.get('jumlah')
            hasil_analisis.clear()
            scrapping_data_query(query, jumlah)
            return render_template('scrapping.html', value=hasil_analisis)
        
        elif request.form.get('scrapping-tanggal') == 'Scrapping':

            since = request.form.get('since')
            query = request.form.get('query2')
            
            until = request.form.get('until')
            hasil_analisis.clear()
            
            scrapping_data_tanggal(query, since, until)
            return render_template('scrapping.html', value=hasil_analisis)


    return render_template('scrapping.html', value=hasil_analisis)

#menjalankan menu prepocessing.html   
@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        if request.form.get('lanjutkan') == 'Lanjutkan':
            labelling_process()
            return redirect(url_for('labelling', value=hasil_labelling))


        # file = request.files['filecsv']
        # if request.form.get('upload') == 'Upload':
        #     if file.filename == '':
        #         return 'filename kosong'


        #     if file and allowed_files(file.filename):
        #         filename = secure_filename(file.filename)
        #         file.save(os.path.join((app.config['UPLOAD_FOLDER'] + filename))) 

        #         open_file_preprocessing(filename)
        #         return render_template('preprocessing.html', value=hasil_analisis)
                
        #     else:
        #         return 'format salah'

    return render_template('preprocessing.html', value=hasil_analisis)

#menjalankan menu labelling.html 
@app.route('/labelling')
def labelling():
    return render_template('Labelling.html', value=hasil_labelling)    

#menjalankan menu klasifikasi.html 
@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')



ALLOWED_EXTENSION = set(['csv'])
app.config['UPLOAD_FOLDER'] = 'files/'
def allowed_files(filename):
    return '.'  in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION



if __name__ == '__main__':
    app.run(debug=True)

