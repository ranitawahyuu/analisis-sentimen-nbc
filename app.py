
from glob import glob
import re
from tkinter.tix import Y_REGION
from flask import Flask, app, render_template, request, url_for, flash
from matplotlib import transforms
from nltk.util import pr
import tweepy
import csv
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')
from wordcloud import WordCloud
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import urllib.request
import pandas as pd
from googletrans import Translator
from textblob import TextBlob
from werkzeug.utils import redirect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import joblib
import pickle


app = Flask(__name__, static_folder="templates/assets")

hasil_scrapping =[]
hasiL_preprocessing =[]
hasil_labelling=[]

app.config['SECRET_KEY'] = 'ranita'


ALLOWED_EXTENSION = set(['csv'])


def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION
def prepropecossing_twitter():
    global normalizad_word_dict
    normalizad_word = pd.read_excel("templates/assets/normalisasi.xlsx")

    normalizad_word_dict = {}
    for index, row in normalizad_word.iterrows():
        if row[0] not in normalizad_word_dict:
            normalizad_word_dict[row[0]] = row[1]
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
            clean = re.sub(r"\b[a-zA-Z]\b", "", clean)
            

            # proses casefolding
            casefold = clean.casefold()

            # proses normalize
            normalisasi = proses_normalisasi(casefold)

            # proses tokenize
            tokenizing = nltk.tokenize.word_tokenize(normalisasi)


            # proses stop removal
            # mengambil data stop word dari library
            stop_factory = StopWordRemoverFactory().get_stop_words()
            # menambah stopword sendiri
            more_stop_word = [ "apa", "yg", 'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo',
                  'kalo', 'amp', 'biar', 'bikin', 'bilang',
                  'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                  'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                  'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'dm',
                  'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                  '&amp', 'yah', 'hallo', 'halo', 'hello', 'bgt', 'td',
                  'no', 'yaa', 'ae', 'kali', 'segera', 'rd', 'kak', 'gmn', 'min']
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
            

            tweets =[row[0], row[1], row[2], clean, casefold, normalisasi, tokenizing, stop_wr, stemming]
            hasiL_preprocessing.append(tweets)

            writer.writerow(tweets)
            flash('Preprocessing Berhasil', 'preprocessing_data')


def normalized_term(mosok):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in mosok]

def proses_normalisasi(data):
    tokens = nltk.tokenize.word_tokenize(data)
    hasil = normalized_term(tokens)
    kalimat = ' '.join(hasil)
    return kalimat

df= None
df2 = None

akurasi = 0

def klasifikasi_data():
    global df
    global df2
    global akurasi

    # membaca csv 
    data = pd.read_csv("templates/assets/files/Data Labelling Ranita.csv")
    tweet = data.iloc[:, 1]
    y =  data.iloc[:, 2]

    # Vectorize text reviews to numbers
    # vec = CountVectorizer()
    # x = vec.fit_transform(tweet)
    
    # # tfidf
    # tf_transform = TfidfTransformer().fit(x)
    # x = tf_transform.transform(x)


    #split data training dan testing
    x_train, x_test, y_train, y_test = train_test_split(tweet,y, test_size=0.2, random_state=42)
     # tfidf
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(tweet)
    # tfidf = vectorizer.fit_transform(X_train)
    names = vectorizer.get_feature_names()
    Train_X_Tfidf= vectorizer.transform(x_train)
    Test_X_Tfidf= vectorizer.transform(x_test)

    # naive  bayes
    clf = MultinomialNB()
    clf.fit(Train_X_Tfidf, y_train)

    # menyimpan tfidf
    df_train_tfidf = pd.DataFrame(data=csr_matrix.todense(Train_X_Tfidf))
    df_train_tfidf.columns = names
    df_train_tfidf.index = y_train
    df_train_tfidf.to_csv('templates/assets/files/TFIDF Training.csv')


    df_test_tfidf = pd.DataFrame(data=csr_matrix.todense(Test_X_Tfidf))
    df_test_tfidf.columns = names
    df_test_tfidf.index = y_test
    df_test_tfidf.to_csv('templates/assets/files/TFIDF Testing.csv')


    



    # joblib.dump(clf, 'templates/assets/files/model.pkl') 
    # joblib.dump(tf_transform_train, 'templates/assets/files/tfidf.pkl') 
    # joblib.dump(vec, 'templates/assets/files/countvec.pkl') 
    pickle.dump(vectorizer, open('templates/assets/files/countvec.pkl', 'wb'))
    pickle.dump(Train_X_Tfidf, open('templates/assets/files/tfidf.pkl', 'wb'))
    pickle.dump(clf, open('templates/assets/files/model.pkl', 'wb'))

    predict = clf.predict(Test_X_Tfidf)

    report = classification_report(y_test, predict, output_dict=True)
    # simpan ke csv
    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv('templates/assets/files/Data Hasil Klasifikasi.csv', index= True)

    unique_label = np.unique([y_test, predict])
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, predict, labels=unique_label), 
        index=['{:}'.format(x) for x in unique_label], 
        columns=['{:}'.format(x) for x in unique_label]
    )
    
    cmtx.to_csv('templates/assets/files/Data Confusion Matrix.csv', index= True)

    df = pd.read_csv('templates/assets/files/Data Confusion Matrix.csv', sep=",")
    df.rename( columns={'Unnamed: 0':''}, inplace=True )

    df2 = pd.read_csv('templates/assets/files/Data Hasil Klasifikasi.csv', sep=",")
    df2.rename( columns={'Unnamed: 0':''}, inplace=True )

    akurasi = round(accuracy_score(y_test, predict)  * 100, 2)

    kalimat = ""

    for i in tweet.tolist():
        s =("".join(i))
        kalimat += s

    urllib.request.urlretrieve("https://firebasestorage.googleapis.com/v0/b/sentimen-97d49.appspot.com/o/Circle-icon.png?alt=media&token=b9647ca7-dfdb-46cd-80a9-cfcaa45a1ee4", 'love.png')
    mask = np.array(Image.open("love.png"))
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color='white', mask = mask)
    wordcloud.generate(kalimat)
    
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('templates/assets/files/wordcloud.png')

    # diagram
    numbers_list = y_test.tolist()
    counter = dict((i, numbers_list.count(i)) for i in numbers_list)
    isPositive = 'Positif' in counter.keys()
    isNegative = 'Negatif' in counter.keys()
    isNeutral = 'Netral' in counter.keys()

    positif = counter["Positif"] if isPositive == True  else 0
    negatif = counter["Negatif"] if isNegative == True  else 0
    netral = counter["Netral"] if isNeutral == True  else 0

    sizes = [positif, netral, negatif]
    labels = ['Positif', 'Netral', 'Negatif']
    plt.pie(sizes, labels=labels, autopct='%1.0f%%', shadow=True, textprops={'fontsize': 20})
    plt.savefig('templates/assets/files/diagram.png')

    # diagram batang
    # creating the bar plot

    plt.figure()

    plt.hist(numbers_list)
    
    plt.xlabel("Tweet tentang Xiaomi")
    plt.ylabel("Jumlah Tweet")
    plt.title("Presentase Sentimen Tweet Xiaomi Redmi")
    plt.savefig('templates/assets/files/diagram-batang.png')
    
        


hasil_model_predict = []
def model_predict():
    global df
    global df2
    global akurasi
    # membca csv
    data = pd.read_csv("templates/assets/files/Data Labelling Model Predict Ranita.csv")
    tweet = data.iloc[:, 1]
    y = data.iloc[:, 2]
    
    # Vectorize text reviews to numbers
    # tfidf = joblib.load('templates/assets/files/tfidf.pkl')
    # nb = joblib.load('templates/assets/files/model.pkl')
    # vec = joblib.load('templates/assets/files/countvec.pkl')
    with open('templates/assets/files/model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    with open('templates/assets/files/countvec.pkl', 'rb') as h:
        vec = pickle.load(h)
   
    with open('templates/assets/files/tfidf.pkl', 'rb') as t:
        tfidf = pickle.load(t)

    file = open('templates/assets/files/Data Hasil Model Predict.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    for i, line in data.iterrows():
        isi = line[1]
        label = line[2]
        # # transform cvector & tfidf
        transform_cvec = vec.transform([isi])
        transform_tfidf = tfidf.transform(transform_cvec)
        print(transform_tfidf)
        # predict start
        predic_result = model.predict(transform_tfidf)
        print(predic_result)

        data = [isi , predic_result[0], label]
        hasil_model_predict.append(data)
        writer.writerow(data)
        
        
    # # x2 = vec.fit_transform(x_test)
    # # # tfidf
    # tf_transform_train = TfidfTransformer().fit(x)
    # x = tf_transform_train.transform(x)

    # tf_transform_test = TfidfTransformer().fit(x2)
    # x2 = tf_transform_test.transform(x2)

    # naive  bayes

    # clf = MultinomialNB()
    # clf.fit(x, y_train)

    # x = vec.fit_transform(tweet)
    # transform_tfidf = tfidf.transform(x)
    # print(transform_tfidf, flush=True)



    # file = open('templates/assets/files/Data Hasil Model Predict.csv', 'w', newline='', encoding='utf-8')
    # writer = csv.writer(file)
    # hasil_model_predict.clear()
    # for i in range(len(predict)):
    #     data = [tweet.tolist()[i],y.tolist()[i] , predict[i]]
    #     hasil_model_predict.append(data)
    #     writer.writerow(data)

    # report = classification_report(y, predict, output_dict=True)
    # # simpan ke csv
    # clsf_report = pd.DataFrame(report).transpose()
    # clsf_report.to_csv('templates/assets/files/Data Hasil Klasifikasi.csv', index= True)




    # unique_label = np.unique([y, predict])
    # cmtx = pd.DataFrame(
    #     confusion_matrix(y, predict, labels=unique_label), 
    #     index=['{:}'.format(x) for x in unique_label], 
    #     columns=['{:}'.format(x) for x in unique_label]
    # )


    
    # cmtx.to_csv('templates/assets/files/Data Confusion Matrix.csv', index= True)

    # df = pd.read_csv('templates/assets/files/Data Confusion Matrix.csv', sep=",")
    # df.rename( columns={'Unnamed: 0':''}, inplace=True )

    # df2 = pd.read_csv('templates/assets/files/Data Hasil Klasifikasi.csv', sep=",")
    # df2.rename( columns={'Unnamed: 0':''}, inplace=True )

    # akurasi = round(accuracy_score(y, predict)  * 100, 2)

    kalimat = ""

    for i in tweet.tolist():
        s =("".join(i))
        kalimat += s 
    urllib.request.urlretrieve("https://firebasestorage.googleapis.com/v0/b/sentimen-97d49.appspot.com/o/Circle-icon.png?alt=media&token=b9647ca7-dfdb-46cd-80a9-cfcaa45a1ee4", 'love.png')
    mask = np.array(Image.open("love.png"))
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color='white', mask = mask)
    wordcloud.generate(kalimat)
    plt.figure(figsize=(12,10))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.savefig('templates/assets/files/wordcloud2.png')

    # diagram
    numbers_list = y.tolist()
    counter = dict((i, numbers_list.count(i)) for i in numbers_list)
    isPositive = 'Positif' in counter.keys()
    isNegative = 'Negatif' in counter.keys()
    isNeutral = 'Netral' in counter.keys()

    positif = counter["Positif"] if isPositive == True  else 0
    negatif = counter["Negatif"] if isNegative == True  else 0
    netral = counter["Netral"] if isNeutral == True  else 0

    sizes = [positif, netral, negatif]
    labels = ['Positif', 'Netral', 'Negatif']
    plt.pie(sizes, labels=labels, autopct='%1.0f%%', shadow=True, textprops={'fontsize': 20})
    plt.savefig('templates/assets/files/diagram2.png')

    # diagram batang
    # creating the bar plot

    plt.figure()

    plt.hist(numbers_list)
    
    plt.xlabel("Tweet tentang Xiaomi")
    plt.ylabel("Jumlah Tweet")
    plt.title("Presentase Sentimen Tweet Xiaomi Redmi")
    plt.savefig('templates/assets/files/diagram-batang2.png')
    flash('Model Predict Berhasil', 'model_berhasil')

            
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
        tweet_properties["tweet"] =  tweet.full_text.replace('\n', ' ')
        
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
            try:
                value = translator.translate(row[8], dest='en')
            except:
                print("Terjadi kesalahan", flush=True)
            terjemahan = value.text
            data_label = TextBlob(terjemahan)


            if data_label.sentiment.polarity > 0.0 :
                tweet['sentiment'] = "Positif"
            elif data_label.sentiment.polarity == 0.0 :
                tweet['sentiment'] = "Netral"
            else : 
                tweet['sentiment'] = "Negatif"

            labelling = tweet['sentiment']
            tweets =[row[1], row[8], labelling]
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
    if request.method == "POST":
        if request.form.get('upload_file') == 'Upload':
            file = request.files['file']
            if not allowed_files(file.filename):
                flash('Format Salah', 'upload_category')
                return render_template('klasifikasi.html')
            if file and allowed_files(file.filename):
                flash('Upload Berhasil', 'upload_category')
                file.save("templates/assets/files/Data Labelling Ranita.csv")
                return render_template('klasifikasi.html')
        if request.form.get('klasifikasi') == 'Klasifikasi':
            klasifikasi_data()
            flash('Klasifikasi Berhasil', 'klasifikasi_data')
            return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-striped', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-striped', index=False, justify='left')], titles2=df2.columns.values)
       
    if akurasi == 0:
        return render_template('klasifikasi.html')
    else:
        return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-striped', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-striped', index=False, justify='left')], titles2=df2.columns.values)
       
            

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        if request.form.get('upload_file') == 'Upload':
            file = request.files['file']
            if not allowed_files(file.filename):
                flash('Format Salah', 'upload_category')
                return render_template('predict.html', value=hasil_model_predict)
            if file and allowed_files(file.filename):
                flash('Upload Berhasil', 'upload_category')
                file.save("templates/assets/files/Data Labelling Model Predict Ranita.csv")
                return render_template('predict.html', value=hasil_model_predict)

        hasil_model_predict.clear()
        if request.form.get('predict') == 'Model Predict':
            model_predict()
            return render_template('predict.html', value=hasil_model_predict)
        if request.form.get('visualisasi') == 'Visualisasi':
            return redirect(url_for('klasifikasi'))

        
    return render_template('predict.html', value=hasil_model_predict)


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
            flash('Klasifikasi Berhasil', 'klasifikasi_data')
            return redirect(url_for('klasifikasi'))


    return render_template('Labelling.html', value=hasil_labelling)



if __name__ == '__main__':
    app.run(debug=True)

