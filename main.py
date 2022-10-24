from utils import *
import pandas as pd
import warnings

warnings.simplefilter("ignore")
from sklearn.feature_extraction.text import CountVectorizer
import re

space = re.compile("["u"\u200d""]+", flags=re.UNICODE)

import numpy as np
from camel_tools.utils.charsets import AR_CHARSET
from ftlangdetect import detect

# Concatinate all Arabic characters into a string
ar_str = u''.join(AR_CHARSET)
# Compile a regular expression using above string
arabic_re = re.compile(r'^[' + re.escape(ar_str) + r']+$')

#we have the following  6 categ names
categ_names = ['Arabizi', 'English', 'French ', 'Arabic ','Shared', 'Other  ']

#using the 200 most frequent char-2gram
char_vectorizer = CountVectorizer(analyzer='char', max_features=250, ngram_range=[1, 3])

#this function returns the feature vector for the recieved token
def get_feature_vector(token):
    chars_vectorized = char_vectorizer.transform([token]).toarray()[0]
    token = token.lower()
    ar_word = 0
    word_with_digit = 0
    frequent_en_word = 0
    frequent_fr_word = 0
    detected_en = 0
    detected_fr = 0
    detected_non_en_fr = 0
    detected_word = detect(text=token)
    if arabic_re.match(token) is not None:
        ar_word = 1
    if any(char.isdigit() for char in token) and token.isnumeric() == False:
        word_with_digit = 1
    if token in frequent_en_words:
        frequent_en_word = 1
    if token in frequent_fr_words:
        frequent_fr_word = 1
    if detected_word['lang'] == 'en' and detected_word['score'] > 0.95:
        detected_en = 1
    if detected_word['lang'] == 'fr' and detected_word['score'] > 0.5:
        detected_fr = 1
    else:
        detected_non_en_fr = 1
    my_vector = [frequent_en_word, frequent_fr_word, word_with_digit, ar_word]
    my_vector.extend([detected_en, detected_fr, detected_non_en_fr])
    my_vector.extend(chars_vectorized/len(token))
    return np.array(my_vector)


'''This function get annotated_file_name - the name of csv word annotated file 
    and returns
       structured_X : list of sentences - each sentence is a list of tokens
       structured_y : list of sentence_tags - each sentence_tag (entry) is a list of tags for the 
                      corresponding sentence in structured_X
'''
def preprocess_structured_data(annotated_file_name):
    #The na_filter=False is to inconsider the word "na" as blank (to consider it as it is)
    data = pd.read_csv(annotated_file_name, dtype=str,na_filter=False)
    structured_data = data.groupby(["sen_id", "sen_num"])["token", "categ"].agg(lambda x: list(x))
    '''Use The following line if you want to use the whole paragraph (tweet or comment) 
        and not the slices to sentences '''
    #structured_data = data.groupby(["sen_id"])["token", "categ"].agg(lambda x: list(x))
    structured_X = structured_data["token"]
    structured_y = structured_data["categ"]
    return [structured_X, structured_y]


def k_fold_cross_validation(X,Y,model,k=10):
    from sklearn.model_selection import KFold
    from sklearn import metrics
    support=np.array([0]*len(categ_names))
    precision_score_arr,recall_score_arr=np.array([0.0]*len(categ_names)),np.array([0.0]*len(categ_names))
    acc_score_arr_sen,precision_score_arr_sen,recall_score_arr_sen=np.array([0.0]*len(categ_names)),np.array([0.0]*len(categ_names)),np.array([0.0]*len(categ_names))
    f1_macro,precision_macro,recall_macro=0.0,0.0,0.0
    f1_weighted,precision_weighted,recall_weighted=0.0,0.0,0.0
    f1_micro,precision_micro,recall_micro=0.0,0.0,0.0
    kf=KFold(n_splits=k,shuffle=True)
    acc_score=0.0
    acc_score_sen_level=0.0
    total_supp=0
    for train_idx,test_idx in kf.split(X):
        X_train,X_test=X[train_idx],X[test_idx]
        y_train,y_test=Y[train_idx],Y[test_idx]
        X_train=[sent2features(s) for s in X_train]
        X_test=[sent2features(s) for s in X_test]
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_test_f=[val for sublist in y_test.tolist() for val in sublist]
        y_pred_f=[val for sublist in y_pred for val in sublist]
        p, r, f1, s = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,labels=['0','1','2','3','4','5'],average=None)
        for i in range(0,len(categ_names)) :
            precision_score_arr[i]+=p[i]
            recall_score_arr[i]+=r[i]
            support[i]+=s[i]

        acc_score+=metrics.accuracy_score(y_test_f,y_pred_f)
        p, r, f1, _ = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,average='macro')
        precision_macro+=p
        recall_macro+=r
        p, r, f1, _ = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,average='micro')
        precision_micro+=p
        recall_micro+=r
        p, r, f1, _ = metrics.precision_recall_fscore_support(y_test_f,y_pred_f,average='weighted')
        precision_weighted+=p
        recall_weighted+=r
        total_supp=sum(support)
        '''CRF when converting tags to binary (sentence level):'''
        y_pred_sen_level= []
        test_y_sents_sen_level=[]
        for tags in y_pred:
            binary_tags=[0,0,0,0,0,0]
            for tag in tags:
                binary_tags[int(tag)]=1
            y_pred_sen_level.append(binary_tags)
        for tags in y_test:
            binary_tags=[0,0,0,0,0,0]
            for tag in tags:
                binary_tags[int(tag)]=1
            test_y_sents_sen_level.append(binary_tags)

        for i in range(0, len(categ_names)):
            y_true=[sen[i] for sen in test_y_sents_sen_level]
            y_pred=[sen[i] for sen in y_pred_sen_level]
            precision_score_arr_sen[i]+=metrics.precision_score(y_true, y_pred)
            recall_score_arr_sen[i]+=metrics.recall_score(y_true, y_pred)
            acc_score_arr_sen[i]+=metrics.accuracy_score(y_true, y_pred)
        acc_score_sen_level+=metrics.accuracy_score(test_y_sents_sen_level,y_pred_sen_level)
    precision_macro=precision_macro/k
    recall_macro=recall_macro/k
    f1_macro=2.0*precision_macro*recall_macro/(precision_macro+recall_macro)
    precision_weighted=precision_weighted/k
    recall_weighted=recall_weighted/k
    f1_weighted=2.0*precision_weighted*recall_weighted/(precision_weighted+recall_weighted)
    precision_micro=precision_micro/k
    recall_micro=recall_micro/k
    f1_micro=2.0*precision_micro*recall_micro/(precision_micro+recall_micro)
    print("---- {0} fold cross validation of the model----".format(k))
    print(acc_score/k)
    print('Category     precision     recall      f1-score      support')
    for i in range(0, len(categ_names)):
        precision_score=precision_score_arr[i]/k
        recall_score=recall_score_arr[i]/k
        f1_score=2*precision_score*recall_score/(recall_score+precision_score)
        print(' {0}  :  {1:.2f}         {2:.2f}      {3:.2f}        {4}  '.format(categ_names[i],precision_score,recall_score,f1_score,support[i]))
    print('micro avg  : {0:.2f}         {1:.2f}      {2:.2f}        {3} '.format(precision_micro,recall_micro,f1_micro,total_supp))
    print('macro avg  : {0:.2f}         {1:.2f}      {2:.2f}        {3} '.format(precision_macro,recall_macro,f1_macro,total_supp))
    print('weighted avg:{0:.2f}         {1:.2f}      {2:.2f}        {3} '.format(precision_weighted,recall_weighted,f1_weighted,total_supp))

    print('10 fold CRF indirect  sentence level:')
    print("total acc score = {0:.3f}".format(acc_score_sen_level/k))
    print('Category     accuracy     precision    recall      f1-score')
    for i in range(0, len(categ_names)):
        precision_score=precision_score_arr_sen[i]/k
        recall_score=recall_score_arr_sen[i]/k
        f1_score=2*precision_score*recall_score/(recall_score+precision_score)
        acc_score=acc_score_arr_sen[i]/k
        print('{0} :    {1:.2f}          {2:.2f}         {3:.2f}         {4:.2f} '.format(categ_names[i],acc_score,precision_score,recall_score,f1_score))




# Functions for CRF

'''This function get as input :
    sent- the sentence
    i : the index of the token in sent
    tags: the tags of the given sentence (sent)
    and returns the features of the token at index i in sent
'''
def word2features(sent, i):
    word = sent[i]
    detected_word = detect(text=word)
    chars_vectorized = char_vectorizer.transform([word]).toarray()[0]
    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'ar_word':arabic_re.match(word) is not None,
        'word_with_digit':any(char.isdigit() for char in word) and word.isnumeric() is False,
        'frequent_en_word':word.lower() in frequent_en_words,
        'frequent_fr_word':word.lower() in frequent_fr_words,
        'detected_en':detected_word['lang'] == 'en' and detected_word['score'] > 0.95,
        'detected_fr':detected_word['lang'] == 'fr' and detected_word['score'] > 0.5
    }
    for iv,value in enumerate(chars_vectorized/len(word)):
        features['v{}'.format(iv)]=value
    if i > 0:
        word_before = sent[i - 1]
        features.update({
            '-1:word.lower()': word_before.lower().encode('utf-8'),
            '-1:word.istitle()': word_before.istitle(),
            '-1:word.isupper()': word_before.isupper(),
            '-1:ar_word':arabic_re.match(word_before) is not None,
            '-1:word_with_digit':any(char.isdigit() for char in word_before) and word_before.isnumeric() is False,
            '-1:frequent_en_word':word_before.lower() in frequent_en_words,
            '-1:frequent_fr_word':word_before.lower() in frequent_fr_words

        })
    else:
        features['BOS'] = True

    if i == len(sent) - 1:
        features['EOS'] = True

    return features

'''
This function returns a list of features of each token in the given sentence (and using the corresponding tags)
'''
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def remove_duplications(list):
    result=[]
    result.extend(list[0])
    for i in range(1,len(list)):
        if list[i]!=list[i-1]:
            result.extend(list[i])
    return result


def CS_shared_statistics(structured_y):
    count_pos=0
    count_neg=0
    structured_y= [y for y in structured_y if('0' in y and '1' in y and '4' in y)]
    for y_list in structured_y:
        y_list=remove_duplications([i for i in y_list if i!='5'])
        idx=0
        switch=False
        last_tag=y_list[0]
        shared_mentioned=False
        if last_tag=='4':
            shared_mentioned=True
            y_list=y_list[1:]
            last_tag=y_list[0]
            idx+=1
        while(idx<len(y_list) and not switch):
        #if(len(y_list)<2):continue
            if(y_list[idx]=='4'):
                shared_mentioned=True
            elif y_list[idx]!=last_tag:
                switch=True
            idx+=1
        if(shared_mentioned):
            count_pos+=1
        else:
            count_neg+=1
    print('number of switches preceded by shared is :{0}'.format(count_pos))
    print('number of switches that are not preceded by shared is :{0}'.format(count_neg))



'''This function do basic statistics and print the relationship between Shared and CS at the sentence level'''
def do_main_CS_statistics(structured_y):
    '''Start with Arabizi and English only'''
    count_CS_no_border=0
    count_CS_Follow=0
    count_Follow_no_CS=0
    count_no_Follow_no_Cs=0
    for y in structured_y:
        if('0' in y and '1' in y and '4' in y):
            count_CS_shared+=1
        elif('0' in y and '1' in y):
            count_CS_no_border+=1
        if (not('0' in y and '1' in 'y') and '4' in y):
            count_shared_no_CS+=1
        if (not('0' in y and '1' in 'y') and not('4' in y)):
            count_no_shared_no_Cs+=1
    print('         Contains Shared word')
    print('          yes         no')
    print('CS  yes   {0}         {1}'.format(count_CS_shared,count_CS_no_shared))
    print('    no    {0}         {1}'.format(count_shared_no_CS,count_no_shared_no_Cs))
    CS_shared_percent=count_CS_shared/(count_CS_shared+count_shared_no_CS)*100
    CS_no_shared_percent=count_CS_no_shared/(count_CS_no_shared+count_no_shared_no_Cs)*100
    print('    %yes  {0:.3f}     {1:.3f}'.format(CS_shared_percent,CS_no_shared_percent))
    from scipy import stats
    observation = np.array([[count_CS_shared, count_CS_no_shared], [count_shared_no_CS, count_no_shared_no_Cs]])
    #print(stats.chi2_contingency(observation))
    print("Fisher oddsratio and p-value")
    print(stats.fisher_exact(observation))
    X_square,p,_,_=stats.chi2_contingency(observation)
    print("X squared value = {0}".format(X_square))
    print("p value = {0}\n".format(p))




def follows_shared_statistics(structured_y):
    '''Start with Arabizi and English only'''
    count_CS_no_shared=0
    count_CS_shared=0
    count_shared_no_CS=0
    count_no_shared_no_Cs=0
    for y in structured_y:
        if('0' in y and '1' in y and '4' in y):
            count_CS_shared+=1
        elif('0' in y and '1' in y):
            count_CS_no_shared+=1
        if (not('0' in y and '1' in 'y') and '4' in y):
            count_shared_no_CS+=1
        if (not('0' in y and '1' in 'y') and not('4' in y)):
            count_no_shared_no_Cs+=1
    print('         Contains Shared word')
    print('          yes         no')
    print('CS  yes   {0}         {1}'.format(count_CS_shared,count_CS_no_shared))
    print('    no    {0}         {1}'.format(count_shared_no_CS,count_no_shared_no_Cs))
    CS_shared_percent=count_CS_shared/(count_CS_shared+count_shared_no_CS)*100
    CS_no_shared_percent=count_CS_no_shared/(count_CS_no_shared+count_no_shared_no_Cs)*100
    print('    %yes  {0:.3f}     {1:.3f}'.format(CS_shared_percent,CS_no_shared_percent))
    from scipy import stats
    observation = np.array([[count_CS_shared, count_CS_no_shared], [count_shared_no_CS, count_no_shared_no_Cs]])
    #print(stats.chi2_contingency(observation))
    print("Fisher oddsratio and p-value")
    print(stats.fisher_exact(observation))
    X_square,p,_,_=stats.chi2_contingency(observation)
    print("X squared value = {0}".format(X_square))
    print("p value = {0}\n".format(p))

#I save only sentences containing arabizi
def annotate_corpus(dir_name,trained_clf,file_source):
     with open('arabizi-'+file_source+'.csv','w',encoding="utf-8",newline="") as ar_f,open("sen_annotated.csv",'r',encoding="utf-8") as sen_annotated:
        sen_annotated_reader=csv.reader(sen_annotated)
        ar_writer= csv.DictWriter(ar_f, fieldnames=('source','user_name','sen_id','sen_num','sen', 'word_level_prediction','sen_level_prediction'))
        ar_writer.writeheader()
        annotated_sentences = [row[2] + ',' + row[3] for row in sen_annotated_reader]
        for file_name in os.listdir(dir_name):
            with open(dir_name+'\\'+file_name, 'r',encoding="utf-8") as read_f:
                reader=csv.reader(read_f)
                for row in reader:
                    if(row[1]+','+row[3] in annotated_sentences): continue
                    line = {'source':file_source,'user_name':row[0],'sen_id':row[1],'sen_num':row[3],'sen':row[4]}
                    sen=row[4].split()
                    prediction=trained_clf.predict([sent2features(sen)])[0]
                    line['word_level_prediction']=prediction
                    if('0' not in prediction): continue
                    binary_tags=[0,0,0,0,0,0]
                    for p in prediction:
                        binary_tags[int(p)]=1
                    line['sen_level_prediction']=''.join(str(b) for b in binary_tags)
                    '''for token,pred in zip(sen,prediction):
                        line['predicted_categ']=pred
                        line['token']=token
                        writer.writerow(line)'''
                    ar_writer.writerow(line)

def automatic_arabizi_corpus_statistics(csv_ar_file):
    #Initially we want to count Mostly Arabizi sentences
    majority_arabizi_count=0
    ar_en_cs=0
    ar_fr_cs=0
    with open(csv_ar_file,'r',encoding="utf-8") as ar_f:
        reader_f=csv.DictReader(ar_f)
        for row in reader_f:
            sen_level_prediction=row['sen_level_prediction']
            word_level_prediction=row['word_level_prediction']
            import ast
            word_level_prediction= ast.literal_eval(word_level_prediction)
            if(len([p for p in word_level_prediction if p=="0" ])>=len(word_level_prediction)/2):
                majority_arabizi_count+=1
            if(sen_level_prediction[0]=='1' and sen_level_prediction[1]=='1'):
                ar_en_cs+=1
            if(sen_level_prediction[0]=='1' and sen_level_prediction[2]=='1'):
                ar_fr_cs+=1
    print('majority arabizi sentences:',majority_arabizi_count)
    print('arabizi english sentences:',ar_en_cs)
    print('arabizi french sentences:',ar_fr_cs)


def error_analysis_ar_cs(csv_ar_file):
     with open(csv_ar_file,'r',encoding="utf-8") as ar_f:
        reader_f=csv.DictReader(ar_f)
        import ast
        at_least_2_en_ar_Words=[row for row in reader_f if len([p for p in ast.literal_eval(row['word_level_prediction']) if p=="0" ])>1 and len([p for p in ast.literal_eval(row['word_level_prediction']) if p=="1" ])>1]
        random_sentences=random.choices(at_least_2_en_ar_Words,k=50)
        for sen in random_sentences:
            print(sen['source'],sen['sen'])

def replace_user_names(source_csv):
    new_csv=source_csv.split('.csv')[0]+'_new.csv'
    data = pd.read_csv(source_csv, dtype=str,na_filter=False)
    user_names=data['user_name'].unique()
    users_dict={k: v for v, k in enumerate(user_names)}
    with open(source_csv,'r',encoding="utf-8") as src_f,open(new_csv,'w',encoding="utf-8",newline="") as write_f:
        reader_f=csv.DictReader(src_f)
        writer_f=csv.DictWriter(write_f,fieldnames=reader_f.fieldnames)
        writer_f.writeheader()
        for row in reader_f:
            user_idx=users_dict[row['user_name']]
            row['user_name']=user_idx
            writer_f.writerow(row)


def select_tweet_ids_into_file(tweet_ids_file):
    data = pd.read_csv('arabizi-twitter.csv', dtype=str,na_filter=False)
    tweet_ids=data['sen_id'].unique()
    with open(tweet_ids_file,'w',encoding="utf-8",newline="") as write_f:
        writer=csv.writer(write_f)
        for id in tweet_ids:
            writer.writerow([id])

if __name__ == '__main__':
    #tweet_ids_file='arabizi_tweet_ids_auto.csv'
    #select_tweet_ids_into_file(tweet_ids_file)
    annotated_file_name= "words_annotated.csv"
    data = pd.read_csv(annotated_file_name, dtype=str,na_filter=False)
    #replace_user_names('arabizi-reddit.csv')
    words_list = data["token"]
    char_vectorizer.fit(words_list)
    print("The following numbers is related to the manually annotated corpus")
    sturctured_data_result = preprocess_structured_data("words_annotated.csv")
    structured_data_list, structured_y = sturctured_data_result[0], sturctured_data_result[1]
    do_main_CS_statistics(structured_y)
    #CS_shared_statistics(structured_y)
    print("The following numbers is related to the reddit automatically annotated corpus")
    auto_annotated_data = pd.read_csv("arabizi-reddit.csv", dtype=str,na_filter=False)
    import ast
    structured_X_aut = auto_annotated_data["sen"]
    structured_y_aut = [ast.literal_eval(s) for s in auto_annotated_data["word_level_prediction"]]
    do_main_CS_statistics(structured_y_aut)
    #CS_shared_statistics(structured_y_aut)
    print("The following numbers is related to the twitter automatically annotated corpus")
    #auto_annotated_data = pd.read_csv("arabizi-twitter.csv", dtype=str,na_filter=False)
    import ast
    structured_X_aut = auto_annotated_data["sen"]
    structured_y_aut = [ast.literal_eval(s) for s in auto_annotated_data["word_level_prediction"]]
    #CS_shared_statistics(structured_y_aut)
    do_main_CS_statistics(structured_y_aut)

    import sklearn_crfsuite
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    #k_fold_cross_validation(structured_data_list,structured_y,crf,k=10)
    X = [sent2features(s) for s in structured_data_list]
    #clf = crf.fit(X, structured_y)
    #print(clf.predict([sent2features('Lava traveled 19km and stopped outside Medina'.split())])[0])
    #annotate_corpus('reddit_tokenized_files',crf,file_source='reddit')
    #automatic_arabizi_corpus_statistics('arabizi-reddit.csv')
    #automatic_arabizi_corpus_statistics('arabizi-twitter.csv')
    #discard_already_annotated(annotated_corpus='words_annotated',reddit_corpus,twitter_corpus)
    #error_analysis_ar_cs('arabizi-twitter.csv')
    #error_analysis_ar_cs('arabizi-reddit.csv')
