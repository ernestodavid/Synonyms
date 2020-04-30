import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.snowball import SnowballStemmer
# Starting with the CountVectorizer/TfidfTransformer approach...
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from itertools import islice

import nltk
nltk.download('punkt')
from bs4 import BeautifulSoup
import requests
import urllib.request
import re

palabrasvac = ['a', 'about', 'above', 'across', 'after', 'afterwards']
palabrasvac += ['again', 'against', 'all', 'almost', 'alone', 'along']
palabrasvac += ['already', 'also', 'although', 'always', 'am', 'among']
palabrasvac += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another']
palabrasvac += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere']
palabrasvac += ['are', 'around', 'as', 'at', 'back', 'be', 'became']
palabrasvac += ['because', 'become', 'becomes', 'becoming', 'been']
palabrasvac += ['before', 'beforehand', 'behind', 'being', 'below']
palabrasvac += ['beside', 'besides', 'between', 'beyond', 'bill', 'both']
palabrasvac += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant']
palabrasvac += ['co', 'con', 'could', 'couldnt', 'cry', 'de',',','.','-']
palabrasvac += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due']
palabrasvac += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else']
palabrasvac += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever']
palabrasvac += ['every', 'everyone', 'everything', 'everywhere', 'except']
palabrasvac += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
palabrasvac += ['five', 'for', 'former', 'formerly', 'forty', 'found']
palabrasvac += ['four', 'from', 'front', 'full', 'further', 'get', 'give']
palabrasvac += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her']
palabrasvac += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers']
palabrasvac += ['herself', 'him', 'himself', 'his', 'how', 'however']
palabrasvac += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
palabrasvac += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
palabrasvac += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
palabrasvac += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
palabrasvac += ['more', 'moreover', 'most', 'mostly', 'move', 'much','/div']
palabrasvac += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
palabrasvac += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
palabrasvac += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
palabrasvac += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
palabrasvac += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
palabrasvac += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
palabrasvac += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
palabrasvac += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
palabrasvac += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
palabrasvac += ['some', 'somehow', 'someone', 'something', 'sometime','way']
palabrasvac += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
palabrasvac += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
palabrasvac += ['then', 'thence', 'there', 'thereafter', 'thereby']
palabrasvac += ['therefore', 'therein', 'thereupon', 'these', 'they']
palabrasvac += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
palabrasvac += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
palabrasvac += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
palabrasvac += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
palabrasvac += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
palabrasvac += ['whatever', 'when', 'whence', 'whenever', 'where']
palabrasvac += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
palabrasvac += ['wherever', 'whether', 'which', 'while', 'whither', 'who']
palabrasvac += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with','[',']','{','}']
palabrasvac += ['within', 'without', 'would', 'yet', 'you', 'your','%','<','>','-','and/or','*','$',':',';',"'s",'``']
palabrasvac += ['yours', 'yourself', 'yourselves','1','2','3','4','5','6','7','8','9','0']

divsAll=[]
topicsSubjects=[]
frecuencySubjects=[]
#def insert_topic_subject(topics,subject,subjects_f,):    
       
     

    #subjects_f.seek(0)
    #subjects_f.write(json.dumps(subject))
    #subjects_f.truncate()
    #del subject["topics"]

    #old_topics = subject["topics"]
    #print(old_topics)
    #for t in old_topics:
        #del old_topics[t]
    
    #page = requests.get(url)
    #print(page)

def termFrecuency(topicsSubjects,frecuencySubjects,topicsAll,frecuencyAll):
    cont=0
    # print(topicsSubjects)
    # print("***************************************************************")
    # print(topicsAll)
    for topicList in topicsSubjects:
        
        #result=[w for w in topicList if w in topics_resultALL]
        for topic in topicList:   
            if topic in topicsAll:
                #Indice dentro de el array de la asignatura
                indexTopic=topicList.index(topic)
                #Valor en frecuencia del Topic dentro del docuemnto de la asignatura
                valueFrecuencyTopic=frecuencySubjects[cont][indexTopic]

                #Tomo el indice de las palabras de cada asignatura en el array de las palabras de todos los docuemtos
                indexTopicGeneral=topicsAll.index(topic)
                #print(indexFrecuencyTopicGeneral)
                
                
                #Tomo el valor de la frecuencia de esa palabra en todos los documentos
                valueFrecuencyTopicGeneral=frecuencyAll[indexTopicGeneral]
                

                # print("En subjects")
                # print(valueFrecuencyTopic)
                # print("En grneral")
                # print(valueFrecuencyTopicGeneral)

                listaSubject=[topic,valueFrecuencyTopic]
                listaTotal=[topic,valueFrecuencyTopicGeneral]

                #
                print(listaSubject)
                print(listaTotal)
                
            else:
                continue
                #print(topic)
        
        list_word_frecuancySUBJECTS=list(zip(topicsSubjects[cont], frecuencySubjects[cont]))
        list_word_frecuancyAllDoc=list(zip(topicsAll, frecuencyAll))
        #print("Esto esta relacionado a la Asignatura "+str(cont))
        cont=cont+1
        # print("************************************   topics_subject  *******************************")
        # print(list_word_frecuancySUBJECTS)
        # print("************************************   topics_total  *******************************")
        # print(list_word_frecuancyAllDoc)

def stop_wear(vocab,palabrasvac):
   
    return [s for s in vocab if s not in palabrasvac]
def wordCount(tokens):
    words = [n.lower() for n in tokens]
    frecuenciaPalab=[]
    for n in words:
        frecuenciaPalab.append(words.count(n))
    return words, frecuenciaPalab

def vocabulario(tokens):
    
    #print(type(words))
    words,frecuenciaPalab=wordCount(tokens)     
    
    vocab = list(sorted(set(words)))
    vocab_need=stop_wear(vocab,palabrasvac)
    for x in vocab_need:
        if ((x is int) or (len(x)<4 )):            
            #print("**************   "+x+"  *************************")            
            vocab_need.remove(x)        
        else:
            continue

    
    list_word_frecuancy=list(zip(vocab_need, frecuenciaPalab))
    orden_lista=sorted(list_word_frecuancy,key=lambda row: row[1], reverse=True)
    

    #print("Pares\n" +str(orden_lista))
    #print(type(vocab))
    #print(vocab)
    return orden_lista
     
        
        
def miner(subjects,subjects_f):
    # Function to extract the description of every subject
    i=0
    for subject in subjects:
        i=i+1
        if i>2:
            break
        url = subject["url"]
        all_text=[]
        #page = urllib.request.urlopen(url)
        page=requests.get(url)
        
        status_code=page.status_code
        if status_code == 200:
            #print(url)
            # print the source code
            #print(page.content)
            u=0
            parsed_webpage = BeautifulSoup(page.content,"html.parser")
            #parsed_webpage=parsed_webpage.get_text()

            p_tag = parsed_webpage.find_all('div',attrs={'class': 'tarea'})
            if p_tag==[]:
                continue
            else:
                #p_tag = parsed_webpage.findAll('div')
                
                for div in p_tag:
                    u=u+1
                    #print("El div que se ve es este"+str(u))
                    all_text.extend(div)
                    #print(str(div))
                token=nltk.word_tokenize(str(all_text))
                
                tokensSubjects = [x.replace('\n','') for x in token]
                tokensSubjects=[x.replace('\\n','') for x in tokensSubjects]
                tokensSubjects=[x.replace('.-','') for x in tokensSubjects]
                tokensSubjects=[x.replace('-','') for x in tokensSubjects]
                tokensSubjects=[x.replace('.t','') for x in tokensSubjects]
                tokensSubjects=[x.replace('\\','') for x in tokensSubjects]

                #Add todos las palabras de los documentos a un array final
                divsAll.extend(all_text)
                # print("************************************   token acumulado  *******************************")
                # print(divsAll)
                #Hago un conteo de la frecuencia en que aparece cada palabra
                unique_words=vocabulario(tokensSubjects)
                #print(str(unique_words))

                topics_result, frecuency=zip(*unique_words)
                # print("************************************   topics_subjects *******************************")
                # print(topics_result)
                

                topicsSubjects.append(topics_result)

                frecuencySubjects.append(frecuency)
                #print(str(topics_result))
                
    return divsAll,topicsSubjects,frecuencySubjects

def writeTopics(subject,subjects_f,topics_result):    
    subject["topics"] = topics_result
    with open('data/subjects.json', 'w') as subjects_f:
        subjects_f.write(json.dumps(subjects))    

if __name__ == "__main__":
    
    # Open subjects file and get subject object
    with open("data/subjects.json") as subjects_f:
        subjects =json.load(subjects_f)
    
    divsAll,topicsSubjects,frecuencySubjects=miner(subjects,subjects_f)
    # print("************************************   tokensAll  *******************************")
    # print(str(divsAll))
    tokenized_before=nltk.word_tokenize(str(divsAll))
    tokenized_before=[x.replace('\\n','') for x in tokenized_before]
    tokenized_before=[x.replace('\n','') for x in tokenized_before]
    tokenized_before=[x.replace('.t','') for x in tokenized_before]
    tokenized_before=[x.replace('.-','') for x in tokenized_before]
    tokenized_before=[x.replace('-','') for x in tokenized_before]
    tokensAll=[x.replace('\\','') for x in tokenized_before]
    
    #word,frecuencyAll=wordCount(tokensAll)    
    listaOrdenada=vocabulario(tokensAll)
    print(tokensAll)

    topicsAll,frecuencyAll=zip(*listaOrdenada)
    
    #print("**************    topics_resultALL   *******************")
    #print(str(topicsAll))
    #print("**************    frecuencyALL   *******************")
    #print(str(frecuencyAll))
    #print("**************    topicsSubjects   *******************")
    #print(str(topicsSubjects))
    #print("**************    frecuencySubjects   *******************")
    #print(str(frecuencySubjects))
    #print("**************    frecuencySubjects  [0] *******************")
    #print(str(frecuencySubjects[0]))

    termFrecuency(topicsSubjects,frecuencySubjects,topicsAll,frecuencyAll)

    #writeTopics(subjects,subjects_f)

    #clean_topics = cleaner(topics)
    #top_words = extract_kwargs(clean_topics)

    #top_words.to_json("topics.json", orient='records')

    