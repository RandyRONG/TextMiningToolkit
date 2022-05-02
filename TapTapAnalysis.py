# -*- coding: utf-8 -*-
import re # 正则表达式库
import jieba # 结巴分词
from jieba import analyse
import numpy as np
import pandas as pd
import json
import openpyxl
import fpGrowth
import collections
import os
import math
import operator
from collections import defaultdict,Counter
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image 
from snownlp import SnowNLP
from snownlp import sentiment
import random
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
# from pylab import mpl
import nltk
from yellowbrick.text import DispersionPlot
import datetime
import time
import networkx as nx
from netwulf import visualize
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull

import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing


import seaborn as sns


def add_sheet(data, path_, sheet_name,idx_bool):
    excel_writer = pd.ExcelWriter(path_, mode="a",engine='openpyxl',if_sheet_exists='replace')
    """
    不改变原有Excel的数据，新增sheet。
    注：
        使用openpyxl操作Excel时Excel必需存在，因此要新建空sheet
        无论如何sheet页都会被新建，只是当sheet_name已经存在时会新建一个以1结尾的sheet，如：test已经存在时，新建sheet为test1，以此类推
    :param data: DataFrame数据
    :param excel_writer: 文件路径
    :param sheet_name: 新增的sheet名称
    :return:
    """
    book = openpyxl.load_workbook(path_)
    excel_writer.book = book
    if idx_bool == 1:
        data.to_excel(excel_writer=excel_writer, sheet_name=sheet_name)
    else:
        data.to_excel(excel_writer=excel_writer, sheet_name=sheet_name, index=None)
 
    excel_writer.close()

def clean_text(path):
    f = open (path,encoding='utf-8') 
    text = f.read()
    f.close()
    # print (text)
    if text.startswith(u'\ufeff'):
        text = text.encode('utf8')[3:].decode('utf8')

    json_dict = json.loads(text,encoding='utf8')
        
    return json_dict

def get_fre_df(freqItems):
    fre_items = []
    for i in freqItems:
        if len(i) == 1:
            continue
        fre_item = []
        for j in i:
            fre_item.append(j)
        fre_items.append(fre_item)
    df_fre = pd.DataFrame(fre_items)
    return df_fre

def get_frequent_items(dataSet,prelix,game_):
    freqItems=fpGrowth.fpGrowth(dataSet,max(int(0.1*len(dataSet)),10))  #n代表最小频繁度，即要找出出现n次或n次以上的频繁项集
    df_fre = get_fre_df(freqItems)
    if df_fre.shape[0]<5:
        freqItems=fpGrowth.fpGrowth(dataSet,max(int(0.05*len(dataSet)),10))
        df_fre = get_fre_df(freqItems)
    # print (df_fre)
    if prelix == '总计-':
        print ('the final frequent items result for {} shows below:'.format(game_))
        print (df_fre)

    add_sheet(df_fre, game_path, prelix+'分词频繁项',0)



def get_total_frequent(dict_):
    df_total_fre = pd.DataFrame.from_dict(dict_, orient='index').reset_index()
    df_total_fre = df_total_fre.rename(columns={'index':'word', 0:'count'})
    df_total_fre = df_total_fre.sort_values(by='count',ascending=False).reset_index(drop=True)
    df_total_fre = df_total_fre[df_total_fre['count']>= df_total_fre.iloc[min(words_range-1,int(0.9*df_total_fre.shape[0])),1]]
    return df_total_fre


def feature_select(list_words,prelix):
    #总词频统计
    doc_frequency=defaultdict(int)
    list_words_extend = []
    for i in list_words:
        list_words_extend.extend(i)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i]+=1
    df_total_fre = get_total_frequent(doc_frequency)
    
    #计算每个词的TF值
    word_tf={}  #存储没个词的tf值
    for i in doc_frequency:
        word_tf[i]=doc_frequency[i]/sum(doc_frequency.values())
 
    #计算每个词的IDF值
    doc_num=len(list_words)
    word_idf={} #存储每个词的idf值
    word_doc=defaultdict(int) #存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i]+=1
    for i in doc_frequency:
        word_idf[i]=math.log(doc_num/(word_doc[i]+1))
 
    #计算每个词的TF*IDF的值

    word_df={}
    for i in doc_frequency:
        word_df[i]= (word_doc[i])/doc_num
    
    word_tf_idf={}
    for i in doc_frequency:
        word_tf_idf[i]= round(word_tf[i]*word_idf[i],4)
    
 
    # 对字典按值由大到小排序
    dict_feature_select=sorted(word_df.items(),key=operator.itemgetter(1),reverse=True)[:words_range]
    df_df = pd.DataFrame(dict_feature_select, columns=['word', 'DF']) 
    dict_feature_select=sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)[:words_range]
    df_tfidf = pd.DataFrame(dict_feature_select, columns=['word', 'TF-IDF']) 
    
    ### making piocture
    # mask = np.array(Image.open(picture_path))
    # wc = WordCloud(
    #     background_color='white',
    #     ### the font path should be adjusted for different operating system
    #     font_path="C:/Windows/Fonts/msyh.ttc",
    #     mask=mask, 
    #     max_words=words_range, 
    #     ### you could customize your output picture here
    #     max_font_size=64, 
    #     scale=32,  
    #     width = 400,
    #     height = 200,
    #     color_func = ImageColorGenerator(mask)
    # )
    # word_counts = collections.Counter(list_words_extend) 
    # wc.generate_from_frequencies(word_counts)
    # wc.to_file(prelix+output_pic_path) 
    
    # plt.imshow(wc) 
    # plt.axis('off') 
    # plt.show() 

    # print (df_tfidf)
    add_sheet(df_total_fre, game_path,prelix+'过滤版词频统计',0)
    add_sheet(df_df, game_path,prelix+'过滤版DF统计',0)
    add_sheet(df_tfidf, game_path, prelix+'TF-IDF版词频统计',0)

def showtable(data, plt):
    x = data.T[1]
    y = data.T[3]
    plt.scatter(x, y)

def euler_distance(point1, point2) :
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)

def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def kpp_centers(data_set, k):
    """
    从数据集中返回 k 个对象可作为质心
    """
    cluster_centers = []
    cluster_centers.append(list(random.choice(data_set)))
    d = [0 for _ in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers) # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d): # 轮盘法选出下一个聚类中心；
            total -= di
            if total > 0:
                continue
            cluster_centers.append(list(random.choice(data_set)))
            break
    return np.array(cluster_centers)


def kmeans(data, n, m, k, plt):
    # 获取k个随机数
    # rarray = np.random.random(size=k)
    # 乘以数据集大小——>数据集中随机的4个点
    # rarray = np.floor(rarray*n)
    # # 转为int
    # rarray = rarray.astype(int)
    # print('数据集中随机索引', rarray)
    # 随机取数据集中的4个点作为初始中心点
    center = kpp_centers(data, k)
    # 测试比较偏、比较集中的点，效果依然完美，测试需要删除以上代码
    # center = np.array([[4.6,-2.5],[4.4,-1.7],[4.3,-0.7],[4.8,-1.1]])
    # 1行80列的0数组，标记每个样本所属的类(k[i])
    cls = np.zeros([n], np.int)
    print('初始center=\n', center)
    run = True
    time = 0
    while run:
        time = time + 1
        for i in range(n):
            # 求差
            tmp = data[i] - center
            # 求平方
            tmp = np.square(tmp)
            # axis=1表示按行求和
            tmp = np.sum(tmp, axis=1)
            # 取最小（最近）的给该点“染色”（标记每个样本所属的类(k[i])）
            cls[i] = np.argmin(tmp)
        # 如果没有修改各分类中心点，就结束循环
        run = False
        # 计算更新每个类的中心点
        for i in range(k):
            # 找到属于该类的所有样本
            club = data[cls==i]
            # axis=0表示按列求平均值，计算出新的中心点
            newcenter = np.mean(club, axis=0)
            # 如果新旧center的差距很小，看做他们相等，否则更新之。run置true，再来一次循环
            ss = np.abs(center[i]-newcenter)
            if np.sum(ss, axis=0) > 1e-4:
                center[i] = newcenter
                run = True
        # print('new center=\n', center)
    # print('程序结束，迭代次数：', time)
    # 按类打印图表，因为每打印一次，颜色都不一样，所以可区分出来
    
    # for i in range(k):
    #     club = data[cls == i]
    #     showtable(club, plt)
    # # 打印最后的中心点
    # showtable(center, plt)
    # plt.show()
    return cls,center

def feature_select_cluster_pre(list_words,list_rates,game_):
    
    doc_frequency=defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i]+=1
    words_dict = {}
    for i in doc_frequency:
        words_dict[i] = {}
        words_dict[i]['word_tf']=doc_frequency[i]/sum(doc_frequency.values())
    doc_num=len(list_words)
    word_doc=defaultdict(int) #存储包含该词的文档数
    for i in words_dict:
        words_dict[i]['rate_list'] = []
        for j in range(len(list_words)):
            if i in list_words[j]:
                word_doc[i]+=1
                words_dict[i]['rate_list'].append(list_rates[j])
    del_list = []
    for i in words_dict:
        if word_doc[i] > 2:
            words_dict[i]['word_idf']=math.log(doc_num/(word_doc[i]+1))   
            
            words_dict[i]['word_tfidf']= round(words_dict[i]['word_tf']*words_dict[i]['word_idf'],4)
            # for rank_ in range(1,6):
            #     words_dict[i]['percent_of_rate_'+str(rank_)] = round(words_dict[i]['rate_list'].count(rank_) / len(words_dict[i]['rate_list']),4)
            words_dict[i]['percent_of_rate_neg'] = round((words_dict[i]['rate_list'].count(1)+words_dict[i]['rate_list'].count(2)) / len(words_dict[i]['rate_list']),4)
            words_dict[i]['percent_of_rate_neu'] = round((words_dict[i]['rate_list'].count(3)) / len(words_dict[i]['rate_list']),4)
            words_dict[i]['percent_of_rate_pos'] = round((words_dict[i]['rate_list'].count(4)+words_dict[i]['rate_list'].count(5)) / len(words_dict[i]['rate_list']),4)
            del words_dict[i]['rate_list']
            del words_dict[i]['word_tf']
            del words_dict[i]['word_idf']
        else:
            del_list.append(i)
    for del_item in del_list:
        del words_dict[del_item]
    
    words_cluter_pre = pd.DataFrame.from_dict(words_dict, orient='index')
    min_max_scaler = preprocessing.MinMaxScaler() 
    # print (np.array(words_cluter_pre['word_tfidf']).reshape(1, -1))
    words_cluter_pre['normalized_word_tfidf'] = min_max_scaler.fit_transform(np.array(words_cluter_pre['word_tfidf']).reshape(-1, 1))
    words_cluter_pre = words_cluter_pre[words_cluter_pre['normalized_word_tfidf']>=0.1] 
    words_cluter_pre = words_cluter_pre.drop(['word_tfidf'],axis = 1)
    words_cluter_pre = words_cluter_pre[['normalized_word_tfidf','percent_of_rate_neg','percent_of_rate_neu','percent_of_rate_pos']]
    # words_cluter_pre = words_cluter_pre.rename(columns={'index':'word_tf', 0:'word_idf',1:'word_tfidf'})
    # for rank_ in range(1,6):
    #     words_cluter_pre = words_cluter_pre.rename(columns={rank_+1:'percent_of_rate_'+str(rank_)})
    words_cluter_pre = words_cluter_pre.sort_values(by='normalized_word_tfidf',ascending=False).reset_index(drop = False)
    # words_cluter_pre = words_cluter_pre[words_cluter_pre['word_tfidf']>= words_cluter_pre.iloc[int(0.9*words_cluter_pre.shape[0]),1]]
    words_cluter_pre = words_cluter_pre.rename(columns={'index':'word',0:'normalized_word_tfidf',1:'percent_of_rate_neg',2:'percent_of_rate_neu',3:'percent_of_rate_pos'})
    
    bar_ = 1000
    if words_cluter_pre.shape[0] > bar_:
        words_cluter_pre = words_cluter_pre[words_cluter_pre['normalized_word_tfidf']>= words_cluter_pre.iloc[bar_-1,1]]
    words_cluter_pre = words_cluter_pre.set_index('word',drop=True)
    # print (words_cluter_pre.columns)
    print (words_cluter_pre)
    add_sheet(words_cluter_pre, game_path, '分词评价占比表',1)

    k_clubs = 4
    cluster_clubs,cluster_center = kmeans(words_cluter_pre.values, words_cluter_pre.shape[0], words_cluter_pre.shape[1], k_clubs, plt) 
    words_cluter_pre['cluster_clubs'] = cluster_clubs

    imp_rate_list = []
    neg_rate_list = []
    neu_rate_list = []
    pos_rate_list = []
    for i in range(0,k_clubs):
        imp_rate_list.append(cluster_center[i][0])
        neg_rate_list.append(cluster_center[i][1])
        neu_rate_list.append(cluster_center[i][2])
        pos_rate_list.append(cluster_center[i][3])

    imp_idx = imp_rate_list.index(max(imp_rate_list))
    # neu_idx = neu_rate_list.index(max(neu_rate_list))
    pos_idx= pos_rate_list.index(max(pos_rate_list))
    neg_idx = neg_rate_list.index(max(neg_rate_list))
    cluster_clubs_dict = {pos_idx:"最正面",neg_idx:"最负面"}
    for i in range(0,k_clubs):
        if i in [pos_idx,neg_idx]:
            continue
        if cluster_center[i][2]+0.1 > cluster_center[i][1] and cluster_center[i][2]+0.1 > cluster_center[i][3]:
            cluster_clubs_dict[i] = '最中立'
        elif cluster_center[i][3] > cluster_center[i][1]:
            cluster_clubs_dict[i] = '次好评'
        elif cluster_center[i][3] < cluster_center[i][1]:
            cluster_clubs_dict[i] = '次差评'
    for i in range(0,k_clubs):
        if i in [imp_idx]:
            cluster_clubs_dict[i] += '_最重要'
 
    cluster_clubs_2 =[]
    for i in range(len(cluster_clubs)):
        if cluster_clubs[i] in cluster_clubs_dict.keys():
            cluster_clubs_2.append(cluster_clubs_dict[cluster_clubs[i]])

    words_cluter_pre['cluster_clubs_rep'] = cluster_clubs_2


    df_cluster_center = pd.DataFrame(cluster_center, columns=['TF-IDF', '差评占比', '中评占比','好评占比']) 
    df_cluster_center = df_cluster_center.rename(index = cluster_clubs_dict)
    print (df_cluster_center)
    add_sheet(df_cluster_center, game_path, '分词聚类中心各均值指标',1)
    cluter_words_list = []
    for i in range(0,k_clubs):
        sub_words_list = words_cluter_pre[words_cluter_pre['cluster_clubs'] == i].index.values
        cluter_words_list.append(sub_words_list)
    df_cluter_words = pd.DataFrame(cluter_words_list).rename(index = cluster_clubs_dict)
    df_cluter_words = pd.DataFrame(df_cluter_words.values.T, index=df_cluter_words.columns, columns=df_cluter_words.index)
    print (df_cluter_words)
    add_sheet(df_cluter_words, game_path, '分词聚类结果',0)


def train_model():

    column_names = ["label","review"]
    data = pd.DataFrame(columns = column_names)

    for file_idx, file_name in enumerate(file_list):
        game_ = file_name.split('_')[1]
        platform = file_name.split('_')[0]
        if platform not in ['Tap','App']: 
            continue
        game_path = os.path.join(root_path,file_name)
        dfc = pd.read_excel(game_path, sheet_name = file_name.split('.')[0],engine='openpyxl')
        text_file = field_dict[platform][0]
        score_field = field_dict[platform][1]
        dfc = dfc[dfc[score_field]!=3]
        dfc['label'] = dfc.apply(lambda x: 1 if x[score_field] >=4 else 0, axis=1)
        dfc.rename(columns={text_file:'review'}, inplace = True)
        data = data.append(dfc[['label','review']])
    ### data = data.sample(frac=0.01).reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(np.array(data['label']))
    ### 得出指标后以下内容注释
    # skf = StratifiedKFold(n_splits=k_num,shuffle=True)
    # record_roc = []
    # for train_index, val_index in skf.split(data,y):
    #     train, test = data.iloc[train_index], data.iloc[val_index]
    #     label_train,label_test = y[train_index], y[val_index]
    #     train_neg=train.iloc[:,1][label_train==0]
    #     train_pos=train.iloc[:,1][label_train==1]
    #     train_neg.to_csv(root_path+"neg.csv",index=0,header=0)
    #     train_pos.to_csv(root_path+"pos.csv",index=0,header=0)
    #     test.to_csv(root_path+"test.csv",index=0,columns=['label','review'])
    #     sentiment.train(root_path+'neg.csv',root_path+'pos.csv')
    #     sentiment.save('sentiment.marshal')
    #     print ('一轮训练完毕...')

    #     test=pd.read_csv(root_path+"test.csv")
    #     review_list=[review for review in test['review']]
    #     label_list=[label for label in test['label']]
    #     list_test=[(label,review) for label,review in list(zip(label_list,review_list)) if type(review)!=float]

    #     # for j in list_test:
    #     #     print(j[1],j[0],SnowNLP(j[1]).sentiments)


    #     senti=[SnowNLP(review).sentiments for label,review in list_test]
    #     roc_ = roc_auc_score(y_true=label_list,y_score=senti)
    #     record_roc.append(roc_)
    # print ('AUC平均数为：',np.mean(record_roc),'  ',record_roc)
    # exit()
    ### 得出指标后以上内容注释
    train_neg=data.iloc[:,1][y==0]
    train_pos=data.iloc[:,1][y==1]
    train_neg.to_csv(root_path+"neg.csv",index=0,header=0)
    train_pos.to_csv(root_path+"pos.csv",index=0,header=0)
    sentiment.train(root_path+'neg.csv',root_path+'pos.csv')
    sentiment.save('sentiment.marshal')
    print ('最终训练完毕...')
    


if __name__ == "__main__":
    
    picture_path = 'module.png'
    output_pic_path = 'output.png'
    output_vecmodel_path = './Word2Vec/Word2VecModel'
    inp = './Word2Vec/train_seg.txt'
    with open(".\stop_words_ch.txt", errors='ignore') as f:    #打开文件
        data = f.read()   #读取文件
        remove_words = data.split()
        # print(data)
    
    with open(".\cn_stopwords.txt", encoding='utf-8',errors='ignore') as f:    #打开文件
        data = f.read()   #读取文件
        remove_words_2 = [str(i) for i in data.split() if i not in remove_words]
    remove_words.extend(remove_words_2)
   
    # game_list = ['半世界之旅','光与夜之恋','梦间集天鹅座','少女的王座','时空中的绘旅人','未定事件簿','遇见逆水寒','恋与制作人']
    # words_range = int(input('请输入要统计英日词频的词数（词频排名前多少个，推荐前100个）：'))
    words_range = 200
    k_num = 3
    ## True  False
    Dispersion = False
    TimeAnalytics = True
    
    
    word_N = 10
    WordVec = False
    n_components = 8
    vecsize = 200
    wordvec_show_len = 100
    KnowledgeMapping = False
    filter_sign = False
    filter_ratio = 0.002
    # 'App':['Content','Rating']
    field_dict = {'App':['contents','score'],'Tap':['contents','score'],'Bili':['comment'],'BiliBulletchat':['bulletchat_text'],
        "Steam":['review','score']}
    score_aggre_dict = {'1':'差评','2':'差评','3':'中评','4':'好评','5':'好评',}
    
    # temp_remove_words = ['匹配','队友','机制','希望','腾讯','人机','BUG','恶心','举报','大师','对面','体验','国服'] '时代','帝国时代','帝国'
    temp_remove_words = []
    # temp_special_words = ['鲁班','李元芳','云缨'] '钟离','甘雨','魈','优菈'
    temp_special_words = []
    # selected_words = ['少前','少女','前线','抽奖','皮肤','二次','联动','老婆','lp']
    remove_words.extend(temp_remove_words)

    temp_remove_words = ['的','了','没有','应该','有点','不了','有没有','不能','不会','游戏']
    remove_words.extend(temp_remove_words)

    remove_words_str = '\t|\n|'
    for remove_word in remove_words:
        remove_words_str += '\\' + str(remove_word) + '|'

    # print (remove_words_str)
    
    game_words_dict_path = r'.\dict_game_words_ch.json'


    root_path = './Target4/'
    names = os.listdir(root_path)    #这将返回一个所有文件名的列表
    file_list = []
    for name in names:
        if '鸿图之下' not in name:
            continue
        if '.xlsx' not in name:
            continue
        file_list.append(name)

    game_words_dict = clean_text(game_words_dict_path)

    

    words_special = game_words_dict['ch']
    words_special.extend(temp_special_words)
    words_special_single = game_words_dict['ch_single']
    replace_words = game_words_dict['replace_words']
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    
    # train_model()
    # exit()
    ### 得出最终模型后以上内容注释
    for file_idx, file_name in enumerate(file_list):
        the_date = file_name.split('_')[-1]
        game_ = file_name.split('_')[-2]
        platform = file_name.split('_')[0]
        print ('*'*10)
        print("{} starts counting...".format(file_name))
        
        
        game_path = os.path.join(root_path,file_name)
        D = []
        D_weights = []
        D_rate = []
        D_total = []
        D_date = []
        # sheet_name = file_name.split('.')[0],
        dfc = pd.read_excel(game_path, engine='openpyxl')
        time_window = 'month'
        filter_cont_len = max(100,0.01*dfc.shape[0])
        # dfc = dfc.sample(1000).reset_index(drop=True)
        if platform in field_dict.keys():
            text_file = field_dict[platform][0]
        else:
            text_file = 'text'
        if platform in ['Tap','App']:
            score_field = field_dict[platform][1]

        for i in range(dfc.shape[0]):

            # if i % 1000 == 0:
            #     break

            if i % 1000 == 0:
                print('*'* 10)
                print('{} has been counting at {}%'.format(file_name,int(round(i/dfc.shape[0],2)*100)))
            string_data_1 = dfc.loc[i,text_file]
            # print (string_data_1)
            if type(string_data_1) != str:
                continue 
            if string_data_1 == "":
                continue
            if '（该条评论已经被删除）' in string_data_1:
                continue
            # pass_sign = 0
            # for selected_word in selected_words:
            #     if selected_word in string_data_1:
            #         pass_sign = 1
            # if pass_sign == 0:
            #     continue
            if platform in ['Tap','App']:
                rate_ = max(int(dfc.loc[i,score_field]),1)
            else:
                rate_ = min(max(round((SnowNLP(string_data_1).sentiments)*4)+1,1),5)
            
            if platform in ['Tap']:
                spent = dfc.loc[i,'spent']
                ups = dfc.loc[i,'ups']
                downs = dfc.loc[i,'downs']
                comments = dfc.loc[i,'comments']
                updated_time = dfc.loc[i,'updated_time']
                if type(updated_time) == str:
                    # print (updated_time)
                    updated_time = time.strptime(updated_time, "%Y-%m-%d %H:%M:%S")
                    updated_time = int(time.mktime(updated_time))
                    # print (updated_time)
                weight = 1
                for indicator in [spent,ups,downs,comments]:
                    if math.isnan(indicator):
                        weight += 0
                        continue
                    weight += indicator
                weight = int(math.log(weight))+1
                # print (weight)
            elif platform in ['Steam']:
                votes_up = dfc.loc[i,'votes_up']
                votes_funny = dfc.loc[i,'votes_funny']
                comment_count = dfc.loc[i,'comment_count']
                weight = 1
                for indicator in [votes_up,votes_funny,comment_count]:
                    if indicator == np.nan:
                        indicator = 0
                    if indicator in [comment_count]:
                        weight += 2*indicator
                    else:
                        weight += indicator
                weight = int(math.log(weight))+1
            else:
                weight = 1
            D_1 = []
            # for remove_word in remove_words:
            #     # print (remove_word)
            # pattern = re.compile(remove_words_str) # 定义正则表达式匹配模式
            # string_data_1 = re.sub(pattern, '', string_data_1) # 将符合模式的字符去除
            # print (string_data_1)
            for replace_word in replace_words.keys():
                if replace_word not in string_data_1:
                    continue
                print ('{} is replaced by {}'.format(replace_word,replace_words[replace_word]))
                string_data_1 = string_data_1.replace(replace_word,replace_words[replace_word])
            for word in words_special:
                if word not in string_data_1:
                    continue
                pattern_special = re.compile(word)
                result_list = pattern_special.findall(string_data_1)
                D_1.extend(result_list)
                for word_special_single in words_special_single:
                    if word_special_single not in result_list[0]:
                        continue
                    D_1.extend([word_special_single] * len(result_list))
                string_data_1 = re.sub(pattern_special, ' ', string_data_1)
                # print (string_data_1)
                
            keywords = analyse.extract_tags(string_data_1, topK=200, withWeight=False, allowPOS=('ns', 'n','nr','nt','nz', 'vn', 'v','nv'))
            seg_list_exact_1 = jieba.cut(string_data_1, cut_all = False)
            for word in seg_list_exact_1: # 循环读出每个分词
                if word in remove_words: # 如果不在去除词库中
                    continue
                if (word in words_special_single) or (word in keywords):
                    D_1.append(word) # 分词追加到列表
            new_en_word_list = pat_letter.sub(' ', string_data_1).strip().lower().split()
            for word_en in new_en_word_list:
                if word_en not in game_words_dict['en'].keys():
                    continue
                en_word = game_words_dict['en'][word_en]
                D_1.append(en_word) 
                
            D.append(D_1)
            D_total.extend(D_1)
            D_weights.append(weight)
            D_rate.append(rate_)
            if platform in ['Tap']:
                D_date.append(updated_time)
                
            
        
        if Dispersion == True:
            D_sortdate = [x for _,x in sorted(zip(D_date,D))]
            D_rate_sortdate = [x for _,x in sorted(zip(D_date,D_rate))]
            D_rate_sortdate = [score_aggre_dict[str(i)] for i in D_rate_sortdate]
            D_date_sort = [x for _,x in sorted(zip(D_date,D_date))]
            try:
                # time_local = time.localtime(D_date_sort[0])
                start_dt = D_date_sort[0]
                # time_local = time.localtime(D_date_sort[-1])
                end_dt = D_date_sort[-1]
            except:
                start_dt = time.strftime("%Y-%m-%d %H:%M:%S",D_date_sort[0])
                end_dt = time.strftime("%Y-%m-%d %H:%M:%S",D_date_sort[-1])
            # print (D_sortdate[:10])
            # print (D_date_sort[:10])
            plt.rcParams['font.sans-serif']=['SimHei']
            # ntext = nltk.Text(D_sortdate)
            # ntext.dispersion_plot(['腾讯'])
            text = D_sortdate
            y = D_rate_sortdate
            target_words = ['腾讯']
            # Lexical Dispersion Plot
            visualizer = DispersionPlot(
                target_words,
                colormap="Accent",
                title='{}:{}-{}'.format(game_,start_dt,end_dt)
            )
            visualizer.fit(text, y)
            visualizer.show()
            exit()

        
        if TimeAnalytics == True:
            D_sortdate = [x for _,x in sorted(zip(D_date,D))]
            D_rate_sortdate = [x for _,x in sorted(zip(D_date,D_rate))]
            D_rate_sortdate = [score_aggre_dict[str(i)] for i in D_rate_sortdate]
            try:
                D_date_sort = [time.localtime(x) for _,x in sorted(zip(D_date,D_date))]
            except:
                D_date_sort = [time.localtime(x.value//10**9) for _,x in sorted(zip(D_date,D_date))]
            if time_window == 'day':
                D_date_sort = [time.strftime("%Y-%m-%d",x) for x in D_date_sort]
            elif time_window == 'week':
                # D_date_sort_m = [time.strftime("%Y-%m",x) for x in D_date_sort]
                D_date_sort = [time.strftime("%Y-%m-%W",x) for x in D_date_sort]
                D_date_sort = [i[:8]+'W'+i[8:] for i in D_date_sort]
                
            elif time_window == 'month':
                D_date_sort = [time.strftime("%Y-%m",x) for x in D_date_sort]
            elif time_window == 'quarter':
                D_date_sort = [time.strftime("%Y-%m",x) for x in D_date_sort]
                D_date_sort = [i[:5]+'Q'+str(((int(i.split('-')[1])-1)//3)+1) for i in D_date_sort]
            elif time_window == 'year':
                D_date_sort = [time.strftime("%Y",x) for x in D_date_sort]
            time_periods = sorted(list(set(D_date_sort)))
            print (time_periods)
            score_lists = []
            words_lists = []
            final_time_periods = []
            record_len = []
            record_scores = []
            for time_period in time_periods:
                D_temp = []
                D_temp_s = []
                D_rate_temp = []
                record_score = []
                temp_count = 0
                score_dict = {}
                for idx, date_ in enumerate(D_date_sort):
                    if date_ == time_period:
                        sub_score = D_rate_sortdate[idx]
                        temp_count += 1
                        record_score.append(sub_score)
                        D_temp.extend(D_sortdate[idx])
                        temp_record = []
                        for sub_item in D_sortdate[idx]:
                            if sub_item in temp_record:
                                continue
                            if sub_item not in score_dict.keys():
                                score_dict[sub_item] = []
                            score_dict[sub_item].append(sub_score)
                            temp_record.append(sub_item)
                if temp_count < filter_cont_len:
                    continue
                record_scores.append(str(int(100*round(record_score.count('好评')/len(record_score),2))))
                record_len.append(str(temp_count))
                final_time_periods.append(time_period)
                most_list = Counter(D_temp).most_common(word_N)
                most_list = [list(i)+[round(score_dict[i[0]].count('好评')/len(score_dict[i[0]]),4)] for i in most_list]
                s_list = [i[2] for i in most_list]
                w_list = [i[0] for i in most_list]
                if len(w_list) < word_N:
                    s_list.extend([0.5]*(word_N-len(s_list)))
                    w_list.extend(['']*(word_N-len(w_list)))
                score_lists.append(s_list)
                words_lists.append(w_list)
                print (time_period)
                print (most_list)
            x_label = final_time_periods
            plt.rcParams['font.family'] = 'Kaiti'     # 中文显示

            fig, ax = plt.subplots(figsize=(25, 9))   # 绘图
            # datas = np.array(score_lists).T
            x_label = [x_label[i] +'\n'+ record_len[i]+'条'+'\n'+record_scores[i] + '%' for i in range(len(x_label))]
            y_label= ['rank'+str(i) for i in range(1,word_N+1)]
            datas=pd.DataFrame(np.array(score_lists).T,index=y_label,
                            columns=x_label)

            im=ax.imshow(datas,cmap='GnBu',aspect='auto')
            cbar=ax.figure.colorbar(im, ax=ax)
            #colorbar的设置
            cbar.ax.set_ylabel('好评度', rotation=-90, va="bottom",fontsize=18,)#colorb
            # cmap='GnBu',
            heatmap = plt.pcolor(datas,cmap='GnBu',linewidths = 0.05, )
            for y in range(datas.shape[0]):
                for x in range(datas.shape[1]):
                    # color="w",
                    plt.text(x + 0.5, y + 0.5, np.array(words_lists).T[y, x],    # 热力图种每个格子添加文本  数据项设置
                            horizontalalignment='center', verticalalignment='center',
                            )
            # sns.heatmap(score_lists, ax=ax,annot=True)
            # cmap='Greys'
            
            ax.set_xticks([i+0.5 for i in np.arange(len(x_label))])
            ax.set_yticks([i+0.5 for i in np.arange(len(y_label))])
            ax.set_xticklabels(x_label)
            ax.set_yticklabels(y_label)
            # ax.set_xticks(np.arange(len(x_label)+1)-.5)
            # ax.set_yticks(np.arange(len(range(1,word_N+1))+1)-.5)
            plt.savefig("./PublicSentiment/TapTimeEmotion_{}_{}.png".format(game_,the_date))
            # plt.show()
            continue
            exit()
        
        if KnowledgeMapping == True:
            contents = D
            weights = D_weights
            filter_value = max(int(filter_ratio * len(contents)),1)
            print ('filter_value:',filter_value)
            distances = {}
            nodes_weights = {}
            for idx,content in enumerate(contents):
                for i in range(0,len(content)):
                    for i_2 in range(i+1,len(content)):
                        if (content[i],content[i_2]) not in distances.keys() or (content[i_2],content[i]) not in distances.keys():
                            distances[(content[i],content[i_2])] = 1
                        else:
                            if (content[i],content[i_2]) in distances.keys():
                                distances[(content[i],content[i_2])] += 1
                            elif (content[i_2],content[i])  in distances.keys():
                                distances[(content[i_2],content[i])] += 1
                        if content[i] not in nodes_weights.keys():
                            nodes_weights[content[i]] = weights[idx]
                        else:
                            nodes_weights[content[i]] += weights[idx]
                        if content[i_2] not in nodes_weights.keys():
                            nodes_weights[content[i_2]] = weights[idx]
                        else:
                            nodes_weights[content[i_2]] += weights[idx]
            
            if filter_sign:
                distances_filtered = {}
                nodes_weights_filtered = {}
                for i in distances.keys():
                    if distances[i] > filter_value:
                        distances_filtered[i] = distances[i]
                        nodes_weights_filtered[i[0]] = nodes_weights[i[0]]
                        nodes_weights_filtered[i[1]] = nodes_weights[i[1]]
                distances = distances_filtered
                nodes_weights = nodes_weights_filtered
            # print (distances)
            # print (nodes_weights)
            G = nx.Graph()
            for k, v in nodes_weights.items():
                G.add_node(k, weight=v)
            for k, v in distances.items():
                G.add_edge(k[0], k[1], weight=v)
            
            visualize(G)

            exit()

        word_counts = collections.Counter(D_total) # 对分词做词频统计
        df_result = get_total_frequent(word_counts)
        print('*'* 10)
        print ('the final count result for {} shows below:'.format(file_name))
        print (df_result) # 输出检查
        highfre_words = list(df_result['word'])[:wordvec_show_len]

        if WordVec == True:
            # with open(inp, 'w') as f:
            #     for i in D:
            #         f.write(' '.join(i)+'\n')
            model = Word2Vec(D, vector_size =vecsize, workers=multiprocessing.cpu_count())
            model.save(output_vecmodel_path)
            model=Word2Vec.load(output_vecmodel_path)
            keys = list(model.wv.index_to_key)
            # print(keys)
            # print(type(keys))
            # print(list(keys)[0])

            # 获取词对于的词向量
            wordvector = []
            selected_keys = []
            for key in keys:
                if key not in highfre_words:
                    continue
                selected_keys.append(key)
                wordvector.append(model.wv[key])
            # print(wordvector)
            D_vec = wordvector
            kpca_transform = KernelPCA(n_components=len(wordvector[0]), kernel='rbf').fit_transform(D_vec)
            explained_variance = np.var(kpca_transform, axis=0)
            explained_variance_ratio = explained_variance / np.sum(explained_variance)
            print (np.cumsum(explained_variance_ratio))

            transformer = KernelPCA(n_components=n_components, kernel='rbf')
            newMat = transformer.fit_transform(D_vec)
            best_performance = 0.0
            for index, gamma in enumerate((0.01,0.1,1,10)):
                for index, k in enumerate((3,4,5,6)):
                    try:
                        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(newMat)
                        performance = metrics.calinski_harabasz_score(newMat, y_pred)
                        if performance>best_performance:
                            best_performance = performance
                            best_gamma = gamma
                            best_k = k
                        print ("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:",  performance)
                    except:
                        continue
            y_pred = SpectralClustering(n_clusters=best_k,gamma=best_gamma).fit_predict(newMat)
            print (best_k,best_gamma,best_performance)
            # print (y_pred)
            vec_tsne = TSNE(n_components=2)  
            # vec_tsne = KernelPCA(n_components=2, kernel='rbf') 
            ShowMat = vec_tsne.fit_transform(D_vec)  
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            fig = plt.figure()
            colors = ['red', 'black', 'blue', 'green', 'orange', 'pink']
            markers = ['o','v','^','<','>','1']    
            # scatter = plt.scatter(ShowMat[:,0], ShowMat[:,1], c=y_pred, s=2, cmap='rainbow')    
            # plt.title('')        
            # for i in np.unique(y_pred):    
            #     points = np.array(ShowMat[:,:2][y_pred == i])        
            #     hull = ConvexHull(points)    
            #     for simplex in hull.simplices:            
            #         plt.plot(points[simplex, 0], points[simplex, 1], 'k-', color=colors[i], linewidth=1.5)
            for i in np.unique(y_pred):
                plt.scatter(ShowMat[:, 0][y_pred == i], ShowMat[:, 1][y_pred == i], c=colors[i], marker=markers[i])
            # print (len(ShowMat[:,0]),len(ShowMat[:,1]),len(selected_keys))
            for i in range(len(selected_keys)):
                plt.text(ShowMat[:,0][i],ShowMat[:,1][i],selected_keys[i])
            # plt.legend()
            plt.show()
            exit()

        feature_select(D,'总计-') #所有词的TF-IDF值
        # if len(D)>100:
        #     get_frequent_items(D,'总计-',game_)
        

        D_neg = []
        D_neu = []
        D_pos = []
        for i in range(len(D_rate)):
            if D_rate[i] in [1,2]:
                D_neg.append(D[i])
            if D_rate[i] in [3]:
                D_neu.append(D[i])
            if D_rate[i] in [4,5]:
                D_pos.append(D[i])

        for sub_D in [D_neg,D_neu,D_pos]:
            if sub_D == D_neu:
                continue
            if sub_D == []:
                continue
            if sub_D == D_neg:
                print ('*'*10)
                print ('负面评论数：',len(sub_D),' 占比：', round(len(sub_D)/len(D),2))
                feature_select(sub_D,'负面-')
                # if len(sub_D)>100:
                #     get_frequent_items(sub_D,'负面-',game_)
            if sub_D == D_pos:
                print ('*'*10)
                print ('正面评论数：',len(sub_D),' 占比：', round(len(sub_D)/len(D),2))
                feature_select(sub_D,'正面-')
                # if len(sub_D)>100:
                #     get_frequent_items(sub_D,'正面-',game_)

        feature_select_cluster_pre(D,D_rate,game_)
        print ('left: ',file_list[file_idx+1:])


        print ('counting for {} is over!'.format(file_name))

        