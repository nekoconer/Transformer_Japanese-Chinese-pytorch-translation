import json
import sys
import jieba
import MeCab

jp_pre = json.load(open('./jp_sen.json'))
cn_pre = json.load(open('./cn_sen.json'))
cn_segment = {}
jp_segment = {}
jp_segment['P'] = 0
jp_segment['E'] = 1
jp_segment['S'] = 2
cn_segment['P'] = 0
cn_after = []
jp_after = []
count = 1
count1 = 3
time = 0
for i in cn_pre:
    if time == 500:
        break
    time += 1
    Ajieba = jieba.cut(i ,cut_all= False)
    Ajieba = (" ".join(Ajieba))
    print(Ajieba)
    cn_after.append(Ajieba)
    Ajieba = Ajieba.split(" ")
    for j in Ajieba:
        if j not in cn_segment:
            cn_segment[str(j)] = count
            count += 1
with open("./cn_seqcount.json", "w", encoding="utf-8") as fp:
     json.dump(cn_segment, fp)
with open('./cn_afterseq.json',"w",encoding='utf-8') as fp:
    json.dump(cn_after,fp)

time = 0
mecab_tagger = MeCab.Tagger('-Owakati')
for i in jp_pre:
    if time == 500:
        break
    time += 1
    Amecab = mecab_tagger.parse(i)
    jp_after.append(Amecab)
    print(Amecab)
    Amecab = Amecab.split(" ")
    for j in Amecab:
        print(j)
        if j not in jp_segment:
            jp_segment[str(j)] = count1
            count1 += 1
with open('./jp_seqcount.json', "w", encoding="utf-8") as fp:
    json.dump(jp_segment, fp)
with open('./jp_afterseq.json',"w",encoding='utf-8') as fp:
    json.dump(jp_after, fp)