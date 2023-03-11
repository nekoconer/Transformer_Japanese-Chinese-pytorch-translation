import json

sentence = []
tmp = []
cn_afterseq = json.load(open('./cn_afterseq.json'))
jp_afterseq = json.load(open('./jp_afterseq.json'))
for i in range(len(cn_afterseq)):
    tmp.append(cn_afterseq[i].strip("\n"))
    strS = jp_afterseq[i]
    strS = "S"+" "+strS.strip('\n')
    tmp.append(strS)
    strE = jp_afterseq[i]
    strE = strE.strip('\n') + " " + 'E'
    tmp.append(strE)
    sentence.append(tmp)
    print(tmp)
    tmp = []
with open('./sentence.json','w',encoding='utf-8') as fp:
    json.dump(sentence, fp)