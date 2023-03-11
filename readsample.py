import json
cn_seqcount = json.load(open('./cn_seqcount.json'))
jp_seqcount = json.load(open('./jp_seqcount.json'))
sentence =json.load(open('./sentence.json'))
print(cn_seqcount)
print(jp_seqcount)
print(sentence)