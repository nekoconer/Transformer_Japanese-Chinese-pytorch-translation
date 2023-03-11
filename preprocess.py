import json

PATH = './filted.txt'
# f = open(PATH, encoding='utf-8')
# byt = f.readline()
# print(byt)
# byt1, byt2 = byt.split()
# print(byt1, byt2)
jp_sen = []
ch_sen = []
try:
    f = open(file=PATH, mode='r',encoding='utf-8', buffering=True)
    for line in f.readlines():
        print(line)
        byt1, byt2= line.split(maxsplit=1)
        jp_sen.append(byt1)
        ch_sen.append(byt2)
        print(byt1, byt2)
except IOError as e:
    print("报错" + str(e))
finally:
    if 'f' in globals():
        f.close
    print()

with open("./jp_sen.json", "w", encoding="utf-8") as fp:
    json.dump(jp_sen, fp)
with open("./cn_sen.json", "w", encoding="utf-8") as fp:
    json.dump(ch_sen, fp)