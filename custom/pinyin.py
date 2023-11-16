from pypinyin import pinyin, lazy_pinyin, Style

def pinyin2ph_func(str):
    pinyin2phs = {'AP': '<AP>', 'SP': '<SP>'}
    with open('custom/pinyin2ph.txt') as rf:
        for line in rf.readlines():
            elements = [x.strip() for x in line.split('|') if x.strip() != '']
            pinyin2phs[elements[0]] = elements[1]
            
    pinyins = lazy_pinyin(str, strict=False)
    ph_per_word_lst = [pinyin for pinyin in pinyins if pinyin.strip() in pinyin2phs]

    return ph_per_word_lst




def pinyin2ph_word(str):
    pinyin2phs = {'AP': '<AP>', 'SP': '<SP>'}
    with open('dictionaries/opencpop.txt') as rf:
        for line in rf.readlines():
            elements = [x.strip() for x in line.split(' ') if x.strip() != '']
            if(len(elements)==1):
                pinyin2phs[elements[0].split("\t")[0]] = elements[0].split("\t")[1]
            if(len(elements)==2):
                pinyin2phs[elements[0].split("\t")[0]] = elements[0].split("\t")[1]+' '+elements[1]
    #print(pinyin2phs)
    pinyins = lazy_pinyin(str, strict=False)
    ph_per_word_lst = [pinyin2phs[pinyin] for pinyin in pinyins if pinyin.strip() in pinyin2phs]

    return ph_per_word_lst



def replace_seq(ostr,tstr):
    nstr=[]
    j=0
    for i,ph in enumerate(ostr.split()):
        if ph == "AP" or ph == "SP":
            nstr.append(ph)
            continue
        else:
            j=j+1
        if(j>len(" ".join(pinyin2ph_word(tstr)).split())):
            nstr.append("SP")
        else:
            nstr.append(" ".join(pinyin2ph_word(tstr)).split()[j-1])
        
    return nstr