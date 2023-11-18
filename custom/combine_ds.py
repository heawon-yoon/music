import pathlib
import os
import json
from custom.pinyin import replace_seq, pinyin2ph_word

def combine_ds(folder_path,wavs,text_inputs):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.ds')]
    txt_files.sort(key=lambda x:int(x.split('_10_')[-1].split('.ds')[0]))
    print("combine_ds",txt_files)
    all_fl = "all_fl.ds"
    all_fl_ph = os.path.join(folder_path, all_fl)
    json_datas = []
    for i,txt_file in enumerate(txt_files):
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, 'r') as file:
            # 读取ds文件内容为数组
            content = file.readlines()
            json_data = json.loads(''.join(content))
        
        ostr = json_data[0]["ph_seq"]
        ph_dur_list = json_data[0]["ph_dur"].split()
        pinyin2phs = {'AP': '<AP>', 'SP': '<SP>'}
        with open('dictionaries/opencpop.txt') as rf:
            for line in rf.readlines():
                elements = [x.strip() for x in line.split(' ') if x.strip() != '']
                if(len(elements)==1):
                    pinyin2phs[elements[0].split("\t")[0]] = elements[0].split("\t")[1]
                if(len(elements)==2):
                    pinyin2phs[elements[0].split("\t")[0]] = elements[0].split("\t")[1]+' '+elements[1]
        #print(pinyin2phs)
    
    
        nstr=[]
        j=0
        q=None
        tstr = text_inputs.split("\n")[i]
        tstr_list = " ".join(pinyin2ph_word(tstr)).split()
        pur_cus = 0
        #print(ostr.split())
        #print(tstr)
        for i,ph in enumerate(ostr.split()):
            if(i==q):
                continue
            if ph == "AP" or ph == "SP":
                nstr.append(ph)
                continue
                
            #j=j+1    
            # print(ph)
            # print(ostr.split()[i-1])
            # print(ostr.split()[i-1]+ph)
            # print(pinyin2phs.get(ostr.split()[i-1]+ph))
            
            if(j-pur_cus>len(" ".join(pinyin2ph_word(tstr)).split())):
                #nstr.append("SP")
                continue
            #print(pinyin2ph_word(tstr))
            #print(j)
            if(j>=len(tstr_list)):
                nstr.append("SP")
                continue
            #print(tstr_list[j])
            #print(pinyin2phs.get(tstr_list[j]))
            #print(pinyin2phs.get(tstr_list[j]+tstr_list[j-1]))
            #是单音节
            if pinyin2phs.get(ph) is not None and pinyin2phs.get(ph) != '' and pinyin2phs.get(ostr.split()[i-1]+ph) is None :
                #print("A")
                if pinyin2phs.get(tstr_list[j]) is not None and pinyin2phs.get(tstr_list[j]) != '' and pinyin2phs.get(tstr_list[j-1]+tstr_list[j]) is None:
                    #print("A1")
                    nstr.append(" ".join(pinyin2ph_word(tstr)).split()[j])
                else:
                    #print("A2")
                    nstr.append(tstr_list[j])
                    if j<len(tstr_list):
                        nstr.append(tstr_list[j+1])
                        j=j+1    
                    ph_dur_list.insert(i+pur_cus+1, str(float(ph_dur_list[i+pur_cus])/2))
                    ph_dur_list[i+pur_cus] = str(float(ph_dur_list[i+pur_cus])/2)
                    pur_cus += 1
            else:
                if pinyin2phs.get(tstr_list[j]) is not None and pinyin2phs.get(tstr_list[j-1]) != '' and pinyin2phs.get(tstr_list[j-1]+tstr_list[j]) is None:
                    #print("B1")
                    nstr.append(" ".join(pinyin2ph_word(tstr)).split()[j])
                    q = i+1
                    ph_dur_list[i+pur_cus] = str(float(ph_dur_list[i+pur_cus])+float(ph_dur_list[i+pur_cus+1]))
                    del ph_dur_list[i+pur_cus+1]
                    pur_cus -= 1
                else:
                    #print("B2")
                    nstr.append(" ".join(pinyin2ph_word(tstr)).split()[j])
            #print(" ".join(pinyin2ph_word(tstr)).split()[j])
            j=j+1          
        json_data[0]["ph_seq"] =  " ".join(nstr)
        json_data[0]["ph_dur"] = " ".join(ph_dur_list)
        #print(json_data[0]["ph_seq"])
        for item in wavs:
             #print(f'{ json_data[0]["offset"]},{item["offset"]},{item["wav_path"]},{file_path}')
             if os.path.splitext(os.path.basename(item["wav_path"]))[0] == os.path.splitext(os.path.basename(file_path))[0]:
            
                 json_data[0]["offset"] = item["offset"]
        json_datas.append(json_data[0])
        
        with open(file_path, 'w') as file:
            # 将修改后的JSON数据写入文件
            file.write(json.dumps(json_data,indent=2, ensure_ascii=False))
    #print(json_datas)
    with open(all_fl_ph, 'w') as file:
        # 将修改后的JSON数据写入文件
        file.write(json.dumps(json_datas,indent=2, ensure_ascii=False))
    return all_fl_ph
    