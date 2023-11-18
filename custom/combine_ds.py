import pathlib
import os
import json

def combine_ds(folder_path,wavs):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.ds')]
    txt_files.sort(key=lambda x:int(x.split('_10_')[-1].split('.ds')[0]))
    print("combine_ds",txt_files)
    all_fl = "all_fl.ds"
    all_fl_ph = os.path.join(folder_path, all_fl)
    json_datas = []
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, 'r') as file:
            # 读取ds文件内容为数组
            content = file.readlines()
            json_data = json.loads(''.join(content))
        
        
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
            file.write(json.dumps(json_datas))
    return all_fl_ph
    