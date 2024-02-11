import requests
import pandas as pd
import os
df=pd.read_csv('Dataset/fitzpatrick17k.csv')
skincon_fitz_df=pd.read_csv('Dataset/image.csv')
count=0
idxs=[78,217,549,986,1092,1756,1987,2136,2471,2552,2593,2688,3084,3602]
for index in idxs:
    pd.reset_option('max_colwidth')
    imageid=skincon_fitz_df.iloc[index]['ImageID'][:-4]
    idx=df.index[df['md5hash'] == imageid].tolist()
    url=df.iloc[idx[0]]['url']
    file_name='Dataset/fitz_images/{imageid}.jpg'.format(imageid=imageid)
    # try:
    #     image = Image.open(file_name).convert("RGB")
    # # except:
    # #     print(index,"====")
    count+=1
    # print('url: ', url," file_name: ", file_name)
    os.system('curl -L {url} > {file_name}'.format(url=url,file_name=file_name))
# print(count)
    
    

