import re
import os
import time

import glob

# 下面这些包用于下载和解析牌谱
import eventlet
import xml.etree.ElementTree as ET
import urllib.request
import gzip

from tqdm import tqdm

import joblib


def get_urls_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    paipu_urls = []

    for line in lines:
        if "三鳳南喰赤" in line:
            continue
        
        replay_urls = re.findall('href="(.+?)">', line)
                
        log_urls = []

        for tmp in replay_urls:
            log_url_split = tmp.split("?log=")
            log_urls.append(log_url_split[0] + "log?" + log_url_split[1])
            
        paipu_urls = paipu_urls + log_urls
    return paipu_urls


def download_trunk(params):
    urls = params["urls"]
    start_idx = params["start"]
    end_idx = params["end"]
    hosts = params["hosts"]
    
    urls = urls[start_idx:end_idx]
    # ----------------- start ---------------------

    # 254281 matches in 2020 year, 4 players, phoenix
    for url in tqdm(urls):
        paipu_id = url.split("?")[-1]
        if os.path.exists("./paipuxmls/" + paipu_id + ".txt"):
            continue
        success_paipu_load = False
        for host in hosts:
            try:
                with eventlet.Timeout(2):
                    # start_time = time.time()
                    HEADER = {
                        'Host': host,
                        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:65.0) Gecko/20100101 Firefox/65.0',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive'
                    }

                    req = urllib.request.Request(url=url, headers=HEADER)
                    opener = urllib.request.build_opener()
                    response = opener.open(req)
                    paipu = gzip.decompress(response.read()).decode('utf-8')
                    success_paipu_load = True
                    # print("读取牌谱成功! （{}）".format(url))
                    break
            except:
                time.sleep(0.01)
                continue

        if not success_paipu_load:
            print("读取牌谱失败! （{}） 跳过".format(url))
            continue

        finish = False

        if finish:
            break

        # Save Paipu
        if not os.path.exists("./paipuxmls"):
            os.makedirs("./paipuxmls")
        text_file = open("./paipuxmls/" + paipu_id + ".txt", "w")
        text_file.write(paipu)
        text_file.close()


def load_all_urls():
    # -------- 读取2020年所有牌谱的url ---------------
    # 参考 https://m77.hatenablog.com/entry/2017/05/21/214529

    tot = 0
    paipu_urls = []

    filenames = glob.glob("./2020_paipu/scc*.html")
            
    for i, filename in enumerate(filenames):
        urls = get_urls_from_file(filename)
        print(i, len(urls))
        tot += len(urls)
        paipu_urls += urls

    print(f"牌谱的URL读取完毕！共有{len(paipu_urls)}个牌谱URL！")
    print("----------------------------------")
    print(tot)
    return paipu_urls



if __name__ == "__main__":
    urls = load_all_urls()
    
    trunks = [
        {"urls": urls, "start": 0, "end": 54281, "hosts": ["e3.mjv.jp", "e4.mjv.jp", "e5.mjv.jp", "k0.mjv.jp", "e.mjv.jp"]},
        {"urls": urls, "start": 54281, "end": 104281, "hosts": ["e4.mjv.jp", "e5.mjv.jp", "k0.mjv.jp", "e.mjv.jp", "e3.mjv.jp"]},
        {"urls": urls, "start": 104281, "end": 154281, "hosts": ["e5.mjv.jp", "k0.mjv.jp", "e.mjv.jp", "e3.mjv.jp", "e4.mjv.jp"]},
        {"urls": urls, "start": 154281, "end": 204281, "hosts": ["k0.mjv.jp", "e.mjv.jp", "e3.mjv.jp", "e4.mjv.jp", "e5.mjv.jp"]},
        {"urls": urls, "start": 204281, "end": 254281, "hosts": ["e.mjv.jp", "e3.mjv.jp", "e4.mjv.jp", "e5.mjv.jp", "k0.mjv.jp"]},
    ]

    joblib.Parallel(n_jobs=5)(joblib.delayed(download_trunk)(trunk) for trunk in trunks)