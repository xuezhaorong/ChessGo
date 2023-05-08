import requests
import time
from bs4 import BeautifulSoup
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.48"
}


def get_manual(manual_num):
    url = "https://www.xqbase.com/xqbase/?gameid="+str(manual_num)
    response = requests.get(url, headers=headers)
    content = response.text
    soup = BeautifulSoup(content, "html.parser")
    title = soup.find("span").find("b").string[:3]
    table = soup.find("pre").string
    table = table.split(" ")
    manual = [item.strip() for item in table if len(
        item) > 2]
    ls = ["１", "２", "３", "４", "５", "６", "７", "８", "９"]
    for i in range(len(manual)):
        if len(manual[i]) > 4:
            manual[i] = manual[i][:4]
        else:
            manual[i] = manual[i]
        # 处理全角数字的问题
        if manual[i][3] in ls:
            ls2 = list(manual[i])
            ls2[3] = chr(ord(ls2[3])-65248)
            if ls2[1] in ls:
                ls2[1] = chr(ord(ls2[1])-65248)
            manual[i] = "".join(ls2)
        # 处理()的问题
        if "(" in manual[i]:
            del manual[i]
            break
    print(manual)
    path = title + "_" + str(manual_num) + ".txt"
    with open(path, 'w', encoding='utf-8') as f:
        for item in manual:
            f.write(item+"\n")


if __name__ == "__main__":
    for i in range(10825, 12142):
        get_manual(i)
    # time.sleep(10)
