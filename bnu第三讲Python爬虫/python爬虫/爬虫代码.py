# -*- coding: utf-8 -*-



import requests
import re


#获取网页信息，requests.get是调用requests库的get函数来获取网页返回的信息
#url是所爬页面url链接，会在下面的main函数中赋值，timeout设定超时时间
#
def getHtmlText(url):
    try:
        r = requests.get(url,timeout = 30)  #最基本的形式，大家可以上网了解一下requests库其它功能
        r.raise_for_status()     #检查返回信息是否成功     
        r.encoding = r.apparent_encoding  #修正响应内容的编码方式
        return r.text       #返回响应信息的字符串形式
    except:
        return '获取信息失败'

#搜寻整理目标信息
#查看我们所需信息的格式，利用re.findall函数可以找出所以符合该格式的信息，找的时候可以上网看一下正则表达式的形式
def parsePage(ist,html):
    itle = re.findall(r'<span class="title">(.*?)</span>',html)  #利用re搜寻电影名称
    mas = re.findall(r'<span class="inq">(.*)</span>',html)
    name = []
    for tle in itle:        #利用for循环去除不需要的标题
        if tle[0]!='&':                      
            name.append(tle)
    for i in range(len(name)):     #将标题和对应的电影经典评价放到一个列表中         
        title = name[i]
        mass = mas[i]
        ist.append([title,mass])

#将获取的电影信息进行整合
#使用format进行格式化输出
#{0:^30}中的0是一个序号，表示格式化输出的第0个字符，依次累加；
#{0:^30}中的30表示输出宽度约束为30个字符；
#{0:^30}中的^表示输出时居中对齐;<和>分别为左右两端对齐
#形式为{n}(n为整数)的占位符将被format()中第n个参数所代替
def printList(ist):
    tplt = "{0:^2}\t{1:{3}^10}\t{2:<10}"     #\t等同于tab
    print(tplt.format("序号","电影名称","电影经典评语",chr(12288)))
    count = 0               #使用chr(12288)的作用是对未对齐部分用中文字符填充
    for g in ist:           #给电影前加上它的排名
        count = count + 1
        print(tplt.format(count,g[0],g[1],chr(12288)))

#调用各个函数进行整合
#for循环用于爬取不同页面的电影信息
#对比要爬取的各个页面的网址，start_url是网址的相同之处，str（25*i）是不同之处
def main():    
    page = 4            #爬取电影的页数
    start_url = 'https://movie.douban.com/top250?start='
    Llist = []
    for i in range(page):
        try:
            url = start_url + str(25*i)
            html = getHtmlText(url)
            parsePage(Llist,html)
        except:
            print ('爬取失败')
    printList(Llist)
main()


#测试代码
def gethtmltext(url):
    try:
        r = requests.get(url,timeout = 30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return '爬取失败'
html = gethtmltext('https://movie.douban.com/top250?start=0&filter=')



itle = re.findall(r'<span class="title">(.*?)</span>',html)
mas = re.findall(r'<span class="inq">(.*)</span>',html)
name = []
for tle in itle:  
    if tle[0]!='&':                      
        name.append(tle)
ist=[]
for i in range(len(name)):
    title = name[i]
    mass = mas[i]
    ist.append([title,mass])
ist
ist[0]