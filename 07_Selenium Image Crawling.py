"""
07_Selenium Image Crawling.py
"""

"""
문) 네이버 url(https://naver.com)을 대상으로 검색 입력 상자에 임의의 '동물' 이름을
    입력하여 이미지 10장을 '동물' 이름의 폴더에 저장하시오.  
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys # 엔터키 
from urllib.request import urlretrieve # image save
import os # 폴더 생성 및 이동 

def naver_img_crawler(search_name) :
    # 1. driver 객체 생성 
    path = r"C:\ITWILL\6_Tensorflow\tools" # driver 경로
    driver = webdriver.Chrome(path + "/chromedriver.exe")
    
    # 2. 대상 url 
    driver.get("https://naver.com") # naver url 이동 
    
    # 3. name 속성으로 element 가져오기 
    query_input = driver.find_element_by_name("query") # 입력상자  
    query_input.send_keys(search_name)
    query_input.send_keys(Keys.ENTER)
    
    # 4. '이미지' 링크 클릭     
    driver.find_element_by_xpath('/html/body/div[3]/div[1]/div/div[2]/div[1]/div/ul/li[2]/a').click()
    driver.implicitly_wait(3) # 3초 대기 
    
    
    '''
    이미지 상대경로 
    //*[@id="main_pack"]/section[2]/div/div[1]/div[1]/div[1]/div/div[1]/a/img
    //*[@id="main_pack"]/section[2]/div/div[1]/div[1]/div[2]/div/div[1]/a/img
    
    //*[@id="main_pack"]/section[2]/div/div[1]/div[1]/div[{i}]/div/div[1]/a/img
    '''
    # 5. 이미지 url 수집 
    image_urls = []
    for i in range(1, 11) :
        img = driver.find_element_by_xpath(f'//*[@id="main_pack"]/section[2]/div/div[1]/div[1]/div[{i}]/div/div[1]/a/img')
        image_urls.append(img.get_attribute('src'))
    
    print('수집된 이미지 개수 : ', len(image_urls))
    
    # 6. image 저장 폴더 생성과 이동 
    path = r'C:\ITWILL\6_Tensorflow\workspace\chapter07_ImageCrawling'
    os.mkdir(path + '/' + search_name) # 폴더 생성 
    os.chdir(path + '/' + search_name) # 폴더 이동 
    
    # 7. image url -> image file save 
    for i in range(len(image_urls)) :
        try :
            file_name = "test" + str(i+1) + ".jpg" # test1.jpg~test10.jpg
            urlretrieve(image_urls[i], file_name) # (url, filename)
            print(str(i+1) + '번째 image 저장됨')
        except :
            print('해당 url에 image 없음')     
    
    driver.close() # driver 닫기 

# 함수 호출 
naver_img_crawler('늑대') # 호랑이

