from streamlit_echarts import JsCode
from streamlit_echarts import st_echarts
import requests,os

FASTAPI_URL1 = os.getenv('FASTAPI_URL1')
FASTAPI_URL2 = os.getenv('FASTAPI_URL2')
FASTAPI_URL3 = os.getenv('FASTAPI_URL3')

def extract_time(input_title):
    if input_title:
        time1 = requests.get(f"{FASTAPI_URL1}/getTime1?title={input_title}")
        response1 = time1.json().get("data", "")
    return response1

def single_bar_all():
    
    
    options = {
        "grid": {
        "left": "10%",  # 왼쪽 여백
        "right": "10%",  # 오른쪽 여백
        "width": "80%",  # 그리드의 너비
        },
        "xAxis": {
            "type": "category",
            "data": ["bart", "t5", "t5+의미론적추론"],
        },
        "yAxis": {
            "name": "%",
            "type": "value",
            "min": 0,  
            "max": 100, 
            "interval":20,  # y축 간격 설정
        },
        "series": [
            {
                "data": [30, 50, 10],
                "type": "bar",
                "barWidth": '30%',  # 막대 너비 설정
                "itemStyle": {
                    "color": "#EFE4B0"  # 막대 색상 설정
                }
            },
               
        ],
    }
    st_echarts(options, height="400px")
    

    
def single_bar_time():
    options = {
        "xAxis": {
            "type": "category",
            "data": ["bart", "t5", "t5+의미론적추론"],
        },
        "yAxis": {
            "name": "%",
            "type": "value",
            "min" : 0,
            "max" : 100,
            "interval":20,  # y축 간격 설정
        },
        "series": [
            {
                "data": [10, 50, 10],
                "type": "bar",
                "barWidth": '30%',  # 막대 너비 설정
                "itemStyle": {
                    "color": "#F0BD75"  # 막대 색상 설정
                }
            },
            
        ],
    }
    st_echarts(options, height="400px")
def single_bar_accuracy():
    options = {
        "legend": {
        "data": ["keybert", "textrank"],
        "top": "top",  # 범례를 상단에 배치
        "right": "right",  # 범례를 오른쪽에 배치
        },
        "xAxis": {
            "type": "category",
            "data": ["bart", "t5", "t5+의미론적추론"],
           
        }, 
        "yAxis": {
            "name": "%",
            "type": "value",
        },
        "series": [
            {
                "name": "keybert",
                "data": [80, 90, 20], # 실제 데이터로 변경해야함
                "type": "bar",
                "barWidth": '25%',  # 막대 너비 설정
                "itemStyle": {
                    "color": "#5470C6"  # 막대 색상 설정
                }
            }, 
            {
                "name": "textrank",
                "data": [90, 10, 40],
                "type": "bar",
                "barWidth": '25%',  # 막대 너비 설정
                "itemStyle": {
                    "color": "#91CC75"  # 막대 색상 설정
                }
            }
        ],
    }
    st_echarts(options, height="400px")

    
    #5470C6
    #91CC75