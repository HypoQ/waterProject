import json
import requests
import time
from apscheduler.schedulers.blocking import BlockingScheduler  # 引入后台
import os
from utils.file import FileUtil
# from utils.requst import postres
# from detect import run,main,parse_opt
from detect import run,main,parse_opt
SPATH = './testfile/input'
TPATH = './testfile/output'

Source = './testfile/input/test1.mp4'
Weight = './runs/train/exp3/weights/best.pt'


def postres(time,dsc,site,img_name,img_path):
    print("准备提交数据")
    url = 'http://113.250.49.15:8084/smartPlant/portal/drone/video/droneInfo'
    data = {"order_time": time, "dsc": dsc, "site": site}
    files = {"files":(img_name, open(img_path, 'rb'), "image/png", {})}
    # print(files)

    print(data)
    print(files)
    res = requests.request("POST", url, data=data, files=files)
    print(res.text)
    # print("请求发送完毕")

def autodetect(srcpath = SPATH, tarpath = TPATH):
    futi = FileUtil(srcpath,tarpath)
    fname = futi.containfile()
    print("fname is ",fname)
    if fname!= "":
        print('检测到新视频')
        print('执行检测算法')
    # 调用检测算法
        filepath = os.path.join(srcpath,fname)
        # 获取检测结果，并打印

        res = main(Source = filepath,Weight =Weight)
        # {'time': '2022-10-18 13:30:54.342617', 'path': 'runs/tmp/exp10', 'class': 'water_leakage', 'frame': '0'}
        if res:
            print()
            print('检测完成，检测结果为', res)
            order_time = res['time']
            desc = res['desc']
            site = res['site']
            img_name = res['img_name']
            img_path = os.path.join(res['path'], res['img_name'])

            # 发送http请求
            #    (time, dsc, site, img_name, img_path)
            print("准备发送请求")
            postres(time=order_time, dsc=desc, site=site, img_name=img_name, img_path=img_path)
            print("发送请求成功")
            # 清理文件夹
            # 检测完成后移动文件
            print()
            print("检测完成后，开始移动文件")
            futi.mvfile()
        else:
            print('检测完成，检测结果为', res)
            print("检测完成后，开始移动文件")
            futi.mvfile()

    else:
        print('没有新视频')

sched = BlockingScheduler()
autodetect()
sched.add_job(autodetect, 'interval', seconds=60, max_instances=1)
sched.start()