#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import io
import base64
import sys
from flask import jsonify, request, session, make_response
from flask_login import login_user, logout_user, current_user, login_required
from datetime import datetime
import re
from PIL import Image

from . import image
from ..models import User, db, Task, Log
from ..util import image_analysis
import config

conf = config.Config()


@image.route('/', methods=['GET'])
def index():
    uid = request.args.get('userid')
    tasks = Task.query.filter_by(kind=1).filter(Task.uid == uid).all()
    tasks_datas = []
    for task in tasks:
        task_datas = dict()
        task_datas['name'] = task.taskname

        if task.dnn == 1:
            task_datas['model'] = 'ssd'
        else:
            task_datas['model'] = 'faster_rcnn'

        if task.state == 1:
            task_datas['state'] = '已完成'
        else:
            task_datas['state'] = '已失败'

        task_datas['date'] = str(task.createdtime)
        tasks_datas.append(task_datas)

    # 查询用户图片文件
    u = User.query.filter_by(uid=uid).first_or_404()
    userpath = conf.FILE_PATH + u.username
    # print(userpath)

    files_datas = []

    for filename in os.listdir(userpath):
        if filename.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            files_datas.append({'label': os.path.join(userpath, filename), 'key': filename})

    # for parent, dirnames, filenames in os.walk(userpath):
    #     for filename in filenames:
    #         if filename.lower().endswith(
    #                 ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
    #             files_datas.append({'label': os.path.join(parent, filename), 'key': filename})
    # print(files_datas)

    return jsonify({'status': '1', 'tableData': tasks_datas, 'tableOption': files_datas})


@image.route('/create', methods=['POST'])
def create():
    """从零开始创建一个图像分析任务"""
    if request.method == 'POST':
        # 获取前端请求的数据
        uid = request.json.get('userid')
        u = User.query.filter_by(uid=uid).first_or_404()
        name = request.json.get('name')   # 任务名
        filepath = request.json.get('file')   # 图像文件路径
        dnn = int(request.json.get('model'))   # 执行任务的模型，1为ssd，2为faster_rcnn
        # partition_level = int(request.json.get('pl'))  # 模型划分点，0-4，0为不执行模型划分

        # 判断任务名是否合法
        if not re.search('^[a-z][a-z0-9-]+$', name):
            return jsonify({'status': '0', 'msg': 'Name is invalid'})

        # 检测该次创建实例的实例名与该用户现有的实例名是否有重复（包括暂停的实例）
        if Task.query.filter_by(uid=uid).filter(Task.taskname == name).first():
            return jsonify({'status': '0', 'msg': 'Name is repetitive'})

        # 执行任务
        create_time = datetime.now()
        filename = filepath
        filepath = conf.FILE_PATH + u.username + '/' + filepath
        image_analysis.run_image_analysis(name, u.username, filename, filepath, dnn)

        end_time = datetime.now()

        user_log = Log(user=u,
                       type_flag=1,
                       content=u'创建任务 %s' % name)
        db.session.add(user_log)

        task = Task(user=u,
                    kind=1,
                    taskname=name,
                    filepath=filename,
                    dnn=dnn,
                    createdtime=create_time,
                    endtime=end_time)
        db.session.add(task)
        db.session.commit()

        return jsonify({'status': '1', 'msg': 'Task complete'})

    else:
        return jsonify("not Post")


@image.route('/detail', methods=['GET'])
def detail():
    uid = request.args.get('userid')
    u = User.query.filter_by(uid=uid).first_or_404()
    tname = request.args.get('url').split('/')[-1]
    task = Task.query.filter_by(taskname=tname).filter(Task.uid == uid).first_or_404()

    tasks_datas = list()
    tasks_datas.append({'info': '任务名称：' + task.taskname})

    if task.state == 1:
        tasks_datas.append({'info': '任务状态：已完成'})
    else:
        tasks_datas.append({'info': '任务状态：已失败'})

    if task.dnn == 1:
        tasks_datas.append({'info': '执行模型：ssd'})
    else:
        tasks_datas.append({'info': '执行模型：faster_rcnn'})

    tasks_datas.append({'info': '输入文件名：' + task.filepath})

    tasks_datas.append({'info': '任务创建时间：' + str(task.createdtime)})
    tasks_datas.append({'info': '任务完成时间：' + str(task.endtime)})

    return jsonify({'status': '1', 'tableData': tasks_datas})


@image.route('/getPic', methods=['GET'])
def getpic():
    uid = request.args.get('userid')
    u = User.query.filter_by(uid=uid).first_or_404()
    tname = request.args.get('url').split('/')[-1]
    task = Task.query.filter_by(taskname=tname).filter(Task.uid == uid).first_or_404()
    userpath = conf.FILE_PATH + u.username

    picpath = os.path.join(userpath, 'result', tname, task.filepath)
    print(picpath)

    img_stream = ''
    with open(picpath, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    # print(img_stream)
    return img_stream


@image.route('/delete', methods=['POST'])
def delete():
    """删除任务"""
    if request.method == 'POST':
        # 获取前端请求的数据
        uid = request.json.get('userid')
        u = User.query.filter_by(uid=uid).first_or_404()
        tname = request.json.get('name')   # 任务名
        task = Task.query.filter_by(taskname=tname).filter(Task.uid == uid).first_or_404()

        """更新用户的操作日志"""
        user_log = Log(user=u,
                       type_flag=2,
                       content=u'删除任务 %s' % tname)  # 记录用户的操作日志
        db.session.add(user_log)
        db.session.delete(task)
        db.session.commit()
        return jsonify({'status': '1', 'msg': 'Task deleted'})
