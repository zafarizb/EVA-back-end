#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from flask import jsonify, request, session, make_response
from flask_login import login_user, logout_user, current_user, login_required
from datetime import datetime
import re
import base64

from . import video
from ..models import User, db, Task, Log
from ..util import video_analysis
import config

conf = config.Config()


# 主页
@video.route('/', methods=['GET'])
def index():
    uid = request.args.get('userid')
    tasks = Task.query.filter_by(kind=2).filter(Task.uid == uid).all()
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

    # 查询用户文件夹
    u = User.query.filter_by(uid=uid).first_or_404()
    userpath = conf.FILE_PATH + u.username
    # print(userpath)

    dirs_datas = []

    for filename in os.listdir(userpath):
        if len(filename.split('.')) == 1 and filename != 'result':
            dirs_datas.append({'label': os.path.join(userpath, filename), 'key': filename})

    return jsonify({'status': '1', 'tableData': tasks_datas, 'tableOption': dirs_datas})

# 创建视频分析任务
@video.route('/create', methods=['POST'])
def create():
    """从零开始创建一个视频分析任务"""
    if request.method == 'POST':
        # 获取前端请求的数据
        uid = request.json.get('userid')
        u = User.query.filter_by(uid=uid).first_or_404()
        name = request.json.get('name')   # 任务名
        dirname = request.json.get('file')   # 图像文件夹路径
        dnn = int(request.json.get('model'))   # 执行任务的模型，1为ssd，2为faster_rcnn
        partition_level = int(request.json.get('pl'))  # 模型划分点，0-4，0为不执行模型划分

        # 判断任务名是否合法
        if not re.search('^[a-z][a-z0-9-]+$', name):
            return jsonify({'status': '0', 'msg': 'Name is invalid'})

        # 检测该次创建实例的实例名与该用户现有的实例名是否有重复（包括暂停的实例）
        if Task.query.filter_by(uid=uid).filter(Task.taskname == name).first():
            return jsonify({'status': '0', 'msg': 'Name is repetitive'})

        # 执行任务
        create_time = datetime.now()
        dirpath = conf.FILE_PATH + u.username + '/' + dirname
        video_analysis.run_video_analysis(name, u.username, dirpath, dnn, partition_level)

        end_time = datetime.now()

        user_log = Log(user=u,
                       type_flag=1,
                       content=u'创建任务 %s' % name)
        db.session.add(user_log)

        task = Task(user=u,
                    kind=2,
                    taskname=name,
                    filepath=dirname,
                    dnn=dnn,
                    pl=partition_level,
                    createdtime=create_time,
                    endtime=end_time)
        db.session.add(task)
        db.session.commit()

        return jsonify({'status': '1', 'msg': 'Task complete'})

    else:
        return jsonify("not Post")

# 查看任务详情
@video.route('/detail', methods=['GET'])
def detail():
    uid = request.args.get('userid')
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

    if task.pl == 0:
        tasks_datas.append({'info': '模型划分：不执行'})
    else:
        tasks_datas.append({'info': '模型划分：划分点-' + str(task.pl)})

    tasks_datas.append({'info': '输入文件夹：' + task.filepath})

    tasks_datas.append({'info': '任务创建时间：' + str(task.createdtime)})
    tasks_datas.append({'info': '任务完成时间：' + str(task.endtime)})

    dirs_datas = []

    # 查询用户文件夹
    u = User.query.filter_by(uid=uid).first_or_404()
    userpath = conf.FILE_PATH + u.username
    resultpath = os.path.join(userpath, 'result', tname)

    for filename in os.listdir(resultpath):
        dirs_datas.append({'label': filename, 'value': os.path.join(resultpath, filename)})

    return jsonify({'status': '1', 'tableData': tasks_datas, 'options': dirs_datas})


@video.route('/getPic', methods=['GET'])
def getpic():
    uid = request.args.get('userid')
    u = User.query.filter_by(uid=uid).first_or_404()
    tname = request.args.get('url').split('/')[-1]
    task = Task.query.filter_by(taskname=tname).filter(Task.uid == uid).first_or_404()
    pagenum = request.args.get('pagenum')

    userpath = conf.FILE_PATH + u.username

    picpath = os.path.join(userpath, 'result', tname, pagenum)
    print(picpath)

    img_stream = ''
    with open(picpath, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    # print(img_stream)
    return img_stream


# 删除任务
@video.route('/delete', methods=['POST'])
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
