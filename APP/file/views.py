#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import io
import sys
import time
import zipfile
from flask import jsonify, request, session, make_response, send_file
from flask_login import login_user, logout_user, current_user, login_required
from datetime import datetime
import re

from . import file
from ..models import User, db, Task, Log
import config

conf = config.Config()


@file.route('/', methods=['GET'])
def index():
    uid = request.args.get('userid')
    files_datas = []
    results_datas = []

    # 查询用户文件夹
    u = User.query.filter_by(uid=uid).first_or_404()
    userpath = conf.FILE_PATH + u.username
    # print(userpath)

    for filename in os.listdir(userpath):
        if len(filename.split('.')) == 1:
            full_path = os.stat(os.path.join(userpath, filename))
            files_datas.append({'name': filename, 'date': str(time.asctime(time.localtime(full_path.st_ctime)))})

    for filename in os.listdir(userpath):
        if len(filename.split('.')) != 1:
            full_path = os.stat(os.path.join(userpath, filename))
            files_datas.append({'name': filename, 'date': str(time.asctime(time.localtime(full_path.st_ctime)))})

    # 查询用户result文件夹
    resultpath = os.path.join(userpath, 'result')
    for filename in os.listdir(resultpath):
        if len(filename.split('.')) == 1:
            results_datas.append({'label': os.path.join(resultpath, filename), 'key': filename})

    return jsonify({'status': '1', 'tableData': files_datas, 'tableOption': results_datas})


@file.route('/upload', methods=['POST'])
def upload():
    uid = request.form.get("userid")
    fileObj = request.files.get("file")

    # 查询用户文件夹
    u = User.query.filter_by(uid=uid).first_or_404()
    userpath = conf.FILE_PATH + u.username

    fileObj.save(os.path.join(userpath, fileObj.filename))  # 保存文件

    return jsonify({'status': '1', 'msg': '上传成功！'})


@file.route('/download', methods=['GET'])
def download():
    uid = request.args.get('userid')
    filename = request.args.get('name')

    # 查询用户文件夹
    u = User.query.filter_by(uid=uid).first_or_404()
    userpath = conf.FILE_PATH + u.username

    filepath = os.path.join(userpath, filename)

    fileobj = open(file=filepath, mode='rb')
    resdata = fileobj.read()

    return resdata


@file.route('/downloadresult', methods=['GET'])
def downloadresult():
    uid = request.args.get('userid')
    filename = request.args.get('name')

    # 查询用户文件夹
    u = User.query.filter_by(uid=uid).first_or_404()
    userpath = conf.FILE_PATH + u.username

    filepath = os.path.join(userpath, 'result', filename)
    zippath = filepath + '.zip'

    def compressFolder(folderPath, compressPathName):
        '''
        :param folderPath: 文件夹路径
        :param compressPathName: 压缩包路径
        :return:
        '''
        zip = zipfile.ZipFile(compressPathName, 'w', zipfile.ZIP_DEFLATED)

        for path, dirNames, fileNames in os.walk(folderPath):
            fpath = path.replace(folderPath, '')
            for name in fileNames:
                fullName = os.path.join(path, name)

                name = fpath + '\\' + name
                zip.write(fullName, name)

        zip.close()

    compressFolder(filepath, zippath)

    fileobj = open(file=zippath, mode='rb')
    resdata = fileobj.read()

    return resdata


@file.route('/delete', methods=['POST'])
def delete():
    """删除文件"""
    if request.method == 'POST':
        # 获取前端请求的数据
        uid = request.json.get('userid')
        fname = request.json.get('name')  # 文件名

        # 查询用户文件夹
        u = User.query.filter_by(uid=uid).first_or_404()
        userpath = conf.FILE_PATH + u.username
        filepath = os.path.join(userpath, fname)

        if os.path.exists(filepath):  # 如果文件存在
            if len(fname.split('.')) != 1:  # 当个文件
                os.remove(filepath)
            else:  # 文件夹
                os.removedirs(filepath)

        """更新用户的操作日志"""
        user_log = Log(user=u,
                       type_flag=2,
                       content=u'删除文件 %s' % fname)  # 记录用户的操作日志
        db.session.add(user_log)
        db.session.commit()
        return jsonify({'status': '1', 'msg': 'File deleted'})
