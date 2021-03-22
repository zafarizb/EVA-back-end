#! /usr/bin/env python
# -*- coding: utf-8 -*-
from . import user
from ..models import User, db
import config
import os
from flask import jsonify, request, session
from flask_login import login_user, logout_user, current_user, login_required
from datetime import datetime

conf = config.Config()


@user.route('/Register', methods=["POST"])
def register():
    if request.method == 'POST':
        username = request.json.get('username')
        email = request.json.get('email')
        password = request.json.get('password')

        # 此处可继续完善，如过滤特殊字符等
        u = User.query.filter_by(username=username).first()
        if u:
            return jsonify({'status': '0', 'msg': 'user already exist '})
        else:
            reg_user = User()
            reg_user.email = email
            reg_user.password = password
            reg_user.username = username
            db.session.add(reg_user)
            db.session.commit()

            # 创建用户的存储空间
            userpath = conf.FILE_PATH + username
            resultpath = os.path.join(userpath, 'result')
            if not os.path.exists(userpath):
                os.makedirs(userpath)
                os.makedirs(resultpath)

        return jsonify({'status': '1', 'msg': 'register success'})
    else:
        return jsonify("not Post")


@user.route("/Login", methods=["POST"])
def login():
    if request.method == 'POST':
        if not request.json:
            return jsonify('not json')
        else:
            data = request.get_json()
            # print(data)
            rec_username = data['username']
            rec_password = data['password']
            u = User.query.filter_by(username=rec_username).first()
            if u:
                if u.verify_password(rec_password):
                    login_user(u)
                    u.last_login = datetime.now()
                    db.session.commit()
                    print(current_user)
                    return jsonify({'status': '0', 'msg': 'login success', 'session': u.uid})
                else:
                    return jsonify({'status': '1', 'msg': 'login fail'})
            else:
                return jsonify({'status': '2', 'msg': 'no user'})
    else:
        return jsonify("not POST")


@user.route("/detail", methods=["GET"])
def detail():
    uid = request.args.get('userid')
    u = User.query.filter_by(uid=uid).first_or_404()

    res_datas = dict()
    res_datas['name'] = u.username
    res_datas['email'] = u.email
    res_datas['realname'] = u.real_name
    res_datas['phone'] = u.phone
    res_datas['address'] = u.address

    return jsonify({'status': '1', 'data': res_datas})


@user.route("/modify", methods=["POST"])
def modify():
    # 获取前端请求的数据
    uid = request.json.get('userid')
    u = User.query.filter_by(uid=uid).first_or_404()
    address = request.json.get('address')
    phone = request.json.get('number')
    realname = request.json.get('realname')

    u.address = address
    u.phone = phone
    u.real_name = realname
    db.session.commit()

    return jsonify({'status': '1', 'msg': 'Modify complete'})
