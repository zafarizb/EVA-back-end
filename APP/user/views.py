#! /usr/bin/env python
# -*- coding: utf-8 -*-
from . import user
from ..models import User, db
from flask import jsonify, request, session
from flask_login import login_user, logout_user, current_user, login_required
from datetime import datetime


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
            print(data)
            rec_username = data['username']
            rec_password = data['password']
            u = User.query.filter_by(username=rec_username).first()
            if u:
                if u.verify_password(rec_password):
                    login_user(u)
                    u.last_login = datetime.now()
                    db.session.commit()
                    print(current_user)
                    return jsonify({'status': '0', 'msg': 'login success'})
                else:
                    return jsonify({'status': '1', 'msg': 'login fail'})
            else:
                return jsonify({'status': '2', 'msg': 'no user'})
    else:
        return jsonify("not POST")
