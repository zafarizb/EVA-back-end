#! /usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from flask import current_app
from . import db, login_manager

""" 用户管理
@User:用户表

用户登录、注册、认证邮箱等
"""


class User(UserMixin, db.Model):
    __tablename__ = 'user'
    uid = db.Column(db.Integer, primary_key=True)
    # 用户名、密码与邮箱，以邮箱作为登录主要凭据
    username = db.Column(db.String(128), unique=True, index=True)
    passWord = db.Column(db.String(128), nullable=False)
    email = db.Column(db.String(128), unique=True, index=True, nullable=False)

    # 基本信息
    description = db.Column(db.String(128))
    real_name = db.Column(db.String(128))
    phone = db.Column(db.String(128))
    address = db.Column(db.String(128))
    last_login = db.Column(db.DateTime(), default=datetime.now)
    createtime = db.Column(db.DateTime(), default=datetime.now)

    # 以下函数分别用于对用户密码进行读取保护、散列化以及验证密码
    @property
    def id(self):
        return self.uid

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        password_hash = generate_password_hash(password)
        self.passWord = password_hash

    @property
    def raw_password(self):
        raise AttributeError('password is not a readable attribute')

    @raw_password.setter
    def raw_password(self, password):
        """用于直接存储来自ehpc的密码哈希"""
        self.passWord = password

    def verify_password(self, password):
        return check_password_hash(self.passWord, password)

    # 以下两个函数用于token的生成和校验
    def generate_token(self, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps({'id': self.uid})

    @staticmethod
    def verify_token(token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return None
        uid = data.get('id')
        if uid:
            return User.query.get(uid)
        return None


# 插件flask_login的回调函数，用于读取用户
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)
