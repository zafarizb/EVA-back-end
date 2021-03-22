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

    logs = db.relationship('Log', backref='user', lazy='dynamic')
    tasks = db.relationship('Task', backref='user', lazy='dynamic')

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


"""
任务管理
@Task:任务表
@Log：存储用户的操作日志
"""


class Task(db.Model):
    __tablename__ = 'task'
    tid = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.Integer, db.ForeignKey('user.uid'))  # 所属用户ID
    kind = db.Column(db.Integer, default=1)  # 任务类型，1表示图像分析，2表示视频分析
    state = db.Column(db.Integer, default=1)  # 任务状态，0表示运行中，1表示已完成，2表示已失败
    taskname = db.Column(db.String(128))  # 任务名
    filepath = db.Column(db.String(128))  # 文件或文件夹路径
    dnn = db.Column(db.Integer, default=1)  # 执行任务的模型，1表示ssd，2表示faster_rcnn
    pl = db.Column(db.Integer, default=0)  # 模型划分点，0-4，0为不执行模型划分
    createdtime = db.Column(db.DateTime(), default=datetime.now)  # 任务启动时间
    endtime = db.Column(db.DateTime())  # 任务结束时间


class Log(db.Model):
    __tablename__ = 'logs'
    id = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey('user.uid'))  # 操作涉及的用户
    content = db.Column(db.Text(), nullable=False)  # 操作的主要内容
    type_flag = db.Column(db.Integer, default=0)  # type_flag 用于记录操作类型:
    # 0代表普通操作如登录等 1代表创建任务 2代表删除任务
    created_time = db.Column(db.DateTime(), default=datetime.now)  # 记录日志时的时间
