#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
    本文件是项目本身的构造文件
    主要包括创建 flask app 的工厂函数
    配置 Flask 扩展插件时往往在工厂函数中对 app 进行相关的初始化。
"""

from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import config
from flask_mail import Mail

import os
import logging
import logging.handlers


db = SQLAlchemy()
login_manager = LoginManager()
login_manager.session_protection = 'strong'  # 设置session保护级别
login_manager.login_view = 'user.login'     # 设置登录视图
mail = Mail()


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # 跨域
    CORS(app)

    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)

    # 日志设置
    app.logger.setLevel(logging.INFO)
    info_log = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../', 'deepnex.log')
    info_file_handler = logging.handlers.RotatingFileHandler(
        info_log, maxBytes=1048576, backupCount=20)
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s: %(message)s '
                          '[in %(pathname)s:%(lineno)d]')
    )
    app.logger.addHandler(info_file_handler)

    # 注册路由
    from .test import test_blueprint
    app.register_blueprint(test_blueprint, url_prefix='/test')

    return app
