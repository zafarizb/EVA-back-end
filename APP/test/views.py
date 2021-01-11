#! /usr/bin/env python
# -*- coding: utf-8 -*-
from . import test_blueprint
from flask import jsonify


@test_blueprint.route('/ping', methods=['GET'])
def ping():
    """测试 API 是否通"""
    return jsonify('Test API pass!')
