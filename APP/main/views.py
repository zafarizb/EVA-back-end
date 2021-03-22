#! /usr/bin/env python
# -*- coding: utf-8 -*-
from . import main
from ..models import User, db
from flask import jsonify, request


@main.route('/')
def index():
    return jsonify({'status': '1'})
