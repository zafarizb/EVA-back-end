#! /usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Blueprint
test_blueprint = Blueprint('test', __name__)

# Register all the filter.
from . import views
