#! /usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Blueprint
file = Blueprint('file', __name__)

# Register all the filter.
from . import views
