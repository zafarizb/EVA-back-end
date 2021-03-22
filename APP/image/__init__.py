#! /usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Blueprint
image = Blueprint('image', __name__)

# Register all the filter.
from . import views
