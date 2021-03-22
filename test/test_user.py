#! /usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import requests

class TestUser(unittest.TestCase):
    def setUp(self):
        print('Test begin')

    def test_reg(self):
        data_json = {'username': 'testuser',
                     'email': 'testuser@qq.com',
                     'password': '123456'}
        re = requests.post('http://127.0.0.1:8000/user/Register', json=data_json)
        self.assertEqual(re.status_code, 200)

    def test_login(self):
        data_json = {'username': 'testuser',
                     'password': '123456'}
        re = requests.post('http://127.0.0.1:8000/user/Login', json=data_json)
        self.assertEqual(re.status_code, 200)

    def tearDown(self):
        print('Test over')


if __name__ == '__main__':
    unittest.main()
