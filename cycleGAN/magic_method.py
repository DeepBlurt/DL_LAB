# -*- coding: utf-8 -*-

"""
--------------------------------------------
File Name:          magic_method
Author:             deepgray
--------------------------------------------
Description:


--------------------------------------------
Date:               18-6-27
Change Activity:

--------------------------------------------
"""


class Entity(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, z):
        self.x = z
        self.z = z
        print(self.x, self.y, self.z)


a = Entity(2, 3)
a(5)
print(a(6) is a(5))
# 竟然是true
