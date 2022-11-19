# -*- coding: utf-8 -*-
# @时间: 2022/11/17 2022/11/17
# @作者： 流星
# @项目：**


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_current)
    return [new_b, new_w]

def gradient_descent_runner(points, starting_b, start_):
    pass
