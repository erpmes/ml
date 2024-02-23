#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:50:52 2024

@author: gaffliu
"""

import random

def monty_hall_simulate(switch_doors):
    # 0, 1, 2 分别代表三扇门
    doors = [0, 1, 2]
    # 随机选择一扇门放置汽车
    car_door = random.choice(doors)
    # 参赛者随机选择一扇门
    contestant_choice = random.choice(doors)
    # 主持人会打开一扇既没有汽车也没有被选中的门
    remaining_doors = [door for door in doors if door != contestant_choice and door != car_door]
    door_opened_by_host = random.choice(remaining_doors)
    # 如果参赛者选择换门
    if switch_doors:
        # 参赛者换到剩下的未被打开的门
        remaining_doors = [door for door in doors if door != contestant_choice and door != door_opened_by_host]
        contestant_choice = remaining_doors[0]
    # 返回参赛者是否赢得了汽车
    return contestant_choice == car_door

# 模拟次数
simulations = 10000
# 不换门的情况下赢得汽车的次数
wins_without_switching = sum(monty_hall_simulate(False) for _ in range(simulations))
# 换门的情况下赢得汽车的次数
wins_with_switching = sum(monty_hall_simulate(True) for _ in range(simulations))

print(f"不换门赢得汽车的概率: {wins_without_switching / simulations}")
print(f"换门赢得汽车的概率: {wins_with_switching / simulations}")
