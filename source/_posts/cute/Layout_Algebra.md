---
title: cute Algebra
date: 2026-08-15 20:00:00
tags: [cute, Layout]
categories: [Cute 学习笔记]
description: 文章介绍了 cute Layout的代数运算
---

# logic_divide(Tiling)
逻辑除法，本质上就是**自动构造Tile块和Tile的补集，然后将它们嵌套在一起**
* 一维的逻辑除法
一块连续内存，大小为8， Layout_A = (8) : (1)
将其切分成大小为4的小块(Tile)
Layout_B = logical_divide(Layout_A, 4)
会返回一个嵌套结构
Layout_B = ((4), (2)) : ((1), (4))
含义如下：
((块内坐标), (块间坐标)) : ((块内步长), (块间步长))

* 二维矩阵的逻辑除法
![alt text](../../assets/Layout_Algebra/image.png)
