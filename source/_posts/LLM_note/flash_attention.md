---
title: FlashAttention 从V2到V4
date: 2026-09-14 20:00:00
tags: [FlashAttention, cute]
categories: [源码阅读]
description: 整理FA2到FA4的实现
---

# 0、序
FlashAttention通过分tile计算，减轻了访存压力。FA3在此基础上，结合了Hopper架构的新特性，通过流水线设计，进一步overlap掉了softmax的开销。FA4则是对新一代blackwelll架构的优化。下面从原理出发，一步步梳理flash attention的发展脉络，也从中窥探硬件的发展趋势。

