#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))

# Дополнительные тесты
print("Test 1: x^3 + 3x + 5 ->", poly_integral([5, 3, 0, 1]))
print("Test 2: constant 5 ->", poly_integral([5]))
print("Test 3: 2x^2 + 3x + 1 ->", poly_integral([1, 3, 2]))
print("Test 4: with C=2 ->", poly_integral([5, 3, 0, 1], 2))
print("Test 5: empty list ->", poly_integral([]))
