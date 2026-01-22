#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

poly = [5, 3, 0, 1]
print(poly_derivative(poly))

# Дополнительные тесты
print("Test 1: x^3 + 3x + 5 ->", poly_derivative([5, 3, 0, 1]))
print("Test 2: constant 5 ->", poly_derivative([5]))
print("Test 3: 2x^2 + 3x + 1 ->", poly_derivative([1, 3, 2]))
print("Test 4: empty list ->", poly_derivative([]))
print("Test 5: zero polynomial ->", poly_derivative([0, 0, 0]))
