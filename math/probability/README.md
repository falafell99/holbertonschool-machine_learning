# Probability

This directory contains Python implementations of various probability distributions.

## Files

### poisson.py
Contains the `Poisson` class for working with Poisson distribution.

**Class: Poisson**
- `__init__(self, data=None, lambtha=1.)` - Initialize Poisson distribution
  - If data is provided, calculates lambtha from the data
  - If no data, uses provided lambtha value

**Attributes:**
- `lambtha` - Expected number of occurrences in a given time frame

**Usage:**
```python
from poisson import Poisson

# Create from data
data = [1, 2, 3, 4, 5]
p1 = Poisson(data)
print(p1.lambtha)

# Create with given lambtha
p2 = Poisson(lambtha=5)
print(p2.lambtha)
Requirements

Python 3.5+
pycodestyle 2.5
