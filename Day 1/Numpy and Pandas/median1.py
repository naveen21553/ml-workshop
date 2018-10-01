import numpy as np
incomes = np.random.normal(27000,15000,10000)
np.mean(incomes)

# We can segment the incomes data into 50 buckets
import matplotlib.pyplot as plt
plt.hist(incomes, 50)
plt.show()

# Computing Median
np.median(incomes)

# Adding Bill Gates
incomes = np.append(incomes, [1000000000])

# Median remains almost same
np.median(incomes)

# Mean changes distinctly
np.mean(incomes)
