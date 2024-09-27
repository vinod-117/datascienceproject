import matplotlib.pyplot as plt
import numpy as np

# Sample data - replace this with your own dataset
ages = np.random.randint(18, 65, size=100)

# Create a histogram
plt.hist(ages, bins=10, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Ages in a Population')

# Show the plot
plt.show()
import matplotlib.pyplot as plt

# Sample data - replace this with your own dataset
genders = ['Male', 'Female', 'Non-Binary', 'Other']
counts = [120, 150, 20, 10]

# Create a bar chart
plt.bar(genders, counts, color=['blue', 'pink', 'purple', 'gray'])

# Add labels and title
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Genders in a Population')

# Show the plot
plt.show()