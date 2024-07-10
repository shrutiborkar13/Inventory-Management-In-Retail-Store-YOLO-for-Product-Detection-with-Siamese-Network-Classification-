import csv
import random

# List of customer names
names = ["John Doe", "Jane Smith", "Michael Johnson", "Emily Brown", "David Williams", "Emma Jones", "Daniel Garcia", "Olivia Martinez", "William Taylor", "Sophia Anderson"]

# List of addresses
addresses = ["123 Main St", "456 Elm St", "789 Oak St", "101 Pine St", "202 Maple St", "303 Cedar St", "404 Walnut St", "505 Birch St", "606 Spruce St", "707 Ash St"]

# List of product names
products = ["Suave", "Dove"]

# Generate fake customer data
customer_data = []
for _ in range(10):
    name = random.choice(names)
    address = random.choice(addresses)
    product = random.choice(products)
    customer_data.append([name, address, product])

# Write customer data to CSV file
with open('customers.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Address', 'Ordered Product'])
    writer.writerows(customer_data)

print("Fake CSV file generated successfully!")
