import csv
from uagents import Agent, Context
from uagents.setup import fund_agent_if_low
from protocols import product_proto
from models import ProductQuery, ProductAvailabilityResponse, CustomerInfo,ProductShortage
import csv
import matplotlib.pyplot as plt

def count_product_sales(csv_file):
    product_sales = {'Suave': 0, 'Dove': 0}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        for row in reader:
            product = row[2]
            product_sales[product] += 1
    return product_sales

def plot_product_sales_pie(product_sales, output_file):
    labels = list(product_sales.keys())
    sizes = list(product_sales.values())
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Product Sales Pie Chart')
    plt.axis('equal')
    plt.savefig(output_file)  # Save the pie chart as an image file
    plt.show()



PRODUCTS_CSV_FILE = "products.csv"

product_info = {}

with open(PRODUCTS_CSV_FILE, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        product_info[row["product_name"]] = {
            "compartment_number": int(row["compartment_number"]),
            "compartment_image_path": row["product_image_path"],
            "Price": int(row["Price"]),
            "Quantity":int(row["Quantity"])
        }


retail_store = Agent(
    name="retail_store",
    port=8001,
    seed="retail_store_secret_phrase",
    endpoint=["http://127.0.0.1:8001/submit"],
)

fund_agent_if_low(retail_store.wallet.address())

retail_store.include(product_proto)
print(retail_store.address)
@retail_store.on_message(ProductQuery, replies=ProductAvailabilityResponse)
async def handle_product_query(ctx: Context, _sender: str, msg: ProductQuery):
    product_name = msg.product_name
    if product_name in product_info:
        for k,v in product_info.items():
            if product_name==k:
                compartment_number = v["compartment_number"]
                compartment_image_path = v["compartment_image_path"]
                price=v["Price"]
                quantity=v["Quantity"]
                ctx.logger.info("Product details:")
                print("Compartment details:",compartment_number)
                print("Compartment address:",compartment_image_path)
        response = ProductAvailabilityResponse(available=True, compartment_number=compartment_number, compartment_image_path=compartment_image_path,product_name=msg.product_name,price=price,qty=quantity)
    else:
        response = ProductAvailabilityResponse(available=False,product_name=msg.product_name)
    await ctx.send(_sender, response)

@retail_store.on_message(model=CustomerInfo)
async def handle_customer_info(ctx: Context, _sender: str, msg: CustomerInfo):
    ctx.logger.info(f"Customer Name: {msg.name}, Address: {msg.address}")
    csv_file = 'customers.csv'
    output_file = 'product_sales_pie_chart.png'
    product_sales = count_product_sales(csv_file)
    plot_product_sales_pie(product_sales, output_file)
    ctx.logger.info(f"Product sales pie chart saved as '{output_file}'")
    # Handle delivery or further actions
@retail_store.on_message(model=ProductShortage)
async def handle_customer_info(ctx: Context, _sender: str, msg: ProductShortage):
    ctx.logger.info(f"Customer Name: {msg.name}, Address: {msg.address},Email-id:{msg.email_id}")
    ctx.logger.info(f"Raising alert to restock product:{msg.product_name}")
    # Handle delivery or further actions

if __name__ == "__main__":
    retail_store.run()
