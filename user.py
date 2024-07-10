from uagents import Agent, Context
from uagents.setup import fund_agent_if_low
from protocols import product_proto
from models import ProductQuery, CustomerInfo, ProductAvailabilityResponse,ProductShortage
import csv
def append_row_to_csv(csv_file, row):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

RETAIL_STORE_ADDRESS = "agent1qt2fd39t6jq26pdkgjlguphnufglasfgt3vdquccqnjh3el98n8esgkj7hz"
customer_dict = {}
CUSTOMER_CSV_FILE = "customers.csv"

with open(CUSTOMER_CSV_FILE, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        for row in reader:
            name, address, product = row
            if name in customer_dict:
                customer_dict[name].append((address, product))
            else:
                customer_dict[name] = [(address, product)]
user = Agent(
    name="user",
    port=8000,
    seed="user_secret_phrase",
    endpoint=["http://127.0.0.1:8000/submit"],
)

fund_agent_if_low(user.wallet.address())

user.include(product_proto)

@user.on_interval(period=15.0, messages=ProductQuery)
async def send_product_query(ctx: Context):
    product_name = input("Enter the product name to check availability: ")
    query = ProductQuery(product_name=product_name)
    await ctx.send(RETAIL_STORE_ADDRESS, query)

@user.on_message(model=ProductAvailabilityResponse)
async def handle_product_availability_response(ctx: Context, _sender: str, msg: ProductAvailabilityResponse):
    if msg.available:
        ctx.logger.info(f"Product is available. Compartment Number: {msg.compartment_number}, Compartment Image Path: {msg.compartment_image_path}")
        quantity=int(input("No.of Units:"))
        delivery_choice = input("Do you want it to be delivered? (yes/no): ").lower()
        if delivery_choice == "yes":
            name = input("Please enter your name: ")
            if name in customer_dict:
                for a,b in customer_dict.items():
                    if(name==a):
                        address=b[0][0]
                        ctx.logger.info(f"Welcome back {a} we have your address details!.Please continue")
                    else:
                        continue
            else:
                address = input("Please enter your address: ")
            csv_file = 'customers.csv'
            new_row = [name, address, msg.product_name]
            append_row_to_csv(csv_file, new_row)
            customer_info = CustomerInfo(name=name, address=address)
            amount=quantity*msg.price
            ctx.logger.info(f"Your total bill: Rs{amount} order will be delivered soon.")
            await ctx.send(RETAIL_STORE_ADDRESS, customer_info)
        else:
            ctx.logger.info("Thank you! Please locate the product and pay at the counter.")
            amount=quantity*msg.price
            ctx.logger.info(f"Your total bill: Rs{amount}.")
    else:
        ctx.logger.info("Product is not available.Add your details here,we'll inform once the products are restocked.Sorry for your inconvenience.")
        name = input("Please enter your name: ")
        address = input("Please enter your address: ")
        email_id = input("Please enter your email_id: ")
        product_shortage_request=ProductShortage(name=name,address=address,email_id=email_id,product_name=msg.product_name)
        await ctx.send(RETAIL_STORE_ADDRESS, product_shortage_request)

if __name__ == "__main__":
    user.run()
