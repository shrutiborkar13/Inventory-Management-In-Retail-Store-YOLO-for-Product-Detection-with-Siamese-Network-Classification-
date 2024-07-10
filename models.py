from uagents import Model

class ProductQuery(Model):
    product_name: str

class ProductAvailabilityResponse(Model):
    product_name:str=None
    available: bool
    compartment_number: int = None
    compartment_image_path: str = None
    price:int = None
    qty:int = None

class CustomerInfo(Model):
    name: str
    address: str
class ProductShortage(Model):
    name: str
    address: str
    email_id:str
    address:str
    product_name:str
    price:int = None

