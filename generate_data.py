"""Script to generate 600 realistic messy product descriptions."""
import csv
import random
import os

random.seed(42)

categories = {
    "Mobile": [
        ("Apple", ["iPhone 15", "iPhone 14 Pro", "iPhone 13", "iPhone SE", "iPhone 15 Pro Max"]),
        ("Samsung", ["Galaxy S25 Ultra", "Galaxy S24", "Galaxy A55", "Galaxy M34", "Galaxy Z Fold 6"]),
        ("OnePlus", ["OnePlus 12", "OnePlus Nord CE 4", "OnePlus 11", "OnePlus Nord 3"]),
        ("Xiaomi", ["Redmi Note 13 Pro", "Redmi 13C", "POCO X6 Pro", "Mi 11X"]),
        ("Realme", ["Realme 12 Pro+", "Realme Narzo 70", "Realme GT 6"]),
        ("iQOO", ["iQOO Z9", "iQOO Neo 9 Pro", "iQOO 12"]),
        ("Nothing", ["Nothing Phone 2a", "Nothing Phone 2"]),
        ("Google", ["Pixel 8", "Pixel 8a", "Pixel 7a"]),
    ],
    "Laptop": [
        ("Apple", ["MacBook Air M3", "MacBook Pro 14 M3", "MacBook Pro 16 M3 Max"]),
        ("Dell", ["XPS 15", "Inspiron 15", "Vostro 15", "G15 Gaming"]),
        ("HP", ["Pavilion 15", "Envy 15", "Spectre x360", "Victus 15", "Omen 16"]),
        ("Lenovo", ["IdeaPad Slim 5", "ThinkPad E14", "Legion 5i", "Legion Slim 5", "Yoga 9i"]),
        ("Asus", ["VivoBook 15", "ZenBook 14", "ROG Strix G16", "TUF Gaming F15"]),
        ("Acer", ["Aspire 5", "Nitro 5", "Swift 3", "Predator Helios 16"]),
        ("MSI", ["Modern 14", "Stealth 16", "Cyborg 15"]),
    ],
    "Headphones": [
        ("Sony", ["WH-1000XM5", "WF-1000XM5", "WH-CH720N", "WF-C700N"]),
        ("boAt", ["Rockerz 550", "Airdopes 141", "Bassheads 242", "Nirvana Ion"]),
        ("JBL", ["JBL Tour One M2", "JBL Wave 300TWS", "JBL Tune 760NC"]),
        ("Bose", ["QuietComfort 45", "Bose Sport Earbuds", "SoundLink Flex"]),
        ("Noise", ["Noise Buds VS104", "Noise Shots X5 Pro"]),
        ("Sennheiser", ["Momentum 4 Wireless", "CX Plus TWS"]),
        ("Anker", ["Soundcore Life Q45", "Soundcore Liberty 4"]),
        ("Jabra", ["Jabra Elite 10", "Jabra Evolve2 65"]),
    ],
    "TV": [
        ("Samsung", ["Samsung 55-inch QLED 4K", "Samsung Crystal 4K 43-inch", "Samsung Neo QLED 65-inch"]),
        ("LG", ["LG OLED C3 55-inch", "LG NanoCell 50-inch", "LG 4K Smart TV 43-inch"]),
        ("Sony", ["Sony Bravia XR 65-inch", "Sony X90L 55-inch", "Sony W830K 32-inch"]),
        ("Xiaomi", ["Mi TV 5X 55-inch", "Mi TV 4A Horizon 32-inch", "Xiaomi Smart TV 5A 43-inch"]),
        ("TCL", ["TCL C835 QLED 65-inch", "TCL P635 4K 50-inch"]),
        ("Vu", ["Vu Premium 4K 55-inch", "Vu GloLED 43-inch"]),
    ],
    "Shoes": [
        ("Nike", ["Air Max 270", "Revolution 7", "Court Vision Low", "Pegasus 41"]),
        ("Adidas", ["Ultraboost 23", "Stan Smith", "Samba OG", "NMD R1"]),
        ("Puma", ["Puma Velocity Nitro 3", "Puma Softride Enzo", "Puma Smash v2"]),
        ("Reebok", ["Reebok Floatride Run Fast", "Reebok Classic Leather"]),
        ("ASICS", ["Gel-Kayano 30", "GT-2000 12", "Gel-Nimbus 25"]),
        ("Skechers", ["Skechers Go Walk 7", "Skechers Air Cooled"]),
        ("Campus", ["Campus Alpha", "Campus Luna", "Campus Shooter"]),
        ("Woodland", ["Woodland Leather Boots", "Woodland Casual Sneakers"]),
    ],
    "Fashion": [
        ("Levis", ["501 Original Jeans", "511 Slim Jeans", "Sherpa Trucker Jacket"]),
        ("H&M", ["Slim Fit Cotton Shirt", "Regular Fit T-Shirt", "Oversized Hoodie"]),
        ("Zara", ["Zara Linen Blazer", "Zara Floral Dress", "Zara Wide Leg Trousers"]),
        ("Allen Solly", ["Allen Solly Formal Shirt", "Allen Solly Chinos", "Allen Solly Polo"]),
        ("Nike", ["Dri-FIT T-Shirt", "Tech Fleece Hoodie", "Running Shorts"]),
        ("Puma", ["Puma Graphic Tee", "Puma Track Pants"]),
    ],
    "Kitchen Appliance": [
        ("Philips", ["Philips HL7756 Mixer Grinder", "Philips HD9252 Air Fryer", "Philips HD2718 Toaster"]),
        ("Bajaj", ["Bajaj FX11 Mixer", "Bajaj Majesty RCX 28 Rice Cooker"]),
        ("Prestige", ["Prestige Iris 750W Mixer", "Prestige Svachh Pressure Cooker"]),
        ("Havells", ["Havells Accord 500W Mixer", "Havells Toaster Pop-up 2 Slice"]),
        ("Pigeon", ["Pigeon Healthifry Air Fryer", "Pigeon Joy Rice Cooker"]),
        ("Wonderchef", ["Wonderchef Nutri-blend", "Wonderchef Crimson Tawa"]),
        ("Kent", ["Kent Grand Plus Water Purifier", "Kent Maxx Water Purifier"]),
    ],
    "Gaming": [
        ("Sony", ["PlayStation 5 Console", "PS5 DualSense Controller"]),
        ("Microsoft", ["Xbox Series X", "Xbox Series S", "Xbox Wireless Controller"]),
        ("Razer", ["Razer DeathAdder V3 Mouse", "Razer BlackWidow V4 Keyboard", "Razer Kraken Headset"]),
        ("Corsair", ["Corsair K70 RGB Keyboard", "Corsair HS65 Headset"]),
        ("Logitech", ["Logitech G502 X Mouse", "Logitech G Pro X Keyboard", "Logitech G435 Headset"]),
        ("SteelSeries", ["SteelSeries Arctis Nova 7", "SteelSeries Rival 650"]),
        ("Nintendo", ["Nintendo Switch OLED", "Nintendo Switch Lite"]),
    ],
    "Book": [
        ("Penguin", ["Atomic Habits", "The Psychology of Money", "Ikigai", "The Alchemist", "Sapiens"]),
        ("HarperCollins", ["Rich Dad Poor Dad", "The 48 Laws of Power"]),
        ("Scholastic", ["Harry Potter Box Set", "Percy Jackson Series"]),
        ("Oxford", ["Oxford Advanced Learners Dictionary", "English Grammar in Use"]),
        ("Penguin", ["Think and Grow Rich", "Zero to One", "Deep Work"]),
    ],
    "Furniture": [
        ("Urban Ladder", ["Warner Sofa", "Dime Study Table", "Alto King Bed"]),
        ("IKEA", ["KALLAX Shelf Unit", "MALM Bed Frame", "POANG Armchair", "BILLY Bookcase"]),
        ("Pepperfry", ["Elara 3-Seater Sofa", "Cosmo Bookshelf"]),
        ("Godrej", ["Slimline Wardrobe", "Interio Study Table"]),
        ("Nilkamal", ["Berry Chair", "Plastic Stool", "Folding Table"]),
    ],
}

price_ranges = {
    "Mobile": (8000, 180000),
    "Laptop": (30000, 280000),
    "Headphones": (500, 35000),
    "TV": (15000, 250000),
    "Shoes": (800, 25000),
    "Fashion": (399, 12000),
    "Kitchen Appliance": (800, 25000),
    "Gaming": (3000, 55000),
    "Book": (199, 3500),
    "Furniture": (2000, 150000),
}

bank_offers = [
    "Flat Rs.5000 OFF with HDFC Card",
    "Extra 10% OFF on SBI Credit Card",
    "No Cost EMI on ICICI Card",
    "Flat Rs.2000 Cashback with Axis Bank",
    "5% Unlimited Cashback on Amazon Pay ICICI Card",
    "Flat Rs.3000 OFF with Kotak Card",
    "Get Rs.1500 back with Yes Bank Card",
    "Rs.500 instant discount on IndusInd Card",
]

exchange_offers = [
    "Exchange Bonus Available",
    "Up to Rs.10000 off on exchange",
    "Extra Rs.5000 on exchange of old device",
    "Exchange old phone for flat Rs.8000 off",
]

delivery_opts = [
    "Ships Tomorrow",
    "Free Delivery",
    "Delivery in 2 Days",
    "Delivery in 3-5 Days",
    "Express Delivery Available",
    "Free Installation",
    "Same Day Delivery",
    "Ships in 24 Hours",
]

availability_opts = [
    "In Stock",
    "Only 2 Left!",
    "Only 1 Left - Hurry!",
    "Out of Stock",
    "Coming Soon",
    "Notify Me",
    "Limited Stock",
    "Only 3 Left",
    "Available",
    "Pre-order Now",
]

coupon_opts = [
    "Use code SAVE200 for extra Rs.200 off",
    "Apply coupon FIRST500",
    "Use TRYNEW for 15% extra off",
    "Coupon: FESTIVE300",
    "Extra 5% off with code PREPAY",
    "",
    "",
]

emi_opts = [
    "EMI from Rs.999/month",
    "No Cost EMI available",
    "Easy EMI: Rs.1500/month for 12 months",
    "0% EMI on 6 months",
    "",
    "",
]

sellers = [
    "Appario Retail",
    "Cloudtail India",
    "RetailNet",
    "SuperComNet",
    "Cocoblu Retail",
    "TechnoMart India",
    "FastRetail",
    "Prime Deals India",
    "",
    "",
]

currencies = ["Rs.", "Rs ", "INR ", "Rs", "INR", "Rupees"]

noise_lines = [
    "Special Price",
    "Limited Time Offer",
    "Deal of the Day",
    "Today only!",
    "Best Seller",
    "Amazons Choice",
    "Sponsored",
    "Buy 2 Get 1 Free on select items",
    "Free Charger Included",
    "Combo offer available",
    "Check pincode for delivery",
]

storage_variants = {
    "Mobile": ["128GB Black", "256GB Blue", "512GB Midnight", "128GB White", "256GB Graphite"],
    "Laptop": ["8GB RAM 512GB SSD", "16GB RAM 1TB SSD", "i5 12th Gen", "Ryzen 5 7520U"],
    "TV": ["4K Ultra HD", "Full HD Smart TV", "QLED HDR10+"],
}


def messy_price(price):
    cur = random.choice(currencies)
    mrp = price + random.randint(500, 8000)
    choice = random.randint(0, 6)
    if choice == 0:
        return f"{cur}{price:,}"
    elif choice == 1:
        return f"MRP: {cur}{mrp:,} | Sale: {cur}{price:,}"
    elif choice == 2:
        return f"Special Price: {cur}{price:,}"
    elif choice == 3:
        return f"Now {cur}{price:,} (Was {cur}{mrp:,})"
    elif choice == 4:
        return f"{price:,} only"
    elif choice == 5:
        return f"Price: INR {price:,}"
    else:
        return f"{cur} {price:,}"


def random_caps(s):
    modes = [s, s.upper(), s.lower(), s.title()]
    return random.choice(modes)


def make_description(category, brand, product):
    base_price = random.randint(*price_ranges[category])
    lines = []

    name_choice = random.randint(0, 3)
    if name_choice == 0:
        lines.append(f"{random_caps(brand)} {random_caps(product)}")
    elif name_choice == 1:
        lines.append(f"{product} by {brand}")
    elif name_choice == 2:
        suffix = random.choice(["Latest", "New Launch", "Trending", "Best", ""])
        lines.append(f"{brand.upper()} {product} - {suffix}".strip(" -"))
    else:
        lines.append(f"{product} ({brand})")

    if category in storage_variants:
        lines.append(random.choice(storage_variants[category]))

    lines.append(messy_price(base_price))
    lines.append(random.choice(availability_opts))

    offer_pool = []
    if random.random() > 0.4:
        offer_pool.append(random.choice(bank_offers))
    if random.random() > 0.6:
        offer_pool.append(random.choice(exchange_offers))
    if random.random() > 0.5:
        emi = random.choice(emi_opts)
        if emi:
            offer_pool.append(emi)
    if random.random() > 0.6:
        coup = random.choice(coupon_opts)
        if coup:
            offer_pool.append(coup)
    lines.extend(offer_pool)

    if random.random() > 0.3:
        lines.append(random.choice(delivery_opts))

    seller = random.choice(sellers)
    if seller:
        lines.append(f"Sold by: {seller}")

    for _ in range(random.randint(0, 2)):
        lines.append(random.choice(noise_lines))

    body = lines[1:]
    random.shuffle(body)
    all_lines = [lines[0]] + body

    sep = random.choice(["\n", " | ", "\n\n", " * ", "\n"])
    return sep.join(all_lines)


records = []
idx = 1
for category, brand_products in categories.items():
    per_cat = 60
    for _ in range(per_cat):
        brand, products = random.choice(brand_products)
        product = random.choice(products)
        desc = make_description(category, brand, product)
        records.append({"id": idx, "raw_description": desc})
        idx += 1

random.shuffle(records)
for i, r in enumerate(records, 1):
    r["id"] = i

os.makedirs("data", exist_ok=True)
with open("data/messy_products.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "raw_description"])
    writer.writeheader()
    writer.writerows(records)

print(f"Generated {len(records)} records successfully.")
