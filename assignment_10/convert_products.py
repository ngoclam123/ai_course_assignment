import json
import re

def parse_products_from_text():
    """Parse Vietnamese product text and convert to structured JSON format."""
    
    # Read the product data
    products = []
    product_id = 1
    
    # Sample product text - you can replace this with reading from your untitled file
    product_lines = [
        "Sản phẩm Tshirt chạy bộ AirRush Gradient có giá 199.000đ",
        "Sản phẩm Combo 3 Quần Lót Nam Trunk Excool có giá 289.000đ (Giảm 12% từ 329.000đ)",
        "Sản phẩm Shorts thể thao nam 6 inch đai siêu co giãn có giá 239.000đ",
        "Sản phẩm Quần Dài Nam Kaki Excool có giá 419.000đ (Giảm 16% từ 499.000đ)",
        "Sản phẩm Áo ba lỗ nam mặc trong thoáng khí nhanh khô Excool có giá 99.000đ",
        "Sản phẩm Áo Thun Gym Cotton Oversized Comfort Washed có giá 299.000đ",
        "Sản phẩm Áo thun chạy bộ \"Việt Nam tiến bước\" có giá 239.000đ",
        "Sản phẩm Áo Thun Nam Viet Devils - Bật Như Man có giá 329.000đ (Giảm 15% từ 389.000đ)",
        "Sản phẩm Áo Thun Nam Viet Devils - Ăn Mày Quá Khứ có giá 329.000đ (Giảm 15% từ 389.000đ)",
        "Sản phẩm Áo Thun Nam Viet Devils - Mùa Sau Vô Địch có giá 329.000đ (Giảm 15% từ 389.000đ)",
    ]
    
    # Read all lines from the untitled file if available
    try:
        with open('untitled_products.txt', 'r', encoding='utf-8') as f:
            product_lines = f.readlines()
    except FileNotFoundError:
        print("Using sample product data instead of file")
    
    for line in product_lines:
        line = line.strip()
        if line and line.startswith("Sản phẩm"):
            # Parse product line using regex
            # Pattern: "Sản phẩm [title] có giá [price] (optional discount info)"
            pattern = r"Sản phẩm (.+?) có giá ([\d.,]+)đ(?:\s*\(Giảm\s*(\d+)%\s*từ\s*([\d.,]+)đ\))?"
            match = re.search(pattern, line)
            
            if match:
                title = match.group(1).strip()
                current_price = float(match.group(2).replace(".", "").replace(",", ""))
                discount_percent = match.group(3)
                original_price = match.group(4)
                
                # Determine category based on keywords
                category = "Khác"  # Default category
                title_lower = title.lower()
                
                if any(word in title_lower for word in ["áo thun", "tshirt", "t-shirt"]):
                    category = "Áo thun"
                elif any(word in title_lower for word in ["quần", "shorts", "pants"]):
                    category = "Quần"
                elif any(word in title_lower for word in ["áo polo", "polo"]):
                    category = "Áo polo"
                elif any(word in title_lower for word in ["áo sơ mi", "shirt"]):
                    category = "Áo sơ mi"
                elif any(word in title_lower for word in ["áo khoác", "jacket", "hoodie", "nỉ"]):
                    category = "Áo khoác"
                elif any(word in title_lower for word in ["quần lót", "trunk", "briefs"]):
                    category = "Đồ lót"
                elif any(word in title_lower for word in ["váy", "dress"]):
                    category = "Váy"
                elif any(word in title_lower for word in ["bra", "áo bra"]):
                    category = "Áo bra"
                elif any(word in title_lower for word in ["legging", "tights"]):
                    category = "Legging"
                elif any(word in title_lower for word in ["tất", "socks"]):
                    category = "Tất"
                
                # Create product object
                product = {
                    "id": f"prod_{product_id:04d}",
                    "title": title,
                    "category": category,
                    "price": current_price,
                    "original_price": float(original_price.replace(".", "").replace(",", "")) if original_price else current_price,
                    "discount_percent": int(discount_percent) if discount_percent else 0,
                    "short_description": title,  # Use title as description for now
                    "description": f"Sản phẩm {title} thuộc danh mục {category} với giá {current_price:,.0f}đ"
                }
                
                products.append(product)
                product_id += 1
    
    return products

if __name__ == "__main__":
    products = parse_products_from_text()
    
    # Save to JSON file
    with open('vietnamese_products.json', 'w', encoding='utf-8') as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(products)} products to vietnamese_products.json")
    print("Sample products:")
    for i, product in enumerate(products[:3]):
        print(f"{i+1}. {product['title']} - {product['price']:,.0f}đ ({product['category']})")
