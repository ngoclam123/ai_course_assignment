#!/usr/bin/env python3
"""
Test script to check Pinecone status, indexes, and all vectors in the coolmate-index
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🔍 {title}")
    print("="*80)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n📋 {title}")
    print("-"*60)

def check_environment():
    """Check if all required environment variables are set"""
    print_section("Environment Variables Check")
    
    required_vars = [
        "PINECONE_API_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "OPENAI_EMBEDDING_TEST_KEY"
    ]
    
    load_dotenv("../.env")
    
    all_good = True
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"   ✅ {var}: {value[:20]}...")
        else:
            print(f"   ❌ {var}: Not found")
            all_good = False
    
    return all_good

def initialize_pinecone():
    """Initialize Pinecone client"""
    print_section("Pinecone Client Initialization")
    
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        print("   ✅ Pinecone client initialized successfully")
        return pc
    except Exception as e:
        print(f"   ❌ Failed to initialize Pinecone: {str(e)}")
        return None

def list_all_indexes(pc):
    """List all Pinecone indexes"""
    print_section("All Pinecone Indexes")
    
    try:
        indexes = pc.list_indexes()
        
        if not indexes:
            print("   📝 No indexes found")
            return []
        
        print(f"   📊 Found {len(indexes)} index(es):")
        
        index_names = []
        for i, index_info in enumerate(indexes, 1):
            name = index_info["name"]
            index_names.append(name)
            print(f"   {i}. Index Name: {name}")
            
            # Get additional info if available
            if hasattr(index_info, 'get'):
                spec = index_info.get('spec', {})
                if spec:
                    print(f"      - Cloud: {spec.get('serverless', {}).get('cloud', 'N/A')}")
                    print(f"      - Region: {spec.get('serverless', {}).get('region', 'N/A')}")
                print(f"      - Dimension: {index_info.get('dimension', 'N/A')}")
                print(f"      - Metric: {index_info.get('metric', 'N/A')}")
                print(f"      - Status: {index_info.get('status', {}).get('ready', 'Unknown')}")
        
        return index_names
    except Exception as e:
        print(f"   ❌ Error listing indexes: {str(e)}")
        return []

def get_index_stats(pc, index_name):
    """Get detailed statistics for a specific index"""
    print_section(f"Index Statistics for '{index_name}'")
    
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        print(f"   📊 Total Vector Count: {stats.get('total_vector_count', 0):,}")
        print(f"   💾 Index Fullness: {stats.get('index_fullness', 0):.2%}")
        print(f"   📏 Dimension: {stats.get('dimension', 'N/A')}")
        
        # Namespace stats
        namespaces = stats.get('namespaces', {})
        if namespaces:
            print(f"   📁 Namespaces:")
            for ns_name, ns_stats in namespaces.items():
                ns_name_display = ns_name if ns_name else "(default)"
                print(f"      - {ns_name_display}: {ns_stats.get('vector_count', 0):,} vectors")
        else:
            print(f"   📁 Using default namespace with {stats.get('total_vector_count', 0):,} vectors")
        
        return index, stats
    except Exception as e:
        print(f"   ❌ Error getting index stats: {str(e)}")
        return None, None

def list_all_vectors(index, limit=50):
    """List all vectors in the index (with pagination)"""
    print_section("Vector Listing")
    
    try:
        # First, try to get a sample of vectors using query with a random vector
        print(f"   🔍 Fetching up to {limit} vectors...")
        
        # Method 1: Try to list using fetch (if we know some IDs)
        # We'll use query with a dummy vector to get some results
        dummy_vector = [0.1] * 512  # 512-dimensional dummy vector
        
        results = index.query(
            vector=dummy_vector,
            top_k=min(limit, 10000),  # Pinecone query limit
            include_metadata=True
        )
        
        if results.matches:
            print(f"   ✅ Found {len(results.matches)} vectors:")
            print(f"   📋 Vector Details:")
            
            for i, match in enumerate(results.matches, 1):
                print(f"      {i}. ID: {match.id}")
                print(f"         Score: {match.score:.6f}")
                
                if match.metadata:
                    metadata = match.metadata
                    print(f"         Title: {metadata.get('title', 'N/A')}")
                    print(f"         Category: {metadata.get('category', 'N/A')}")
                    print(f"         Price: {metadata.get('price', 'N/A'):,}đ")
                    if metadata.get('discount_percent', 0) > 0:
                        print(f"         Discount: {metadata.get('discount_percent')}%")
                
                if i >= 10:  # Limit detailed display to first 10
                    print(f"      ... and {len(results.matches) - 10} more vectors")
                    break
                    
        else:
            print("   📝 No vectors found (index might be empty)")
            
        return results.matches
        
    except Exception as e:
        print(f"   ❌ Error listing vectors: {str(e)}")
        return []

def search_test(index):
    """Test search functionality with sample queries"""
    print_section("Search Functionality Test")
    
    test_queries = [
        "áo thun cotton nam",
        "quần short thể thao", 
        "áo polo premium"
    ]
    
    # Load embedding client for testing
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_version="2024-07-01-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["OPENAI_EMBEDDING_TEST_KEY"],
        )
        
        for i, query in enumerate(test_queries, 1):
            print(f"   🔍 Test {i}: '{query}'")
            
            try:
                # Get embedding
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=query,
                    dimensions=512
                )
                query_embedding = response.data[0].embedding
                
                # Search
                results = index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True
                )
                
                if results.matches:
                    print(f"      ✅ Found {len(results.matches)} matches:")
                    for j, match in enumerate(results.matches, 1):
                        title = match.metadata.get('title', 'N/A')
                        score = match.score
                        print(f"         {j}. {title} (Score: {score:.4f})")
                else:
                    print(f"      📝 No matches found")
                    
            except Exception as e:
                print(f"      ❌ Search failed: {str(e)}")
    
    except ImportError:
        print("   ⚠️ OpenAI client not available, skipping search test")
    except Exception as e:
        print(f"   ❌ Search test failed: {str(e)}")

def load_product_data():
    """Load and display product data statistics"""
    print_section("Product Data Analysis")
    
    try:
        with open('vietnamese_products.json', 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        print(f"   📊 Total Products Loaded: {len(products)}")
        
        # Category analysis
        categories = {}
        price_range = []
        
        for product in products:
            category = product.get('category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
            if product.get('price'):
                price_range.append(product['price'])
        
        print(f"   📁 Product Categories:")
        for category, count in sorted(categories.items()):
            print(f"      - {category}: {count} products")
        
        if price_range:
            print(f"   💰 Price Range: {min(price_range):,}đ - {max(price_range):,}đ")
            print(f"   💰 Average Price: {sum(price_range)/len(price_range):,.0f}đ")
        
        # Sample products
        print(f"   📋 Sample Products:")
        for i, product in enumerate(products[:5], 1):
            print(f"      {i}. {product.get('title', 'N/A')} - {product.get('price', 0):,}đ ({product.get('category', 'N/A')})")
        
        return products
    except FileNotFoundError:
        print("   ❌ vietnamese_products.json not found")
        return []
    except Exception as e:
        print(f"   ❌ Error loading products: {str(e)}")
        return []

def main():
    """Main function to run all tests"""
    print_header(f"Pinecone Status Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    # 2. Initialize Pinecone
    pc = initialize_pinecone()
    if not pc:
        print("\n❌ Failed to initialize Pinecone. Cannot continue.")
        return
    
    # 3. List all indexes
    index_names = list_all_indexes(pc)
    
    # 4. Focus on coolmate-index if it exists
    target_index = "coolmate-index"
    if target_index in index_names:
        print(f"\n🎯 Focusing on target index: '{target_index}'")
        index, stats = get_index_stats(pc, target_index)
        
        if index and stats:
            # List vectors if index has content
            total_vectors = stats.get('total_vector_count', 0)
            if total_vectors > 0:
                print(f"\n🔍 Index contains {total_vectors:,} vectors")
                vectors = list_all_vectors(index, limit=20)
                
                # Test search functionality
                search_test(index)
            else:
                print(f"\n📝 Index '{target_index}' is empty")
    else:
        print(f"\n⚠️ Target index '{target_index}' not found")
        if index_names:
            print(f"   Available indexes: {', '.join(index_names)}")
    
    # 5. Load and analyze product data
    products = load_product_data()
    
    # 6. Summary
    print_header("Summary")
    print(f"✅ Environment: {'OK' if check_environment() else 'Issues found'}")
    print(f"✅ Pinecone Client: {'Connected' if pc else 'Failed'}")
    print(f"✅ Total Indexes: {len(index_names)}")
    print(f"✅ Target Index: {'Found' if target_index in index_names else 'Not found'}")
    print(f"✅ Product Data: {len(products)} items loaded")
    
    if target_index in index_names:
        try:
            index = pc.Index(target_index)
            stats = index.describe_index_stats()
            vectors_count = stats.get('total_vector_count', 0)
            print(f"✅ Vectors in DB: {vectors_count:,}")
            
            if vectors_count == len(products) and vectors_count > 0:
                print("🎉 Perfect! All products are embedded and stored in Pinecone!")
            elif vectors_count == 0:
                print("⚠️  Index is empty. Run 'Generate & Upload Embeddings' in the app.")
            elif vectors_count != len(products):
                print(f"⚠️  Vector count ({vectors_count}) doesn't match product count ({len(products)})")
        except:
            pass
    
    print("\n" + "="*80)
    print("🏁 Status check completed!")

if __name__ == "__main__":
    main()
