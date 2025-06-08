#!/usr/bin/env python3
"""
Contoh penggunaan multiple faces per user
"""

import requests
import json

def register_user_with_multiple_faces(api_url, user_name, image_paths):
    """
    Register user dengan multiple faces
    
    Args:
        api_url: URL API server
        user_name: Nama user
        image_paths: List path ke gambar-gambar
    """
    print(f"📝 Registering {user_name} with {len(image_paths)} faces...")
    
    # Prepare files
    files = []
    data = {'name': user_name}
    
    try:
        # Open all image files
        for i, img_path in enumerate(image_paths):
            file_handle = open(img_path, 'rb')
            files.append(('images', (f'image_{i}.jpg', file_handle, 'image/jpeg')))
        
        # Send request
        response = requests.post(f"{api_url}/register-identity", data=data, files=files)
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        # Process response
        result = response.json()
        
        if result['success']:
            print(f"✅ {user_name} registered successfully!")
            print(f"   Processed: {result['processed_images']}/{result['total_images']} images")
        else:
            print(f"❌ Registration failed: {result['message']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

def add_more_faces(api_url, user_name, image_paths):
    """
    Tambah faces untuk user yang sudah ada
    
    Args:
        api_url: URL API server
        user_name: Nama user yang sudah terdaftar
        image_paths: List path ke gambar-gambar baru
    """
    print(f"➕ Adding {len(image_paths)} more faces to {user_name}...")
    
    # Prepare files
    files = []
    data = {'name': user_name}
    
    try:
        # Open all image files
        for i, img_path in enumerate(image_paths):
            file_handle = open(img_path, 'rb')
            files.append(('images', (f'new_image_{i}.jpg', file_handle, 'image/jpeg')))
        
        # Send request
        response = requests.post(f"{api_url}/add-faces", data=data, files=files)
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        # Process response
        result = response.json()
        
        if result['success']:
            print(f"✅ Added faces to {user_name} successfully!")
            print(f"   New faces: {result['new_faces_added']}")
            print(f"   Processed: {result['processed_images']}/{result['total_images']} images")
        else:
            print(f"❌ Adding faces failed: {result['message']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

def verify_face(api_url, image_path):
    """
    Verifikasi wajah
    
    Args:
        api_url: URL API server
        image_path: Path ke gambar untuk verifikasi
    """
    print(f"🔍 Verifying face: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{api_url}/verify-face", files=files)
        
        result = response.json()
        
        if result['success']:
            print(f"✅ Face verified!")
            print(f"   Identity: {result['name']}")
            print(f"   Confidence: {result['data']['confidence']:.1f}%")
            print(f"   Similarity: {result['similarity']:.3f}")
        else:
            print(f"❌ Face not recognized")
            if 'best_match' in result:
                print(f"   Best match: {result['best_match']}")
                print(f"   Confidence: {result['data']['confidence']:.1f}%")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

def list_identities(api_url):
    """List semua identitas yang terdaftar"""
    try:
        response = requests.get(f"{api_url}/identities")
        result = response.json()
        
        if result['success']:
            print(f"📋 Registered identities ({result['count']}):")
            for identity in result['identities']:
                print(f"   - {identity}")
        else:
            print(f"❌ Failed to get identities: {result['message']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Contoh penggunaan"""
    API_URL = "http://localhost:5000"
    
    print("🚀 Multiple Faces per User - Example Usage")
    print("=" * 60)
    
    # Contoh 1: Register user dengan multiple faces
    print("\n1️⃣ REGISTER USER WITH MULTIPLE FACES")
    print("-" * 40)
    
    # Ganti dengan path gambar yang sebenarnya
    john_images = [
        "dataset/John_Doe/john_doe_0001.jpg",
        "dataset/John_Doe/john_doe_0002.jpg", 
        "dataset/John_Doe/john_doe_0003.jpg",
        "dataset/John_Doe/john_doe_0004.jpg"
    ]
    
    # register_user_with_multiple_faces(API_URL, "John_Doe", john_images)
    print("# Uncomment line above and provide real image paths")
    
    # Contoh 2: List identitas
    print("\n2️⃣ LIST REGISTERED IDENTITIES")
    print("-" * 40)
    
    list_identities(API_URL)
    
    # Contoh 3: Tambah faces untuk user existing
    print("\n3️⃣ ADD MORE FACES TO EXISTING USER")
    print("-" * 40)
    
    additional_john_images = [
        "dataset/John_Doe/john_doe_0005.jpg",
        "dataset/John_Doe/john_doe_0006.jpg"
    ]
    
    # add_more_faces(API_URL, "John_Doe", additional_john_images)
    print("# Uncomment line above and provide real image paths")
    
    # Contoh 4: Verifikasi
    print("\n4️⃣ VERIFY FACE")
    print("-" * 40)
    
    # verify_face(API_URL, "test_image.jpg")
    print("# Uncomment line above and provide real image path")
    
    print("\n" + "=" * 60)
    print("📋 API ENDPOINTS SUMMARY:")
    print("=" * 60)
    print("POST /register-identity  - Register dengan multiple faces")
    print("POST /add-faces          - Tambah faces ke user existing")
    print("POST /verify-face        - Verifikasi wajah")
    print("GET  /identities         - List semua identitas")
    print("DELETE /remove-identity  - Hapus identitas")

if __name__ == "__main__":
    main()