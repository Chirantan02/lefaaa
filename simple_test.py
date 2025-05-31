#!/usr/bin/env python3
import requests
import time

print('Testing Fixed Leffa Virtual Try-On API')
print('='*60)

# Test data with the provided URLs
test_data = {
    'user_image_url': 'https://storage.googleapis.com/mask_images/f0518e08_Subliminator%20Printed%20_model.jpeg',
    'garment_image_url': 'https://storage.googleapis.com/mask_images/image_prompt.jpeg',
    'mask_type': 'full',
    'steps': 20,
    'cfg': 2.5,
    'seed': 42
}

api_url = 'https://zebels-main--leffa-simple-viton-generate-tryon-api.modal.run'

print(f'API URL: {api_url}')
print(f'User Image: {test_data["user_image_url"]}')
print(f'Garment Image: {test_data["garment_image_url"]}')
print('='*60)

try:
    print('Sending request...')
    start_time = time.time()
    
    response = requests.post(
        api_url,
        json=test_data,
        timeout=300  # 5 minutes timeout
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f'Request completed in {total_time:.2f} seconds')
    print(f'Response status: {response.status_code}')
    
    if response.status_code == 200:
        result = response.json()
        print('SUCCESS!')
        print('='*60)
        print('RESPONSE DETAILS:')
        print(f'Status: {result.get("status", "Unknown")}')
        print(f'Message: {result.get("message", "No message")}')
        
        if 'result_image_url' in result:
            print(f'Generated Image URL: {result["result_image_url"]}')
        
        print('='*60)
        print('NOTE: Detailed time and cost logs are available in Modal logs')
        print('   Check Modal dashboard for complete execution details')
        print('='*60)
        
    else:
        print(f'ERROR: {response.status_code}')
        print(f'Response: {response.text}')
        print('NOTE: Check Modal logs for detailed error information')
        
except requests.exceptions.Timeout:
    print('Request timed out after 5 minutes')
except requests.exceptions.RequestException as e:
    print(f'Request failed: {e}')
except Exception as e:
    print(f'Unexpected error: {e}')
