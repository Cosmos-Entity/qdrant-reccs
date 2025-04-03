import requests
import base64
import re
import logging

logger = logging.getLogger(__name__)

def url_to_base64(url: str) -> str:
    """
    Convert an image URL to a base64 data URL.
    
    Args:
        url: URL of the image
        
    Returns:
        Base64 data URL of the image
    """
    # Handle Cosmos URLs by extracting the actual image URL
    if "cosmos.so/e/" in url:
        # For Cosmos element URLs, we need to get the actual image URL
        response = requests.get(url)
        if response.status_code == 200:
            # Try to find the image URL in the HTML
            match = re.search(r'<meta property="og:image" content="([^"]+)"', response.text)
            if match:
                url = match.group(1)
            else:
                raise ValueError(f"Could not extract image URL from Cosmos page: {url}")
        else:
            raise ValueError(f"Failed to access Cosmos URL: {url}")
    
    # Download the image
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image from URL: {url}")
    
    # Get the content type
    content_type = response.headers.get('Content-Type', 'image/jpeg')
    
    # Convert to base64
    base64_data = base64.b64encode(response.content).decode('utf-8')
    
    # Create data URL
    return f"data:{content_type};base64,{base64_data}"

def download_image(url: str) -> bytes:
    """
    Download an image from a URL.
    
    Args:
        url: URL of the image
        
    Returns:
        Image data as bytes
    """
    # Handle Cosmos URLs by extracting the actual image URL
    if "cosmos.so/e/" in url:
        # For Cosmos element URLs, we need to get the actual image URL
        response = requests.get(url)
        if response.status_code == 200:
            # Try to find the image URL in the HTML
            match = re.search(r'<meta property="og:image" content="([^"]+)"', response.text)
            if match:
                url = match.group(1)
            else:
                raise ValueError(f"Could not extract image URL from Cosmos page: {url}")
        else:
            raise ValueError(f"Failed to access Cosmos URL: {url}")
    
    # Download the image
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image from URL: {url}")
    
    return response.content 