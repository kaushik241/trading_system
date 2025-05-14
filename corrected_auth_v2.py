#!/usr/bin/env python
"""
Zerodha Authentication Test Script (v2) - With URL Parsing

This script tests the authentication process with Zerodha API using our custom ZerodhaAuth class.
It extracts the request token from the full redirect URL to minimize input errors.
"""
import os
import sys
import logging
import re
import time
import urllib.parse
from dotenv import load_dotenv, set_key

# Import our ZerodhaAuth class
from auth.zerodha_auth import ZerodhaAuth

load_dotenv(override=True)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_to_env(api_key, api_secret, access_token):
    """Save credentials to .env file."""
    try:
        # Create .env file if it doesn't exist
        if not os.path.exists('.env'):
            with open('.env', 'w') as f:
                pass
                
        # Update environment variables
        set_key('.env', 'KITE_API_KEY', api_key)
        set_key('.env', 'KITE_API_SECRET', api_secret)
        set_key('.env', 'KITE_ACCESS_TOKEN', access_token)
        
        logger.info("Credentials saved to .env file")
        return True
    except Exception as e:
        logger.error(f"Failed to save credentials: {e}")
        return False

def extract_request_token(redirect_url):
    """Extract request token from redirect URL."""
    try:
        # Parse the URL
        parsed_url = urllib.parse.urlparse(redirect_url)
        
        # Get query parameters
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        # Extract request token
        request_token = query_params.get('request_token', [None])[0]
        
        if not request_token:
            # Try alternative method with regex
            match = re.search(r'request_token=([^&]+)', redirect_url)
            if match:
                request_token = match.group(1)
        
        return request_token
    except Exception as e:
        logger.error(f"Error extracting request token: {e}")
        return None

def test_authentication():
    """Test the Zerodha authentication process using ZerodhaAuth class."""
    print("\n" + "="*80)
    print("ZERODHA AUTHENTICATION TEST USING ZerodhaAuth CLASS (v2)")
    print("="*80)
    
    # Step 1: Load environment variables
    load_dotenv()
    
    # Check if API key and secret are available
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    
    if not api_key:
        api_key = input("\nEnter your Zerodha API key: ")
    
    if not api_secret:
        api_secret = input("Enter your Zerodha API secret: ")
    
    if not api_key or not api_secret:
        logger.error("API key and secret are required")
        return False
    
    # Step 2: Initialize ZerodhaAuth
    try:
        logger.debug(f"Initializing ZerodhaAuth with API key: {api_key[:4]}...{api_key[-4:]}")
        auth = ZerodhaAuth(api_key, api_secret)
        logger.info("ZerodhaAuth initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ZerodhaAuth: {e}")
        return False
    
    # Step 3: Generate login URL
    login_url = auth.get_login_url()
    print("\nSTEP 1: Open this URL in your browser to log in:")
    print(login_url)
    print("\nAfter logging in, you will be redirected to your redirect URL.")
    
    # Step 4: Get complete redirect URL from user
    redirect_url = input("\nSTEP 2: Enter the COMPLETE redirect URL after login: ")
    if not redirect_url:
        logger.error("Redirect URL is required")
        return False
    
    # Step 5: Extract request token from redirect URL
    request_token = extract_request_token(redirect_url)
    if not request_token:
        logger.error("Could not extract request token from the URL")
        print("Make sure the URL contains 'request_token=' parameter")
        return False
    
    logger.debug(f"Extracted request token: {request_token}")
    print(f"\nExtracted request token: {request_token}")
    
    # Add a short delay to ensure token processing
    print("Waiting 2 seconds before proceeding...")
    time.sleep(2)
    
    # Step 6: Generate session and get access token
    try:
        print("\nSTEP 3: Generating access token...")
        logger.debug(f"Generating session with request token: {request_token}")
        logger.debug(f"API key: {api_key[:4]}...{api_key[-4:]}, API secret length: {len(api_secret)}")
        
        # Show the checksum calculation details (without revealing full secret)
        secret_preview = f"{api_secret[:2]}...{api_secret[-2:]}" if len(api_secret) > 4 else "***"
        logger.debug(f"Checksum will be calculated from: API key + request token + API secret")
        logger.debug(f"Values: {api_key} + {request_token} + {secret_preview}")
        
        access_token = auth.generate_session(request_token)
        logger.info("Access token generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate access token: {e}")
        logger.error("Please make sure the request token is correct and not expired")
        logger.error("Request tokens are usually valid for only a few minutes")
        return False
    
    # Step 7: Verify connection
    try:
        print("\nSTEP 4: Verifying connection...")
        if auth.validate_connection():
            print("\nConnection verified!")
            # Get the kite connect instance to fetch profile
            kite = auth.get_kite()
            profile = kite.profile()
            print(f"Logged in as: {profile['user_name']}")
            print(f"User ID: {profile['user_id']}")
            print(f"Email: {profile['email']}")
        else:
            logger.error("Connection validation failed")
            return False
    except Exception as e:
        logger.error(f"Failed to verify connection: {e}")
        return False
    
    # Step 8: Save credentials to .env file
    print("\nSTEP 5: Saving credentials to .env file...")
    if not save_to_env(api_key, api_secret, access_token):
        logger.warning("Failed to save credentials to .env file. You may need to update them manually.")
    
    # Final summary
    print("\n" + "="*80)
    print("AUTHENTICATION TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nYour API key, secret, and access token are working correctly.")
    print("ZerodhaAuth class is functioning properly.")
    print("\nIMPORTANT: The access token is valid until the end of the trading day.")
    print("You will need to generate a new access token tomorrow.")
    
    return True

if __name__ == "__main__":
    if not test_authentication():
        sys.exit(1)