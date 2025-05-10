#!/usr/bin/env python
"""
Zerodha Authentication Test Script (Using ZerodhaAuth Class)

This script tests the authentication process with Zerodha API using our custom ZerodhaAuth class.
"""
import os
import sys
import logging
from dotenv import load_dotenv, set_key

# Import our ZerodhaAuth class
# Note: Adjust the import path based on your actual project structure
from auth.zerodha_auth import ZerodhaAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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

def test_authentication():
    """Test the Zerodha authentication process using ZerodhaAuth class."""
    print("\n" + "="*80)
    print("ZERODHA AUTHENTICATION TEST USING ZerodhaAuth CLASS")
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
    print("The redirect URL will contain a 'request_token' parameter.")
    print("Example: https://your-redirect-url.com/callback?request_token=xxxxx&action=login&status=success")
    
    # Step 4: Get request token from user
    request_token = input("\nSTEP 2: Enter the request token from the redirect URL: ")
    if not request_token:
        logger.error("Request token is required")
        return False
    
    # Step 5: Generate session and get access token
    try:
        print("\nSTEP 3: Generating access token...")
        access_token = auth.generate_session(request_token)
        logger.info("Access token generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate access token: {e}")
        logger.error("Please make sure the request token is correct and not expired")
        return False
    
    # Step 6: Verify connection
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
    
    # Step 7: Save credentials to .env file
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