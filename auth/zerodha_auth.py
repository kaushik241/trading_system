"""
Authentication module for Zerodha API.
"""
import os
import logging
from typing import Dict, Optional, Union
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

class ZerodhaAuth:
    """Handles authentication with Zerodha's Kite Connect API."""
    
    def __init__(self, api_key: str, api_secret: str, access_token: Optional[str] = None) -> None:
        """
        Initialize the authentication module.
        
        Args:
            api_key: Zerodha API key
            api_secret: Zerodha API secret
            access_token: Optional access token (if already authenticated)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        
        if access_token:
            self.set_access_token(access_token)
    
    def get_login_url(self) -> str:
        """Generate the login URL for Zerodha authentication."""
        return self.kite.login_url()
    
    def generate_session(self, request_token: str) -> str:
        """
        Generate a session using the request token.
        
        Args:
            request_token: Request token obtained after login
            
        Returns:
            Access token for API requests
        """
        data = self.kite.generate_session(request_token, api_secret=self.api_secret)
        access_token = data["access_token"]
        self.set_access_token(access_token)
        logger.info("Successfully authenticated with Zerodha")
        return access_token
    
    def set_access_token(self, access_token: str) -> None:
        """
        Set the access token for API requests.
        
        Args:
            access_token: Access token to use for API requests
        """
        self.access_token = access_token
        self.kite.set_access_token(access_token)
    
    def validate_connection(self) -> bool:
        """
        Validate that the connection is working by fetching user profile.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            profile = self.kite.profile()
            logger.info(f"Connected to Zerodha as: {profile['user_name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Zerodha: {e}")
            return False
    
    def get_kite(self) -> KiteConnect:
        """
        Get the KiteConnect instance.
        
        Returns:
            Configured KiteConnect instance
        """
        return self.kite