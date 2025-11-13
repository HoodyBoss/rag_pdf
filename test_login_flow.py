#!/usr/bin/env python3
"""
Test script for login flow
"""
import gradio as gr
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from login_page import create_login_interface
from authenticated_app import AuthenticatedRAGApp

def test_login_integration():
    """Test the complete login flow"""

    def show_interface():
        # Check if user is already authenticated
        from login_page import get_current_user_info
        auth_info = get_current_user_info()

        if auth_info["authenticated"]:
            # Show main app
            app = AuthenticatedRAGApp()
            main_interface = app.create_main_rag_interface(auth_info["user"])
            return main_interface
        else:
            # Show login
            return create_login_interface()

    return show_interface()

if __name__ == "__main__":
    print("Testing login flow...")

    # Create the interface
    app = test_login_integration()

    # Launch
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True
    )