#!/usr/bin/env python3
"""
Simple test for login interface only
"""
import gradio as gr
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from login_page import create_login_interface

def test_login_only():
    """Test just the login interface"""
    print("Testing login interface...")

    # Create the login interface
    login_app = create_login_interface()

    # Launch the interface
    login_app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    test_login_only()