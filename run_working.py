#!/usr/bin/env python3
"""
Simple launcher for the working RAG PDF app
"""
from working_app import create_working_app

if __name__ == "__main__":
    print("Starting RAG PDF Application...")
    print("Demo credentials: admin / admin123")
    print()

    app = create_working_app()

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )