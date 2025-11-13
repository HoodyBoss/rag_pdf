#!/usr/bin/env python3
"""
Simple launcher for the integrated RAG PDF app
"""
from integrated_app import create_integrated_app

def main():
    print("=" * 50)
    print("RAG PDF Application")
    print("=" * 50)
    print("Demo credentials: admin / admin123")
    print("Starting application...")
    print()

    app = create_integrated_app()

    print("Launching web interface...")
    print("Access the app at: http://localhost:7860")
    print()

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,  # Open browser automatically
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()