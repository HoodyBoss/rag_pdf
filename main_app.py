#!/usr/bin/env python3
"""
Main application launcher with authentication
"""
import gradio as gr
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from login_page import create_login_interface, login_user, CURRENT_USER, AUTH_TOKEN
from auth_models import auth_manager

def create_main_app():
    """Create the main application with authentication flow"""

    def get_app_interface():
        """Get the appropriate interface based on authentication status"""
        try:
            # Check if user is authenticated
            auth_info = get_current_user_info()

            if auth_info["authenticated"]:
                # Create main RAG interface
                print(f"User {auth_info['user']['username']} authenticated, showing main interface")
                return create_rag_interface(auth_info["user"])
            else:
                # Show login interface
                print("No user authenticated, showing login interface")
                return create_login_interface()

        except Exception as e:
            print(f"Error checking authentication: {e}")
            # If auth fails, show demo interface
            print("Showing demo interface (no authentication required)")
            return create_rag_interface({
                "username": "Demo User",
                "role": "demo",
                "profile": {"full_name": "Demo User"}
            })

    def create_rag_interface(user):
        """Create a simple RAG interface for authenticated users"""
        with gr.Blocks(title="RAG PDF - Main", theme=gr.themes.Soft()) as main_app:

            # Header with user info
            gr.HTML(f"""
                <div style="background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
                           color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
                    <h1>RAG PDF - Main Application</h1>
                    <p>ยินดีต้อนรับ {user.get('profile', {}).get('full_name', user['username'])} ({user['role']})</p>
                </div>
            """)

            # Main content area
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("""
                        <div style="padding: 2rem; text-align: center; background: #f8f9fa; border-radius: 8px;">
                            <h2>🤖 RAG PDF System</h2>
                            <p>ระบบสืบค้นเอกสาร PDF ด้วย AI</p>
                            <p>อัปโหลด PDF และถามคำถามได้ทันที</p>
                        </div>
                    """)

                    # File upload
                    file_upload = gr.File(
                        label="อัปโหลดไฟล์ PDF",
                        file_types=[".pdf"],
                        type="filepath"
                    )

                    # Question input
                    question_input = gr.Textbox(
                        label="ถามคำถาม",
                        placeholder="กรอกคำถามเกี่ยวกับเอกสาร...",
                        lines=3
                    )

                    # Submit button
                    submit_btn = gr.Button("ส่งคำถาม", variant="primary", size="lg")

                with gr.Column(scale=1):
                    # Answer display
                    answer_output = gr.HTML(
                        label="คำตอบ",
                        value="<div style='padding: 1rem; background: #e9ecef; border-radius: 4px;'>คำตอบจะปรากฏที่นี่...</div>"
                    )

            # Footer with logout
            gr.HTML("""
                <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <button onclick="window.location.reload()" style="background: #dc3545; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer;">
                        ออกจากระบบ
                    </button>
                </div>
            """)

            # Handle file upload and question
            def handle_question(question, file):
                if not question:
                    return "<div style='padding: 1rem; background: #f8d7da; border-radius: 4px;'>กรุณากรอกคำถาม</div>"

                if not file:
                    return "<div style='padding: 1rem; background: #f8d7da; border-radius: 4px;'>กรุณาอัปโหลดไฟล์ PDF ก่อน</div>"

                # Simple response for now
                return f"""
                <div style="padding: 1rem; background: #d4edda; border-radius: 4px;">
                    <h4>คำตอบ:</h4>
                    <p>ได้รับคำถาม: "{question}"</p>
                    <p>ไฟล์: {os.path.basename(file) if file else 'ไม่มี'}</p>
                    <p><em>ระบบกำลังประมวลผล... (ฟีเจอร์ AI จะเพิ่มเติมในภายหลัง)</em></p>
                </div>
                """

            submit_btn.click(
                fn=handle_question,
                inputs=[question_input, file_upload],
                outputs=[answer_output]
            )

        return main_app

    return get_app_interface()

if __name__ == "__main__":
    print("Starting RAG PDF Application...")

    # Try to initialize auth manager (but don't fail if it doesn't work)
    try:
        if not auth_manager.client:
            auth_manager.connect()
            auth_manager.create_default_admin()
        print("Authentication system initialized successfully")
    except Exception as e:
        print(f"Warning: Authentication system not available ({e})")
        print("Running in demo mode without authentication")

    # Create and launch the app
    app = create_main_app()

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )