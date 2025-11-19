#!/usr/bin/env python3
"""
Integrated RAG PDF Application with Login
Handles login and main interface in a single Gradio session
"""
import gradio as gr
import sys
import os
import json
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auth_models import auth_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_integrated_app():
    """Create integrated app with login and main interface"""

    # Authentication state stored in Gradio session
    with gr.Blocks(title="RAG PDF", theme=gr.themes.Soft(), css="""
        .login-container {
            max-width: 400px;
            margin: 50px auto;
            padding: 2rem;
            border-radius: 8px;
            background: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem;
        }
    """) as app:

        # Session state for authentication
        user_state = gr.State(value=None)
        auth_token = gr.State(value=None)

        # Login interface
        with gr.Column(visible=True, elem_classes=["login-container"]) as login_column:
            gr.HTML("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h1>RAG PDF</h1>
                    <h3>เข้าสู่ระบบ</h3>
                    <p>กรุณาเข้าสู่ระบบเพื่อใช้งานระบบ</p>
                </div>
            """)

            username_input = gr.Text(
                label="ชื่อผู้ใช้",
                placeholder="กรอกชื่อผู้ใช้",
                max_lines=1
            )

            password_input = gr.Text(
                label="รหัสผ่าน",
                type="password",
                placeholder="กรอกรหัสผ่าน",
                max_lines=1
            )

            login_btn = gr.Button("เข้าสู่ระบบ", variant="primary", size="lg")
            login_status = gr.HTML("")

        # Main interface
        with gr.Column(visible=False, elem_classes=["main-container"]) as main_column:
            # Header
            with gr.Row():
                gr.HTML(f"""
                    <div style="background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
                               color: white; padding: 1rem; border-radius: 8px; text-align: center; width: 100%;">
                        <h1>RAG PDF - Main Application</h1>
                    </div>
                """)

            # User info bar
            with gr.Row():
                user_info = gr.HTML("")

            # Main content
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("""
                        <div style="padding: 2rem; text-align: center; background: #f8f9fa; border-radius: 8px; margin-bottom: 1rem;">
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
            logout_btn = gr.Button("ออกจากระบบ", variant="secondary", size="sm")

        # Login function
        def handle_login(username, password):
            try:
                # Simple demo authentication (in production, use proper auth)
                if username == "admin" and password == "admin123":
                    user_data = {
                        "username": "admin",
                        "role": "admin",
                        "profile": {"full_name": "Administrator"}
                    }
                    token = "demo_token_123"

                    return (
                        "เข้าสู่ระบบสำเร็จ!",
                        gr.update(visible=False),  # Hide login column
                        gr.update(visible=True),   # Show main column
                        user_data,
                        token,
                        f"<div style='padding: 0.5rem; background: #d4edda; border-radius: 4px; text-align: right;'>ยินดีต้อนรับ {user_data['profile']['full_name']} ({user_data['role']})</div>"
                    )
                else:
                    # Try actual authentication if available
                    try:
                        if not auth_manager.client:
                            auth_manager.connect()

                        user = auth_manager.authenticate_user(username, password)
                        if user:
                            tokens = auth_manager.generate_tokens(user["user_id"])
                            if tokens:
                                user_data = {
                                    "username": user["username"],
                                    "role": user["role"],
                                    "profile": user.get("profile", {})
                                }
                                token = tokens["access_token"]

                                return (
                                    "เข้าสู่ระบบสำเร็จ!",
                                    gr.update(visible=False),
                                    gr.update(visible=True),
                                    user_data,
                                    token,
                                    f"<div style='padding: 0.5rem; background: #d4edda; border-radius: 4px; text-align: right;'>ยินดีต้อนรับ {user_data['profile'].get('full_name', user_data['username'])} ({user_data['role']})</div>"
                                )
                    except Exception as e:
                        logger.info(f"Auth system not available: {e}")

                    return (
                        "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง<br><small>ทดลอง: admin / admin123</small>",
                        gr.update(visible=True),
                        gr.update(visible=False),
                        None,
                        None,
                        ""
                    )

            except Exception as e:
                return (
                    f"เกิดข้อผิดพลาด: {str(e)}",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    None,
                    None,
                    ""
                )

        # Logout function
        def handle_logout():
            return (
                "",  # Clear user info
                gr.update(visible=True),   # Show login column
                gr.update(visible=False),  # Hide main column
                None,  # Clear user state
                None,  # Clear token
                "",    # Clear login status
                ""     # Clear username/password
            )

        # Handle question function
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

        # Connect events
        login_btn.click(
            fn=handle_login,
            inputs=[username_input, password_input],
            outputs=[login_status, login_column, main_column, user_state, auth_token, user_info]
        )

        logout_btn.click(
            fn=handle_logout,
            inputs=[],
            outputs=[user_info, login_column, main_column, user_state, auth_token, login_status, username_input, password_input]
        )

        submit_btn.click(
            fn=handle_question,
            inputs=[question_input, file_upload],
            outputs=[answer_output]
        )

    return app

if __name__ == "__main__":
    print("Starting Integrated RAG PDF Application...")
    print("Demo credentials: admin / admin123")

    app = create_integrated_app()

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )