#!/usr/bin/env python3
"""
Railway-optimized version of RAG PDF Application
Fixed for Railway deployment with proper environment handling
"""
import gradio as gr
import sys
import os
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_railway_app():
    """Create Railway-optimized RAG PDF app"""

    # Railway port configuration
    port = int(os.environ.get("PORT", 7860))

    with gr.Blocks(
        title="RAG PDF",
        theme=gr.themes.Soft(),
        css="""
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
        """
    ) as app:

        # Authentication state
        logged_in = gr.State(False)
        user_data = gr.State({})

        # Main container
        with gr.Column() as main_container:

            # Login interface (shown by default)
            with gr.Group(visible=True) as login_group:
                gr.HTML("""
                    <div style="text-align: center; padding: 2rem; max-width: 400px; margin: 50px auto; background: #f8f9fa; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h1>RAG PDF</h1>
                        <h3>เข้าสู่ระบบ</h3>
                        <p>กรุณาเข้าสู่ระบบเพื่อใช้งานระบบ</p>
                        <p><small>Demo: admin / admin123</small></p>
                    </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
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

            # Main interface (hidden by default)
            with gr.Group(visible=False) as main_group:
                # Header
                gr.HTML("""
                    <div style="background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
                               color: white; padding: 1.5rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
                        <h1>RAG PDF - Main Application</h1>
                        <p>ระบบสืบค้นเอกสาร PDF ด้วย AI</p>
                        <p><small>Deployed on Railway</small></p>
                    </div>
                """)

                # User info
                user_info_display = gr.HTML("")

                # Main content
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML("""
                            <div style="padding: 2rem; text-align: center; background: #f8f9fa; border-radius: 8px; margin-bottom: 1rem;">
                                <h2>RAG PDF System</h2>
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
                            lines=4
                        )

                        # Submit button
                        submit_btn = gr.Button("ส่งคำถาม", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        # Answer display
                        answer_output = gr.HTML(
                            value="<div style='padding: 1rem; background: #e9ecef; border-radius: 4px;'>คำตอบจะปรากฏที่นี่...</div>"
                        )

                # Logout button
                logout_btn = gr.Button("ออกจากระบบ", variant="secondary", size="sm")

        # Login function
        def handle_login(username, password):
            try:
                if username == "admin" and password == "admin123":
                    user_info = {
                        "username": "admin",
                        "full_name": "Administrator",
                        "role": "admin"
                    }

                    user_html = f"""
                    <div style="padding: 1rem; background: #d4edda; border-radius: 4px; margin-bottom: 1rem; text-align: center;">
                        <strong>ยินดีต้อนรับ {user_info['full_name']} ({user_info['role']})</strong>
                    </div>
                    """

                    logger.info(f"User {username} logged in successfully")

                    return (
                        user_html,  # user_info_display
                        gr.update(visible=False),  # login_group
                        gr.update(visible=True),   # main_group
                        True,      # logged_in
                        user_info, # user_data
                        "เข้าสู่ระบบสำเร็จ!",  # login_status
                        "", "",     # clear username/password
                    )
                else:
                    logger.warning(f"Failed login attempt for username: {username}")
                    return (
                        "",  # user_info_display
                        gr.update(visible=True),   # login_group
                        gr.update(visible=False),  # main_group
                        False,     # logged_in
                        {},        # user_data
                        "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง<br><small>ทดลอง: admin / admin123</small>",  # login_status
                        username,  # keep username
                        "",        # clear password
                    )
            except Exception as e:
                logger.error(f"Login error: {e}")
                return (
                    "",  # user_info_display
                    gr.update(visible=True),   # login_group
                    gr.update(visible=False),  # main_group
                    False,     # logged_in
                    {},        # user_data
                    f"เกิดข้อผิดพลาด: {str(e)}",  # login_status
                    username,  # keep username
                    password,  # keep password for retry
                )

        # Logout function
        def handle_logout():
            logger.info("User logged out")
            return (
                "",  # user_info_display
                gr.update(visible=True),   # login_group
                gr.update(visible=False),  # main_group
                False,     # logged_in
                {},        # user_data
                "",        # login_status
                "", ""      # clear username/password
            )

        # Handle question function
        def handle_question(question, file):
            try:
                if not question:
                    return "<div style='padding: 1rem; background: #f8d7da; border-radius: 4px;'>กรุณากรอกคำถาม</div>"

                if not file:
                    return "<div style='padding: 1rem; background: #f8d7da; border-radius: 4px;'>กรุณาอัปโหลดไฟล์ PDF ก่อน</div>"

                logger.info(f"Processing question: {question[:50]}...")

                # Simple response for now
                return f"""
                <div style="padding: 1rem; background: #d4edda; border-radius: 4px;">
                    <h4>คำตอบ:</h4>
                    <p>ได้รับคำถาม: "{question}"</p>
                    <p>ไฟล์: {os.path.basename(file) if file else 'ไม่มี'}</p>
                    <p><em>ระบบกำลังประมวลผล... (ฟีเจอร์ AI จะเพิ่มเติมในภายหลัง)</em></p>
                </div>
                """
            except Exception as e:
                logger.error(f"Question processing error: {e}")
                return f"<div style='padding: 1rem; background: #f8d7da; border-radius: 4px;'>เกิดข้อผิดพลาด: {str(e)}</div>"

        # Connect events
        login_btn.click(
            fn=handle_login,
            inputs=[username_input, password_input],
            outputs=[user_info_display, login_group, main_group, logged_in, user_data, login_status, username_input, password_input]
        )

        logout_btn.click(
            fn=handle_logout,
            outputs=[user_info_display, login_group, main_group, logged_in, user_data, login_status, username_input, password_input]
        )

        submit_btn.click(
            fn=handle_question,
            inputs=[question_input, file_upload],
            outputs=[answer_output]
        )

    return app, port

if __name__ == "__main__":
    logger.info("Starting RAG PDF Application on Railway...")
    logger.info("Demo credentials: admin / admin123")

    app, port = create_railway_app()

    logger.info(f"Launching web interface on port {port}...")

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        quiet=False
    )