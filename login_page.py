import gradio as gr
import jwt
import json
import time
import logging
from datetime import datetime, timedelta
from auth_models import auth_manager

# Global authentication state
CURRENT_USER = None
AUTH_TOKEN = None

def create_login_interface():
    """Create Gradio login interface"""

    def login_user(username, password):
        """Handle user login"""
        global CURRENT_USER, AUTH_TOKEN

        try:
            # Connect to auth database
            if not auth_manager.client:
                auth_manager.connect()

            # Check rate limit based on username (before authentication)
            # Use IP-based or session-based rate limiting instead of temp user ID
            # For now, we'll implement a simple in-memory rate limit check
            if not auth_manager.check_rate_limit_by_identifier(username, "login_attempt", limit_per_hour=20):
                return "❌ คุณได้พยายามเข้าสู่ระบบมากเกินไป กรุณารอสักครู่"

            # Authenticate user
            user = auth_manager.authenticate_user(username, password)

            if not user:
                return "❌ ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"

            # Generate tokens
            tokens = auth_manager.generate_tokens(user["user_id"])

            if not tokens:
                return "❌ เกิดการสร้าง token ผิดพลาด"

            # Create session
            session_id = auth_manager.create_session(user["user_id"], tokens, "unknown")

            if not session_id:
                return "❌ เกิดการสร้าง session ผิดพลาด"

            # Update global state
            CURRENT_USER = user
            AUTH_TOKEN = tokens["access_token"]

            # Update user usage
            auth_manager.update_user_usage(user["user_id"], "login", {
                "ip_address": "unknown",
                "user_agent": "gradio_client"
            })

            # Log activity for security
            auth_manager.log_activity(user["user_id"], "login", {
                "ip_address": "unknown",
                "user_agent": "gradio_client",
                "success": True
            })

            # Return success message with token for storage
            return {
                "success": True,
                "message": f"✅ เข้าสู่ระบบสำเร็จ {user['profile']['full_name'] or user['username']}",
                "user": {
                    "username": user["username"],
                    "role": user["role"],
                    "full_name": user["profile"].get("full_name", "")
                },
                "token": AUTH_TOKEN
            }

        except Exception as e:
            logging.error(f"❌ Login error: {e}")
            return f"❌ เกิดการเข้าสู่ระบบ: {str(e)}"

    def check_session():
        """Check if user has active session"""
        global CURRENT_USER, AUTH_TOKEN

        try:
            if AUTH_TOKEN and CURRENT_USER:
                # Validate token
                payload = auth_manager.validate_token(AUTH_TOKEN)
                if payload:
                    return {
                        "logged_in": True,
                        "user": CURRENT_USER
                    }
                else:
                    # Token invalid, logout
                    CURRENT_USER = None
                    AUTH_TOKEN = None
                    return {"logged_in": False}

            return {"logged_in": False}

        except Exception as e:
            logging.error(f"❌ Session check error: {e}")
            return {"logged_in": False}

    def logout_user():
        """Handle user logout"""
        global CURRENT_USER, AUTH_TOKEN

        try:
            if AUTH_TOKEN:
                auth_manager.logout_user(AUTH_TOKEN)

            CURRENT_USER = None
            AUTH_TOKEN = None

            return "✅ ออกจากระบบสำเร็จสำเร็จ"

        except Exception as e:
            logging.error(f"❌ Logout error: {e}")
            return "❌ เกิดการออกจากระบบ"

    def get_current_user():
        """Get current authenticated user"""
        global CURRENT_USER
        return CURRENT_USER

    def get_auth_token():
        """Get current auth token"""
        global AUTH_TOKEN
        return AUTH_TOKEN

    def handle_login(username, password):
        result = login_user(username, password)
        if isinstance(result, dict) and result.get("success"):
            # Store user data and token globally
            global CURRENT_USER, AUTH_TOKEN
            CURRENT_USER = result["user"]
            AUTH_TOKEN = result["token"]

            success_html = f"""
                <div style="text-align: center; padding: 2rem; background: #d4edda; border-radius: 8px;">
                    <h2>✅ เข้าสู่ระบบสำเร็จ!</h2>
                    <p>ยินดีต้อนรับ {result['user'].get('profile', {}).get('full_name', result['user']['username'])}</p>
                    <p>กำลังนำท่านไปยังแอปพลิเคชันหลัก...</p>
                    <p><small>หน้าจอจะรีเฟรชอัตโนมัติภายใน 3 วินาที</small></p>
                </div>
            """

            return (
                result["message"],
                gr.update(visible=False),  # Hide login form
                gr.update(value=json.dumps(result["user"])),
                gr.update(value=result["token"]),
                gr.update(visible=True, value=success_html)  # Show success message
            )
        else:
            return (
                result,
                gr.update(visible=True),   # Show login form
                gr.update(value=""),
                gr.update(value=""),
                gr.update(visible=False)   # Hide success message
            )

    # Create and return the login interface
    with gr.Blocks(title="RAG PDF - Login", css="""
        .gradio-container {
            max-width: 400px;
            margin: 0 auto;
        }
        .login-container {
            padding: 2rem;
            border-radius: 8px;
            background: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .login-form {
            margin-bottom: 1rem;
        }
    """) as login_app:

        with gr.Column(elem_classes=["login-container"]) as login_form:
            # Header
            gr.HTML("""
                <div class="login-header">
                    <h1>🤖 RAG PDF</h1>
                    <h3>เข้าสู่ระบบสำหรัญ</h3>
                    <p>กรุณาเข้าสู่ระบบเพื่อใช้งาน RAG PDF</p>
                </div>
            """)

            # Login Form
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

            # Login Button
            login_btn = gr.Button("🔐 เข้าสู่ระบบ", variant="primary", size="lg")

            # Status Display
            login_status = gr.HTML("")

            # Hidden inputs for token storage
            user_data_json = gr.Text(visible=False)
            token_input = gr.Text(visible=False)

            # Success message (will be shown after login)
            success_message = gr.HTML(visible=False)

        # Connect login button
        login_btn.click(
            fn=handle_login,
            inputs=[username_input, password_input],
            outputs=[login_status, login_form, user_data_json, token_input, success_message]
        )

    return login_app

# Export functions for use in main app
def get_current_user_info():
    """Get current user information for main app"""
    global CURRENT_USER, AUTH_TOKEN

    try:
        # Check if we have a token and user
        if AUTH_TOKEN and CURRENT_USER:
            # Validate token
            if auth_manager.client:
                payload = auth_manager.validate_token(AUTH_TOKEN)
                if payload:
                    return {
                        "authenticated": True,
                        "user": CURRENT_USER,
                        "token": AUTH_TOKEN
                    }
            else:
                # Fallback: if auth_manager not connected, check simple session
                return {
                    "authenticated": True,
                    "user": CURRENT_USER,
                    "token": AUTH_TOKEN
                }

        return {
            "authenticated": False,
            "user": None,
            "token": None
        }

    except Exception as e:
        logging.error(f"❌ Error in get_current_user_info: {e}")
        return {
            "authenticated": False,
            "user": None,
            "token": None
        }

def require_auth(func):
    """Decorator to require authentication"""
    def wrapper(*args, **kwargs):
        auth_info = get_current_user_info()
        if not auth_info["authenticated"]:
            return "❌ กรุณาเข้าสู่ระบบก่อนใช้งาน"

        return func(*args, **kwargs)

    return wrapper

def logout_current_user():
    """Logout current user"""
    return logout_user()

if __name__ == "__main__":
    # Create and launch login interface
    login_app = create_login_ui()
    login_app.launch(server_name="0.0.0.0", server_port=7861)