import gradio as gr
import os
import sys
from datetime import datetime
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pdf import RAGPDFApplication
from login_page import create_login_interface, get_current_user_info, logout_current_user, require_auth
from auth_models import auth_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AuthenticatedRAGApp:
    def __init__(self):
        self.rag_app = RAGPDFApplication()
        self.auth_manager = auth_manager

    def create_authenticated_interface(self):
        """Create the main authenticated application interface"""

        def create_main_interface():
            """Create the main RAG interface with authentication wrapper"""

            # Check authentication status
            auth_info = get_current_user_info()

            if not auth_info["authenticated"]:
                # Show login interface
                return create_login_interface()

            # User is authenticated, show main interface
            user = auth_info["user"]

            with gr.Blocks(
                title="RAG PDF - Authenticated",
                css="""
                    .header {
                        text-align: center;
                        padding: 1rem;
                        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border-radius: 8px;
                        margin-bottom: 1rem;
                    }
                    .user-info {
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        background: rgba(255,255,255,0.9);
                        padding: 0.5rem;
                        border-radius: 4px;
                        font-size: 0.9rem;
                    }
                    .logout-btn {
                        background: #dc3545;
                        color: white;
                        border: none;
                        padding: 0.25rem 0.5rem;
                        border-radius: 4px;
                        cursor: pointer;
                        margin-left: 0.5rem;
                    }
                """
            ) as authenticated_app:

                # User info and logout
                with gr.Row():
                    gr.HTML(f"""
                        <div class="user-info">
                            <strong>👤 {user.get('profile', {}).get('full_name', user['username'])}</strong>
                            <span>({user['role']})</span>
                            <button class="logout-btn" onclick="window.location.reload()">ออกจากระบบ</button>
                        </div>
                    """)

                # Header
                gr.HTML(f"""
                    <div class="header">
                        <h1>🤖 RAG PDF - Authenticated Interface</h1>
                        <p>ยินดีต้อนรับ {user.get('profile', {}).get('full_name', user['username'])}</p>
                        <p>บทบาท: {user['role']} | เข้าสู่ระบบเมื่อ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                """)

                # Create the main RAG interface
                self.rag_app.create_interface()

            return authenticated_app

        def authenticate_and_redirect(username, password):
            """Handle authentication and redirect to main interface"""
            from login_page import login_user

            result = login_user(username, password)

            if isinstance(result, dict) and result.get("success"):
                logger.info(f"✅ User {username} authenticated successfully")
                # Return the main interface
                main_interface = create_main_interface()
                return main_interface
            else:
                logger.warning(f"❌ Authentication failed for user {username}")
                return result

        def check_and_show_interface():
            """Check authentication and show appropriate interface"""
            try:
                # Initialize auth manager if not connected
                if not self.auth_manager.client:
                    self.auth_manager.connect()
                    self.auth_manager.create_default_admin()

                auth_info = get_current_user_info()

                if auth_info["authenticated"]:
                    logger.info(f"✅ User {auth_info['user']['username']} already authenticated")
                    return create_main_interface()
                else:
                    logger.info("ℹ️ No active session, showing login interface")
                    return create_login_interface()

            except Exception as e:
                logger.error(f"❌ Error checking authentication: {e}")
                # Fallback to login interface
                return create_login_interface()

        def handle_logout():
            """Handle user logout"""
            try:
                result = logout_current_user()
                logger.info(f"✅ Logout result: {result}")
                # Return to login interface
                return create_login_interface()
            except Exception as e:
                logger.error(f"❌ Logout error: {e}")
                return f"❌ เกิดข้อผิดพลาดในการออกจากระบบ: {str(e)}"

        return check_and_show_interface()

    def launch(self, server_name="0.0.0.0", server_port=7860, **kwargs):
        """Launch the authenticated application"""

        logger.info("🚀 Starting Authenticated RAG PDF Application...")

        # Initialize authentication
        try:
            if not self.auth_manager.client:
                self.auth_manager.connect()
                self.auth_manager.create_default_admin()
                logger.info("✅ Authentication system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize authentication: {e}")
            raise

        # Create and launch the interface
        app = self.create_authenticated_interface()

        logger.info(f"🌐 Launching on {server_name}:{server_port}")
        app.launch(
            server_name=server_name,
            server_port=server_port,
            **kwargs
        )

def main():
    """Main entry point"""
    try:
        # Create authenticated application
        auth_app = AuthenticatedRAGApp()

        # Launch the application
        auth_app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )

    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()