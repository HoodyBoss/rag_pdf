#!/usr/bin/env python3
"""
Social Media Bots Module - Discord, Facebook, Line
"""
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class SocialBotConfig(BaseModel):
    """Social Bot Configuration"""
    platform: str
    bot_token: str
    webhook_url: Optional[str] = None
    channels: List[str] = []
    is_active: bool = True
    created_at: Optional[str] = None

class SocialMediaBotManager:
    """Social Media Bot Management System"""

    def __init__(self):
        self.bot_configs = {}
        self.bot_instances = {}
        self.message_history = {}

    def add_bot_config(self, platform: str, bot_token: str, webhook_url: str = None, channels: List[str] = None) -> Dict[str, Any]:
        """Add bot configuration"""
        try:
            config = SocialBotConfig(
                platform=platform,
                bot_token=bot_token,
                webhook_url=webhook_url,
                channels=channels or [],
                is_active=True,
                created_at=datetime.now().isoformat()
            )

            self.bot_configs[platform] = config.dict()

            return {
                "success": True,
                "message": f"✅ เพิ่ม {platform} bot configuration สำเร็จ",
                "platform": platform,
                "config": config.dict()
            }

        except Exception as e:
            logger.error(f"Error adding bot config: {e}")
            return {
                "success": False,
                "message": f"❌ เพิ่ม bot configuration ล้มเหลว: {str(e)}"
            }

    def get_bot_configs(self) -> Dict[str, Any]:
        """Get all bot configurations"""
        try:
            return {
                "success": True,
                "configs": self.bot_configs,
                "total_bots": len(self.bot_configs)
            }
        except Exception as e:
            logger.error(f"Error getting bot configs: {e}")
            return {
                "success": False,
                "message": f"❌ ดึง bot configurations ล้มเหลว: {str(e)}"
            }

    def send_message_to_platform(self, platform: str, channel: str, message: str, user_id: str = None) -> Dict[str, Any]:
        """Send message to specific platform and channel"""
        try:
            if platform not in self.bot_configs:
                return {
                    "success": False,
                    "message": f"❌ ไม่พบ configuration สำหรับ {platform}"
                }

            config = self.bot_configs[platform]

            if platform == "discord":
                return self._send_discord_message(config, channel, message, user_id)
            elif platform == "facebook":
                return self._send_facebook_message(config, channel, message, user_id)
            elif platform == "line":
                return self._send_line_message(config, channel, message, user_id)
            else:
                return {
                    "success": False,
                    "message": f"❌ ไม่รองรับ platform: {platform}"
                }

        except Exception as e:
            logger.error(f"Error sending message to {platform}: {e}")
            return {
                "success": False,
                "message": f"❌ ส่งข้อความไปยัง {platform} ล้มเหลว: {str(e)}"
            }

    def _send_discord_message(self, config: Dict, channel: str, message: str, user_id: str = None) -> Dict[str, Any]:
        """Send Discord message"""
        try:
            # Discord webhook implementation
            webhook_url = config.get("webhook_url")
            if not webhook_url:
                return {
                    "success": False,
                    "message": "❌ ไม่พบ Discord webhook URL"
                }

            import requests

            payload = {
                "content": message,
                "username": "RAG PDF Bot",
                "avatar_url": "https://via.placeholder.com/150"
            }

            response = requests.post(webhook_url, json=payload, timeout=10)

            if response.status_code == 204:
                return {
                    "success": True,
                    "message": "✅ ส่งข้อความไป Discord สำเร็จ",
                    "platform": "discord",
                    "channel": channel
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Discord API error: {response.status_code}"
                }

        except Exception as e:
            logger.error(f"Discord message error: {e}")
            return {
                "success": False,
                "message": f"❌ Discord ล้มเหลว: {str(e)}"
            }

    def _send_facebook_message(self, config: Dict, channel: str, message: str, user_id: str = None) -> Dict[str, Any]:
        """Send Facebook message"""
        try:
            # Facebook Graph API implementation
            bot_token = config.get("bot_token")
            if not bot_token:
                return {
                    "success": False,
                    "message": "❌ ไม่พบ Facebook bot token"
                }

            import requests

            # Facebook page messages API
            api_url = f"https://graph.facebook.com/v18.0/me/messages"

            payload = {
                "recipient": {"id": channel},
                "message": {"text": message},
                "messaging_type": "RESPONSE"
            }

            headers = {
                "Authorization": f"Bearer {bot_token}",
                "Content-Type": "application/json"
            }

            response = requests.post(api_url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "✅ ส่งข้อความไป Facebook สำเร็จ",
                    "platform": "facebook",
                    "channel": channel
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Facebook API error: {response.status_code} - {response.text}"
                }

        except Exception as e:
            logger.error(f"Facebook message error: {e}")
            return {
                "success": False,
                "message": f"❌ Facebook ล้มเหลว: {str(e)}"
            }

    def _send_line_message(self, config: Dict, channel: str, message: str, user_id: str = None) -> Dict[str, Any]:
        """Send LINE message"""
        try:
            # LINE Messaging API implementation
            bot_token = config.get("bot_token")
            if not bot_token:
                return {
                    "success": False,
                    "message": "❌ ไม่พบ LINE bot token"
                }

            import requests

            # LINE Messaging API
            api_url = "https://api.line.me/v2/bot/message/push"

            payload = {
                "to": channel,
                "messages": [
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            }

            headers = {
                "Authorization": f"Bearer {bot_token}",
                "Content-Type": "application/json"
            }

            response = requests.post(api_url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "✅ ส่งข้อความไป LINE สำเร็จ",
                    "platform": "line",
                    "channel": channel
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ LINE API error: {response.status_code} - {response.text}"
                }

        except Exception as e:
            logger.error(f"LINE message error: {e}")
            return {
                "success": False,
                "message": f"❌ LINE ล้มเหลว: {str(e)}"
            }

    def get_message_history(self, platform: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get message history"""
        try:
            history = self.message_history

            if platform:
                history = {k: v for k, v in history.items() if k.startswith(platform)}

            # Sort by timestamp and limit
            all_messages = []
            for platform_name, messages in history.items():
                all_messages.extend(messages)

            all_messages.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            limited_messages = all_messages[:limit]

            return {
                "success": True,
                "messages": limited_messages,
                "total": len(limited_messages),
                "platform": platform
            }

        except Exception as e:
            logger.error(f"Error getting message history: {e}")
            return {
                "success": False,
                "message": f"❌ ดึงประวัติข้อความล้มเหลว: {str(e)}"
            }

    def log_message(self, platform: str, channel: str, message: str, direction: str = "outgoing", user_id: str = None) -> Dict[str, Any]:
        """Log message to history"""
        try:
            log_entry = {
                "platform": platform,
                "channel": channel,
                "message": message,
                "direction": direction,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }

            if platform not in self.message_history:
                self.message_history[platform] = []

            self.message_history[platform].append(log_entry)

            # Keep only recent messages (last 1000 per platform)
            if len(self.message_history[platform]) > 1000:
                self.message_history[platform] = self.message_history[platform][-1000:]

            return {
                "success": True,
                "message": "✅ บันทึกข้อความสำเร็จ"
            }

        except Exception as e:
            logger.error(f"Error logging message: {e}")
            return {
                "success": False,
                "message": f"❌ บันทึกข้อความล้มเหลว: {str(e)}"
            }

    def get_platform_stats(self) -> Dict[str, Any]:
        """Get platform statistics"""
        try:
            stats = {}

            for platform in ["discord", "facebook", "line"]:
                platform_messages = self.message_history.get(platform, [])
                stats[platform] = {
                    "total_messages": len(platform_messages),
                    "outgoing_messages": len([m for m in platform_messages if m.get("direction") == "outgoing"]),
                    "incoming_messages": len([m for m in platform_messages if m.get("direction") == "incoming"]),
                    "last_message": platform_messages[-1]["timestamp"] if platform_messages else None,
                    "active_channels": len(set(m["channel"] for m in platform_messages))
                }

            return {
                "success": True,
                "stats": stats,
                "total_platforms": len(stats),
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting platform stats: {e}")
            return {
                "success": False,
                "message": f"❌ ดึงสถิติ platform ล้มเหลว: {str(e)}"
            }

    def test_bot_connection(self, platform: str) -> Dict[str, Any]:
        """Test bot connection"""
        try:
            if platform not in self.bot_configs:
                return {
                    "success": False,
                    "message": f"❌ ไม่พบ configuration สำหรับ {platform}"
                }

            config = self.bot_configs[platform]

            # Send test message
            test_message = f"🤖 Bot test message from RAG PDF System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # Use first available channel for testing
            channels = config.get("channels", [])
            test_channel = channels[0] if channels else "test_channel"

            result = self.send_message_to_platform(platform, test_channel, test_message)

            if result["success"]:
                result["message"] = f"✅ {platform} bot connection successful"
                result["test_details"] = {
                    "platform": platform,
                    "test_channel": test_channel,
                    "test_message": test_message,
                    "timestamp": datetime.now().isoformat()
                }

            return result

        except Exception as e:
            logger.error(f"Error testing bot connection: {e}")
            return {
                "success": False,
                "message": f"❌ ทดสอบการเชื่อมต่อ {platform} ล้มเหลว: {str(e)}"
            }

    def delete_bot_config(self, platform: str) -> Dict[str, Any]:
        """Delete bot configuration"""
        try:
            if platform in self.bot_configs:
                del self.bot_configs[platform]

                # Clear message history for this platform
                if platform in self.message_history:
                    del self.message_history[platform]

                return {
                    "success": True,
                    "message": f"✅ ลบ {platform} bot configuration สำเร็จ"
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ ไม่พบ configuration สำหรับ {platform}"
                }

        except Exception as e:
            logger.error(f"Error deleting bot config: {e}")
            return {
                "success": False,
                "message": f"❌ ลบ bot configuration ล้มเหลว: {str(e)}"
            }

# Global instance
social_bot_manager = SocialMediaBotManager()