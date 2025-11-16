#!/usr/bin/env python3
"""
Feedback System Module
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class FeedbackSystem:
    """Feedback management and analysis system"""

    def __init__(self):
        self.feedback_db = {}  # In-memory storage (replace with database later)
        self.stats_cache = {}

    def save_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save user feedback"""
        try:
            feedback_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            feedback_entry = {
                "id": feedback_id,
                "timestamp": datetime.now().isoformat(),
                "user_id": feedback_data.get("user_id", "1"),
                "chat_id": feedback_data.get("chat_id"),
                "question": feedback_data.get("question"),
                "answer": feedback_data.get("answer"),
                "rating": feedback_data.get("rating"),  # 1-5 stars
                "feedback_type": feedback_data.get("feedback_type"),  # "good", "bad", "improvement"
                "feedback_text": feedback_data.get("feedback_text"),
                "ai_provider": feedback_data.get("ai_provider"),
                "model": feedback_data.get("model"),
                "sources_quality": feedback_data.get("sources_quality"),
                "response_quality": feedback_data.get("response_quality")
            }

            self.feedback_db[feedback_id] = feedback_entry

            return {
                "success": True,
                "feedback_id": feedback_id,
                "message": "✅ บันทึกความคิดเห็นสำเร็จ"
            }

        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return {
                "success": False,
                "message": f"❌ บันทึกความคิดเห็นล้มเหลว: {str(e)}"
            }

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get comprehensive feedback statistics"""
        try:
            if not self.feedback_db:
                return self._get_empty_stats()

            total_feedback = len(self.feedback_db)
            ratings = [f["rating"] for f in self.feedback_db.values() if f.get("rating")]

            # Rating distribution
            rating_dist = {i: 0 for i in range(1, 6)}
            for rating in ratings:
                if 1 <= rating <= 5:
                    rating_dist[rating] += 1

            # AI Provider stats
            provider_stats = {}
            for feedback in self.feedback_db.values():
                provider = feedback.get("ai_provider", "unknown")
                if provider not in provider_stats:
                    provider_stats[provider] = {"count": 0, "avg_rating": 0}
                provider_stats[provider]["count"] += 1

            # Model stats
            model_stats = {}
            for feedback in self.feedback_db.values():
                model = feedback.get("model", "unknown")
                if model not in model_stats:
                    model_stats[model] = {"count": 0, "avg_rating": 0}
                model_stats[model]["count"] += 1

            # Feedback type distribution
            feedback_types = {}
            for feedback in self.feedback_db.values():
                ftype = feedback.get("feedback_type", "unknown")
                feedback_types[ftype] = feedback_types.get(ftype, 0) + 1

            # Calculate averages
            avg_rating = sum(ratings) / len(ratings) if ratings else 0

            # Daily feedback trends (last 7 days)
            daily_trends = self._get_daily_trends()

            return {
                "success": True,
                "total_feedback": total_feedback,
                "average_rating": round(avg_rating, 2),
                "rating_distribution": rating_dist,
                "provider_stats": provider_stats,
                "model_stats": model_stats,
                "feedback_types": feedback_types,
                "daily_trends": daily_trends,
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {
                "success": False,
                "message": f"❌ ดึงสถิติล้มเหลว: {str(e)}"
            }

    def _get_daily_trends(self) -> Dict[str, int]:
        """Get daily feedback trends for last 7 days"""
        from datetime import timedelta

        trends = {}
        today = datetime.now().date()

        for i in range(7):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            count = sum(1 for f in self.feedback_db.values()
                       if f.get("timestamp", "").startswith(date_str))
            trends[date_str] = count

        return dict(sorted(trends.items()))

    def _get_empty_stats(self) -> Dict[str, Any]:
        """Get empty stats structure"""
        return {
            "success": True,
            "total_feedback": 0,
            "average_rating": 0,
            "rating_distribution": {i: 0 for i in range(1, 6)},
            "provider_stats": {},
            "model_stats": {},
            "feedback_types": {},
            "daily_trends": {},
            "last_updated": datetime.now().isoformat()
        }

    def get_recent_feedback(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent feedback entries"""
        try:
            feedback_list = list(self.feedback_db.values())
            # Sort by timestamp descending
            feedback_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            recent_feedback = feedback_list[:limit]

            return {
                "success": True,
                "feedback": recent_feedback,
                "total": len(recent_feedback)
            }

        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return {
                "success": False,
                "message": f"❌ ดึงความคิดเห็นล่าสุดล้มเหลว: {str(e)}"
            }

    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze feedback patterns for insights"""
        try:
            if not self.feedback_db:
                return {"success": True, "insights": []}

            insights = []

            # Low rating patterns
            low_ratings = [f for f in self.feedback_db.values() if f.get("rating", 0) <= 2]
            if low_ratings:
                insights.append({
                    "type": "low_ratings",
                    "message": f"พบคะแนนต่ำ {len(low_ratings)} รายการ - ควรตรวจสอบคุณภาพคำตอบ",
                    "count": len(low_ratings),
                    "severity": "high" if len(low_ratings) > 5 else "medium"
                })

            # Provider performance
            provider_performance = {}
            for feedback in self.feedback_db.values():
                provider = feedback.get("ai_provider", "unknown")
                rating = feedback.get("rating")
                if rating:
                    if provider not in provider_performance:
                        provider_performance[provider] = []
                    provider_performance[provider].append(rating)

            for provider, ratings in provider_performance.items():
                avg = sum(ratings) / len(ratings)
                if avg < 3.0:
                    insights.append({
                        "type": "provider_performance",
                        "message": f"{provider} มีคะแนนเฉลี่ย {avg:.1f} - ควรปรับปรุงประสิทธิภาพ",
                        "provider": provider,
                        "avg_rating": avg,
                        "severity": "medium"
                    })

            # Common feedback themes
            feedback_texts = [f.get("feedback_text", "") for f in self.feedback_db.values()
                            if f.get("feedback_text")]
            if feedback_texts:
                # Simple keyword analysis
                common_issues = []
                for text in feedback_texts:
                    text_lower = text.lower()
                    if any(word in text_lower for word in ["ช้า", "รอนาน", "timeout"]):
                        common_issues.append("response_time")
                    elif any(word in text_lower for word in ["ผิด", "ไม่ตรง", "error"]):
                        common_issues.append("accuracy")
                    elif any(word in text_lower for word in ["ไม่เข้าใจ", "สับสน", "ซับซ้อน"]):
                        common_issues.append("clarity")

                if common_issues:
                    issue_counts = {issue: common_issues.count(issue) for issue in set(common_issues)}
                    most_common = max(issue_counts.items(), key=lambda x: x[1])
                    insights.append({
                        "type": "common_issues",
                        "message": f"ปัญหาที่พบบ่อย: {most_common[0]} ({most_common[1]} ครั้ง)",
                        "issues": issue_counts,
                        "severity": "medium"
                    })

            return {
                "success": True,
                "insights": insights,
                "total_insights": len(insights),
                "analyzed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {e}")
            return {
                "success": False,
                "message": f"❌ วิเคราะห์ความคิดเห็นล้มเหลว: {str(e)}"
            }

# Global instance
feedback_system = FeedbackSystem()