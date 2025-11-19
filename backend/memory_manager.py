#!/usr/bin/env python3
"""
LightRAG Memory Management Module
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class MemoryManager:
    """LightRAG Memory Management System"""

    def __init__(self):
        self.memory_store = {}  # In-memory storage (replace with database later)
        self.session_memories = {}
        self.long_term_memories = {}
        self.working_memories = {}

    def add_memory(self, memory_type: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add memory to specified type"""
        try:
            memory_id = f"{memory_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            memory_entry = {
                "id": memory_id,
                "type": memory_type,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "access_count": 0,
                "last_accessed": datetime.now().isoformat()
            }

            if memory_type == "session":
                self.session_memories[memory_id] = memory_entry
            elif memory_type == "long_term":
                self.long_term_memories[memory_id] = memory_entry
            elif memory_type == "working":
                self.working_memories[memory_id] = memory_entry

            self.memory_store[memory_id] = memory_entry

            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"✅ เพิ่ม {memory_type} memory สำเร็จ"
            }

        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return {
                "success": False,
                "message": f"❌ เพิ่ม memory ล้มเหลว: {str(e)}"
            }

    def get_relevant_memories(self, query: str, memory_types: List[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Get relevant memories based on query"""
        try:
            relevant_memories = []

            # Search through memory types
            memory_stores = {
                "session": self.session_memories,
                "long_term": self.long_term_memories,
                "working": self.working_memories
            }

            if memory_types:
                memory_stores = {k: v for k, v in memory_stores.items() if k in memory_types}

            for memory_type, store in memory_stores.items():
                for memory_id, memory in store.items():
                    # Simple keyword matching for relevance
                    query_lower = query.lower()
                    content_lower = memory["content"].lower()

                    # Calculate simple relevance score
                    relevance_score = 0
                    if query_lower in content_lower:
                        relevance_score += 10
                    # Add more sophisticated matching logic here

                    if relevance_score > 0:
                        memory_copy = memory.copy()
                        memory_copy["relevance_score"] = relevance_score
                        memory_copy["memory_type"] = memory_type
                        relevant_memories.append(memory_copy)

            # Sort by relevance and access frequency
            relevant_memories.sort(
                key=lambda x: (x["relevance_score"], x["access_count"]),
                reverse=True
            )

            # Update access count and last accessed
            for memory in relevant_memories[:limit]:
                memory_id = memory["id"]
                if memory_id in self.memory_store:
                    self.memory_store[memory_id]["access_count"] += 1
                    self.memory_store[memory_id]["last_accessed"] = datetime.now().isoformat()

            return {
                "success": True,
                "memories": relevant_memories[:limit],
                "total_found": len(relevant_memories),
                "query": query
            }

        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return {
                "success": False,
                "message": f"❌ ดึง memories ล้มเหลว: {str(e)}"
            }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            stats = {
                "total_memories": len(self.memory_store),
                "session_memories": len(self.session_memories),
                "long_term_memories": len(self.long_term_memories),
                "working_memories": len(self.working_memories),
                "recent_activities": []
            }

            # Get recent activities
            all_memories = list(self.memory_store.values())
            all_memories.sort(key=lambda x: x["timestamp"], reverse=True)
            stats["recent_activities"] = all_memories[:5]

            # Memory type distribution
            type_counts = {}
            for memory in all_memories:
                mem_type = memory.get("type", "unknown")
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            stats["type_distribution"] = type_counts

            # Most accessed memories
            most_accessed = sorted(all_memories, key=lambda x: x["access_count"], reverse=True)[:5]
            stats["most_accessed"] = most_accessed

            return {
                "success": True,
                "stats": stats,
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {
                "success": False,
                "message": f"❌ ดึงสถิติ memory ล้มเหลว: {str(e)}"
            }

    def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate working memories to long-term"""
        try:
            consolidated_count = 0

            # Move old working memories to long-term
            cutoff_time = datetime.now() - timedelta(hours=24)

            for memory_id, memory in list(self.working_memories.items()):
                memory_time = datetime.fromisoformat(memory["timestamp"])

                if memory_time < cutoff_time:
                    # Move to long-term
                    self.long_term_memories[memory_id] = memory
                    del self.working_memories[memory_id]
                    consolidated_count += 1

            return {
                "success": True,
                "message": f"✅ รวม memories {consolidated_count} รายการเข้า long-term memory",
                "consolidated_count": consolidated_count
            }

        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            return {
                "success": False,
                "message": f"❌ รวม memories ล้มเหลว: {str(e)}"
            }

    def cleanup_old_memories(self, days: int = 30) -> Dict[str, Any]:
        """Clean up old memories"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            deleted_count = 0

            # Clean up each memory type
            for store_name, store in [
                ("session", self.session_memories),
                ("working", self.working_memories)
            ]:
                to_delete = []
                for memory_id, memory in store.items():
                    memory_time = datetime.fromisoformat(memory["timestamp"])
                    if memory_time < cutoff_time:
                        to_delete.append(memory_id)

                for memory_id in to_delete:
                    del store[memory_id]
                    if memory_id in self.memory_store:
                        del self.memory_store[memory_id]
                    deleted_count += 1

            return {
                "success": True,
                "message": f"✅ ลบ memories เก่า {deleted_count} รายการ",
                "deleted_count": deleted_count
            }

        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")
            return {
                "success": False,
                "message": f"❌ ลบ memories ล้มเหลว: {str(e)}"
            }

    def search_memories(self, query: str, memory_types: List[str] = None) -> Dict[str, Any]:
        """Search memories with full-text search"""
        try:
            results = []
            query_lower = query.lower()

            memory_stores = {
                "session": self.session_memories,
                "long_term": self.long_term_memories,
                "working": self.working_memories
            }

            if memory_types:
                memory_stores = {k: v for k, v in memory_stores.items() if k in memory_types}

            for store_name, store in memory_stores.items():
                for memory_id, memory in store.items():
                    content_lower = memory["content"].lower()

                    # Simple text matching
                    if query_lower in content_lower:
                        memory_copy = memory.copy()
                        memory_copy["memory_type"] = store_name

                        # Highlight matches
                        matches = []
                        start = 0
                        while True:
                            pos = content_lower.find(query_lower, start)
                            if pos == -1:
                                break
                            matches.append({
                                "start": pos,
                                "end": pos + len(query),
                                "text": memory["content"][pos:pos + len(query)]
                            })
                            start = pos + 1

                        memory_copy["matches"] = matches
                        results.append(memory_copy)

            return {
                "success": True,
                "results": results,
                "total_results": len(results),
                "query": query
            }

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return {
                "success": False,
                "message": f"❌ ค้นหา memories ล้มเหลว: {str(e)}"
            }

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary for quick overview"""
        try:
            summary = {
                "overview": {
                    "total_memories": len(self.memory_store),
                    "memory_types": {
                        "session": len(self.session_memories),
                        "long_term": len(self.long_term_memories),
                        "working": len(self.working_memories)
                    }
                },
                "activity": {
                    "today_count": 0,
                    "this_week_count": 0,
                    "this_month_count": 0
                },
                "health": {
                    "needs_consolidation": len(self.working_memories) > 10,
                    "needs_cleanup": False
                }
            }

            # Count activities by time period
            now = datetime.now()
            today = now.date()
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)

            for memory in self.memory_store.values():
                memory_time = datetime.fromisoformat(memory["timestamp"])

                if memory_time.date() == today:
                    summary["activity"]["today_count"] += 1
                if memory_time > week_ago:
                    summary["activity"]["this_week_count"] += 1
                if memory_time > month_ago:
                    summary["activity"]["this_month_count"] += 1

            # Check if cleanup is needed
            cutoff_time = now - timedelta(days=30)
            for memory in self.memory_store.values():
                memory_time = datetime.fromisoformat(memory["timestamp"])
                if memory_time < cutoff_time:
                    summary["health"]["needs_cleanup"] = True
                    break

            return {
                "success": True,
                "summary": summary,
                "generated_at": now.isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {
                "success": False,
                "message": f"❌ ดึง memory summary ล้มเหลว: {str(e)}"
            }

# Global instance
memory_manager = MemoryManager()