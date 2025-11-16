#!/usr/bin/env python3
"""
Google Sheets Integration Module
"""
import logging
import re
from typing import Optional, Dict, Any
import pandas as pd
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class GoogleSheetsProcessor:
    """Google Sheets Data Processor"""

    def extract_sheet_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract Sheet ID from Google Sheets URL

        Args:
            url: Google Sheets URL

        Returns:
            Sheet ID or None if not found
        """
        try:
            # Pattern สำหรับ Google Sheets URL
            patterns = [
                r"/d/([a-zA-Z0-9-_]+)",
                r"id=([a-zA-Z0-9-_]+)",
                r"/spreadsheets/d/([a-zA-Z0-9-_]+)"
            ]

            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)

            return None

        except Exception as e:
            logger.error(f"Error extracting sheet ID: {e}")
            return None

    def extract_gid_from_url(self, url: str) -> Optional[str]:
        """
        Extract GID (Sheet ID) from Google Sheets URL

        Args:
            url: Google Sheets URL

        Returns:
            GID (Sheet ID) or None if not found
        """
        try:
            # Pattern สำหรับ GID ใน URL
            patterns = [
                r"gid=([0-9]+)",
                r"#gid=([0-9]+)"
            ]

            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)

            return None

        except Exception as e:
            logger.error(f"Error extracting GID: {e}")
            return None

    def export_as_csv(self, sheet_id: str, gid: str = "0") -> Optional[str]:
        """
        Export Google Sheets as CSV

        Args:
            sheet_id: Google Sheets ID
            gid: Sheet ID (default to first sheet)

        Returns:
            CSV content as string or None if failed
        """
        try:
            export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(export_url, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.text

            logger.error(f"Failed to export sheet: HTTP {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error exporting Google Sheets: {e}")
            return None

    def process_sheet_data(self, csv_content: str, sheet_url: str) -> str:
        """
        Process CSV content from Google Sheets

        Args:
            csv_content: CSV content as string
            sheet_url: Original Google Sheets URL

        Returns:
            Formatted text content
        """
        try:
            # Read CSV content
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))

            # Format data as text
            text_parts = []
            text_parts.append(f"📊 Google Sheets Data: {sheet_url}")
            text_parts.append("=" * 50)

            # Convert DataFrame to readable text
            for index, row in df.iterrows():
                row_data = []
                for col in df.columns:
                    if pd.notna(row[col]):
                        row_data.append(f"{col}: {row[col]}")

                if row_data:
                    text_parts.append(" | ".join(row_data))

            text_parts.append("=" * 50)

            return "\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error processing sheet data: {e}")
            return f"❌ Error processing Google Sheets data: {str(e)}"

    def extract_google_sheets_data(self, sheets_url: str) -> Dict[str, Any]:
        """
        Extract data from Google Sheets URL

        Args:
            sheets_url: Google Sheets URL

        Returns:
            Dictionary with success status and data/content
        """
        try:
            # Validate URL
            if not sheets_url or "docs.google.com" not in sheets_url:
                return {
                    "success": False,
                    "message": "❌ กรุณใส่ Google Sheets URL ที่ถูกต้อง"
                }

            # Extract sheet ID
            sheet_id = self.extract_sheet_id_from_url(sheets_url)
            if not sheet_id:
                return {
                    "success": False,
                    "message": "❌ ไม่สามารถแยก Sheet ID จาก URL ได้"
                }

            # Extract GID (Sheet tab ID) from URL
            gid = self.extract_gid_from_url(sheets_url)
            logger.info(f"Extracted sheet ID: {sheet_id}, GID: {gid}")

            # Export as CSV with correct GID
            csv_content = self.export_as_csv(sheet_id, gid)
            if not csv_content:
                return {
                    "success": False,
                    "message": f"❌ ไม่สามารถ export Google Sheets ได้ (Sheet ID: {sheet_id}, GID: {gid})"
                }

            # Process the data
            processed_content = self.process_sheet_data(csv_content, sheets_url)

            return {
                "success": True,
                "message": "✅ ดึงข้อมูลจาก Google Sheets สำเร็จ",
                "content": processed_content,
                "sheet_id": sheet_id,
                "source_url": sheets_url,
                "rows_count": len(csv_content.split('\n')) - 1 if csv_content else 0
            }

        except Exception as e:
            logger.error(f"Error extracting Google Sheets data: {e}")
            return {
                "success": False,
                "message": f"❌ ไม่สามารถดึงข้อมูลจาก Google Sheets ได้: {str(e)}"
            }

# Global instance
google_sheets_processor = GoogleSheetsProcessor()