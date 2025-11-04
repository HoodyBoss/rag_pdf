#!/usr/bin/env python3
"""
Simple RAG PDF launcher for testing LightRAG integration
"""
import gradio as gr
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting Simple RAG PDF...")

try:
    # Test imports
    import rag_pdf
    print("SUCCESS: RAG PDF imported successfully")
    print(f"LightRAG Available: {rag_pdf.LIGHT_RAG_AVAILABLE}")

    # Skip status function due to emoji encoding issues
    print("Skipping status display due to console encoding limitations")

    # Simple interface for testing
    with gr.Blocks(title="RAG PDF with LightRAG") as demo:
        gr.Markdown("# 🚀 RAG PDF with LightRAG Graph Reasoning")

        with gr.Row():
            gr.Markdown("## 🧠 LightRAG Status")

        with gr.Row():
            status_display = gr.Textbox(
                label="System Status",
                value="Loading...",
                lines=5,
                interactive=False
            )

        with gr.Row():
            test_btn = gr.Button("🧪 Test LightRAG", variant="primary")
            test_output = gr.Textbox(
                label="Test Results",
                lines=10,
                interactive=False
            )

        def run_lightrag_test():
            try:
                result = rag_pdf.test_graph_reasoning_interface()
                return f"Test completed:\n{result[0]}\nVisible: {result[1]}"
            except Exception as e:
                return f"Test failed: {str(e)}"

        test_btn.click(
            fn=run_lightrag_test,
            inputs=[],
            outputs=[test_output]
        )

              refresh_btn = gr.Button("🔄 Refresh Status")
        def safe_update_status():
            try:
                return rag_pdf.update_lightrag_status()
            except:
                return "Status check failed due to encoding issues"

        refresh_btn.click(
            fn=safe_update_status,
            inputs=[],
            outputs=[status_display]
        )

    print("SUCCESS: Interface created successfully")
    print("Starting Gradio server...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()