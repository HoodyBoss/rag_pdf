#!/usr/bin/env python3
"""
Test LightRAG Integration
Test script to verify LightRAG graph reasoning capabilities
"""
import asyncio
import logging
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_lightrag_import():
    """Test if LightRAG can be imported"""
    print("Testing LightRAG import...")
    try:
        import lightrag
        from lightrag import QueryParam
        print("SUCCESS: LightRAG import successful")
        print(f"   Version: {getattr(lightrag, '__version__', 'Unknown')}")
        print(f"   Available: {dir(lightrag)}")
        return True
    except ImportError as e:
        print(f"FAILED: LightRAG import failed: {e}")
        return False

def test_integration_import():
    """Test LightRAG integration import"""
    print("\nTesting LightRAG integration import...")
    try:
        from lightrag_integration import (
            LightRAGManager,
            initialize_lightrag_system,
            query_with_graph_reasoning,
            multi_hop_reasoning,
            get_lightrag_status
        )
        print("SUCCESS: LightRAG integration import successful")
        return True
    except ImportError as e:
        print(f"FAILED: LightRAG integration import failed: {e}")
        return False

def test_rag_system_import():
    """Test main RAG system with LightRAG"""
    print("\nTesting main RAG system with LightRAG...")
    try:
        import rag_pdf
        print("SUCCESS: RAG system import successful")
        print(f"   LightRAG Available: {getattr(rag_pdf, 'LIGHT_RAG_AVAILABLE', False)}")
        return True
    except ImportError as e:
        print(f"FAILED: RAG system import failed: {e}")
        return False

async def test_lightrag_manager():
    """Test LightRAG manager initialization"""
    print("\nTesting LightRAG manager initialization...")
    try:
        from lightrag_integration import LightRAGManager

        # Create manager
        manager = LightRAGManager(working_dir="./test_data/lightrag")
        print("SUCCESS: LightRAG manager created")

        # Get statistics
        stats = manager.get_statistics()
        print(f"   Statistics: {stats}")

        return True

    except Exception as e:
        print(f"FAILED: LightRAG manager test failed: {e}")
        return False

async def test_graph_reasoning():
    """Test graph reasoning capabilities"""
    print("\nTesting graph reasoning...")
    try:
        from lightrag_integration import query_with_graph_reasoning

        # Test query
        test_query = "What are the main components of this system?"
        result = await query_with_graph_reasoning(test_query, mode="naive")

        print("SUCCESS: Graph reasoning query completed")
        print(f"   Query: {test_query}")
        print(f"   Result length: {len(result.get('result', ''))}")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"   Insights: {result.get('graph_insights', {})}")

        return True

    except Exception as e:
        print(f"FAILED: Graph reasoning test failed: {e}")
        return False

def test_system_status():
    """Test system status functions"""
    print("\nTesting system status...")
    try:
        import rag_pdf

        # Check if function exists
        if hasattr(rag_pdf, 'get_lightrag_system_status'):
            status = rag_pdf.get_lightrag_system_status()
            print("SUCCESS: System status retrieved")
            print(f"   Status: {status}")
            return True
        else:
            print("FAILED: get_lightrag_system_status function not found")
            return False

    except Exception as e:
        print(f"FAILED: System status test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("=== LightRAG Integration Test ===\n")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    tests = [
        ("LightRAG Import", test_lightrag_import),
        ("Integration Import", test_integration_import),
        ("RAG System Import", test_rag_system_import),
        ("System Status", test_system_status),
    ]

    async_tests = [
        ("LightRAG Manager", test_lightrag_manager),
        ("Graph Reasoning", test_graph_reasoning),
    ]

    results = []

    # Run synchronous tests
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"FAILED {test_name}: {e}")
            results.append((test_name, False, str(e)))

    # Run asynchronous tests
    for test_name, test_func in async_tests:
        print(f"Running {test_name}...")
        try:
            result = await test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"FAILED {test_name}: {e}")
            results.append((test_name, False, str(e)))

    # Summary
    print("\n=== Test Summary ===")
    passed = 0
    total = len(results)

    for test_name, result, error in results:
        status = "PASSED" if result else "FAILED"
        print(f"{status} {test_name}")
        if error:
            print(f"    Error: {error}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! LightRAG integration is working correctly.")
        return True
    else:
        print("Some tests failed. Check the errors above for details.")
        return False

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())

    # Exit with appropriate code
    sys.exit(0 if success else 1)