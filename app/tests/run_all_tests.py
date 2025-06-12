#!/usr/bin/env python3
"""
Main test runner for all tests in the project
"""
import sys
import os
import subprocess
import asyncio

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_unittest_file(test_file):
    """Run a unittest file and return success status"""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "unittest", f"app.tests.{test_file}", "-v"
        ], cwd=project_root, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False

async def run_async_tests():
    """Run async tests"""
    print(f"\n{'='*60}")
    print("Running Async Tests (Conversation Guide)")
    print('='*60)
    
    try:
        from test_conversation_guide import test_conversation_guide_service, test_cumulative_input
        await test_conversation_guide_service()
        print("\n" + "-"*40)
        await test_cumulative_input()
        return True
    except Exception as e:
        print(f"Error running async tests: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Running All Tests for Toweel Backend")
    print("="*60)
    
    # List of unittest files to run
    unittest_files = [
        "test_retry_mechanism",
        "test_rag_service",
        "test_api_endpoints",
        "test_session_service"
    ]
    
    results = {}
    
    # Run unittest files
    for test_file in unittest_files:
        results[test_file] = run_unittest_file(test_file)
    
    # Run async tests
    print(f"\n{'='*60}")
    print("Running Async Tests")
    print('='*60)
    
    try:
        async_result = asyncio.run(run_async_tests())
        results["async_tests"] = async_result
    except Exception as e:
        print(f"Error running async tests: {e}")
        results["async_tests"] = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 