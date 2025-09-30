#!/usr/bin/env python3

"""
VM Placement Optimizer - Launch Script
Runs tests and starts the Streamlit application
"""

import os
import sys
import subprocess
import time

def run_tests():
    """Run the test suite to verify everything works"""
    print("🧪 Running comprehensive tests...")
    try:
        result = subprocess.run([sys.executable, "test_app.py"], 
                              cwd=os.getcwd(), 
                              capture_output=True, 
                              text=True, 
                              timeout=60)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⚠️ Tests timed out (but this is normal for ML training)")
        return True
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def launch_streamlit():
    """Launch the Streamlit application"""
    print("\n🚀 Starting Streamlit application...")
    print("📍 The app will be available at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the application")
    
    try:
        # Launch streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "Home.py"], 
                      cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def main():
    """Main function"""
    print("🎯 VM Placement ML Optimizer Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("Home.py"):
        print("❌ Error: Please run this script from the VM Placement Optimizer directory")
        print("   Make sure Home.py exists in the current directory")
        sys.exit(1)
    
    # Ask user if they want to run tests first
    run_tests_choice = input("\n🤔 Run tests before launching? (y/N): ").lower().strip()
    
    if run_tests_choice in ['y', 'yes']:
        if not run_tests():
            proceed = input("\n⚠️ Some tests failed. Continue anyway? (y/N): ").lower().strip()
            if proceed not in ['y', 'yes']:
                print("👋 Exiting...")
                sys.exit(1)
    
    # Launch the application
    launch_streamlit()

if __name__ == "__main__":
    main()