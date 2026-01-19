import time
from util.ThreadManager import ThreadManager

# --- Usage Example ---

def heavy_math(x: int, y: int) -> int:
    """Simulates heavy computation."""
    print(f" -> Processing Math {x} + {y}")
    time.sleep(2)
    return x + y

def fetch_data(name: str) -> str:
    """Simulates data fetching."""
    print(f" -> Fetching {name}")
    time.sleep(3)
    return f"Data for {name}"

def might_fail(should_fail: bool = False):
    """Task that might raise an exception."""
    if should_fail:
        raise ValueError("Intentional failure!")
    return "Success!"


if __name__ == "__main__":
    print("=" * 60)
    print("  Thread Manager - Press Ctrl+C Anytime to Stop")
    print("=" * 60)
    print()

    # Create manager with 4 worker threads
    manager = ThreadManager(max_workers=4)

    try:
        print("📤 Submitting tasks...\n")
        
        # Submit various tasks
        t1 = manager.submit(heavy_math, 10, 20)
        t2 = manager.submit(fetch_data, "User_A")
        t3 = manager.submit(heavy_math, 50, 50)
        t4 = manager.submit(fetch_data, "User_B")
        t5 = manager.submit(heavy_math, 100, 200)
        t6 = manager.submit(might_fail, should_fail=False)
        t7 = manager.submit(might_fail, should_fail=True)  # This will fail
        
        total = manager.get_total_count()
        print(f"✅ {total} tasks submitted. Waiting for completion...\n")
        
        # Optional: Monitor progress while waiting
        while manager.task_queue.unfinished_tasks > 0:
            completed = manager.get_completed_count()
            print(f"⏳ Progress: {completed}/{total} tasks completed", end='\r')
            time.sleep(0.5)
        
        print(f"\n\n{'=' * 60}")
        print("  ✨ All Tasks Completed!")  
        print("=" * 60)
        
        # Access individual results
        print(f"\n📊 Individual Results:")
        print(f"  Task 1 (10+20):        {t1.result}")
        print(f"  Task 2 (Fetch User_A): {t2.result}")
        print(f"  Task 3 (50+50):        {t3.result}")
        print(f"  Task 4 (Fetch User_B): {t4.result}")
        print(f"  Task 5 (100+200):      {t5.result}")
        print(f"  Task 6 (Success):      {t6.result}")
        print(f"  Task 7 (Fail):         {t7.result}")
        
        # Get all results at once
        all_results = manager.get_results()
        print(f"\n📦 All Results: {all_results}")
        
        # Check for errors
        print(f"\n⚠️  Error Check:")
        print(f"  Task 7 Error: {t7.error}")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("  🛑 Ctrl+C Detected - Exiting Immediately!")
        print("=" * 60)
        
        # Show what was completed before interrupt
        completed = manager.get_completed_count()
        total = manager.get_total_count()
        print(f"\n  Completed: {completed}/{total} tasks")
        print("  Daemon threads will be terminated by Python.\n")
        
        sys.exit(0)