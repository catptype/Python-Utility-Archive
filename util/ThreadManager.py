import threading
import time
import sys
import queue
from typing import Any, Callable, List, Optional

class Task:
    """Thread-safe task container."""
    
    def __init__(self, func: Callable, args: tuple, kwargs: dict):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._result = None
        self._error = None
        self._lock = threading.Lock()
        self._done = threading.Event()
    
    def execute(self):
        """Execute the task and store result."""
        try:
            result = self.func(*self.args, **self.kwargs)
            self.set_result(result)
        except Exception as e:
            self.set_error(e)
    
    def set_result(self, value: Any):
        with self._lock:
            self._result = value
            self._done.set()
    
    def set_error(self, error: Exception):
        with self._lock:
            self._error = error
            self._result = f"Error: {error}"
            self._done.set()
    
    @property
    def result(self) -> Any:
        """Get result (thread-safe)."""
        with self._lock:
            return self._result
    
    @property
    def error(self) -> Optional[Exception]:
        """Get error if any."""
        with self._lock:
            return self._error
    
    @property
    def is_finished(self) -> bool:
        """Check if task completed."""
        return self._done.is_set()
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for task to finish. Returns True if completed."""
        return self._done.wait(timeout)


class ThreadManager:
    """
    Lightweight thread pool with instant Ctrl+C response.
    Uses daemon threads for clean interruption.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.task_queue = queue.Queue()
        self._tasks = []
        self._tasks_lock = threading.Lock()
        self._workers = []
        
        # Start daemon workers immediately
        for i in range(max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                name=f"Worker-{i}",
                daemon=True  # KEY: Makes Ctrl+C instant
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self, worker_id: int):
        """
        Worker thread loop.
        Uses timeout to allow quick shutdown on Ctrl+C.
        """
        while True:
            try:
                # Short timeout allows quick exit on Ctrl+C
                task = self.task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                task.execute()
            except Exception as e:
                # Catch any unexpected errors
                print(f"Worker {worker_id} unexpected error: {e}", file=sys.stderr)
            finally:
                self.task_queue.task_done()
    
    def submit(self, func: Callable, *args, **kwargs) -> Task:
        """Submit a task for execution."""
        task = Task(func, args, kwargs)
        
        with self._tasks_lock:
            self._tasks.append(task)
        
        self.task_queue.put(task)
        return task
    
    def wait_completion(self, check_interval: float = 0.1) -> List[Any]:
        """
        Wait for all tasks to complete.
        
        Args:
            check_interval: How often to check (shorter = more responsive to Ctrl+C)
        
        Returns:
            List of all results
        """
        while self.task_queue.unfinished_tasks > 0:
            time.sleep(check_interval)  # Short sleep for Ctrl+C responsiveness
        
        with self._tasks_lock:
            return [task.result for task in self._tasks]
    
    def get_results(self) -> List[Any]:
        """Get all results without waiting."""
        with self._tasks_lock:
            return [task.result for task in self._tasks]
    
    def get_completed_count(self) -> int:
        """Get number of completed tasks."""
        with self._tasks_lock:
            return sum(1 for task in self._tasks if task.is_finished)
    
    def get_total_count(self) -> int:
        """Get total number of submitted tasks."""
        with self._tasks_lock:
            return len(self._tasks)
    
    @property
    def active_count(self) -> int:
        """Get number of active worker threads."""
        return sum(1 for w in self._workers if w.is_alive())