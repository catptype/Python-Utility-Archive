import time
import inspect

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        elapsed_time = int((end_time - start_time) * 1000)  # Calculate the elapsed time
        return elapsed_time, result  # Return the time and the result
    return wrapper

def get_func_path():
    """
    Returns the full path of a function in the format: 'module_name.function_name'.
    It is similar to logger `format='[%(module)s.%(funcName)s]'` but this use for `print()`
    ```python
    def foo()
        func_path = get_func_path()
        print(func_path) # Output: '__main__.foo' or 'your_module.foo'
    ```
    """
    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    function_name = caller_frame.f_code.co_name
    module_path = module.__name__ if module else '__main__'
    full_path = f"{module_path}.{function_name}"
    return full_path
