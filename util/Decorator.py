import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        elapsed_time = int((end_time - start_time) * 1000)  # Calculate the elapsed time ms
        return elapsed_time, result  # Return the time and the result
    return wrapper