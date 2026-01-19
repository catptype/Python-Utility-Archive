import sys

def get_machine_epsilon():
    epsilon = 1
    while 1 + epsilon != 1:
        epsilon /= 2.0
    
    epsilon *= 2.0

    print(f"from function call {epsilon}")
    print(f"from system {sys.float_info.epsilon}") 
    return epsilon