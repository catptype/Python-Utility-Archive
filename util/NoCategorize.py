def get_epsilon():
    epsilon = 1
    while 1 + epsilon != 1:
        epsilon /= 2.
    
    return epsilon