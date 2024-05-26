import time

class TimeKeeper:
    def __init__(self):
        self.records = {}

    def record(self, class_name, func_name, duration):
        key = (class_name, func_name)
        if key not in self.records:
            self.records[key] = {'total_time': 0, 'calls': 0}
        self.records[key]['total_time'] += duration
        self.records[key]['calls'] += 1

    def summary(self):
        for (class_name, func_name), data in self.records.items():
            avg_time = data['total_time'] / data['calls']
            print(f"{class_name}.{func_name} - Total Time: {data['total_time']:.8f}s, Average Time: {avg_time:.8f}s, Calls: {data['calls']}")

time_keeper = TimeKeeper()

def time_function(func):
    """Decorator to measure execution time of a method, conditional on an instance flag."""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        duration = time.time() - start_time
        time_keeper.record(self.__class__.__name__, func.__name__, duration)
        return result
    return wrapper