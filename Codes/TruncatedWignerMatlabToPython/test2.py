from multiprocessing import Pool, cpu_count
import time

print('randonm')
def t():
    # Make a dummy dictionary
    d = {k: k**2 for k in range(10)}

    pool = Pool(processes=(cpu_count() - 1))
    print('cpu',cpu_count())

    for key, value in d.items():
        pool.apply_async(thread_process, args=(key, value))

    pool.close()
    pool.join()


def thread_process(key, value):
    print(__name__)
    time.sleep(1)  # Simulate a process taking some time to complete
    print("For", key, value)


if __name__ == '__main__':
    None

