from multiprocessing import Process, Manager, Pool
import os
import time

data = [i for i in range(10000)]
expected = [i**1000 for i in data]


def default():
    result = []
    for i in data:
        result.append(i**1000)

    assert set(result) == set(expected)


def worker(wid, numbers, manager_dict):
    # print(f"wid: {wid}")
    # print(numbers)
    manager_dict[wid] = [n**1000 for n in numbers]


def pool_worker(number):
    return number**1000


def basic_usage():
    print("starting basic process test")
    manager = Manager().dict()
    processes = []
    num_workers = 5

    # 0-19, 20
    for i in range(num_workers):
        start = i * (len(data) // num_workers)
        end = (i + 1) * (len(data) // num_workers)
        numbers = data[start:end]
        processes.append(Process(target=worker, args=(i, numbers, manager)))
    # if data size not divisible by num_workers, process the remainder
    if len(data) % num_workers != 0:
        start = (i + 1) * (len(data) // num_workers)
        end = len(data)
        numbers = data[start:end]
        processes.append(Process(target=worker, args=(num_workers, numbers, manager)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    result = []
    for e in manager.values():
        result += e
    # print(result)
    assert set(result) == set(expected)
    print("done")


def pool_usage():
    print("starting pool usage")
    num_workers = 5
    with Pool(processes=num_workers) as pool:
        result = pool.map(pool_worker, data)
    assert result == expected


if __name__ == "__main__":
    print("starting main process")

    start1 = time.process_time()
    start2 = time.time()
    default()
    end1 = time.process_time()
    end2 = time.time()
    print(f"default took {end1-start1} cpu seconds and {end2 - start2} clock seconds")

    start1 = time.process_time()
    start2 = time.time()
    basic_usage()
    end1 = time.process_time()
    end2 = time.time()
    print(f"basic took {end1-start1} cpu seconds and {end2 - start2} clock seconds")

    start1 = time.process_time()
    start2 = time.time()
    pool_usage()
    end1 = time.process_time()
    end2 = time.time()
    print(f"pool took {end1-start1} cpu seconds and {end2 - start2} clock seconds")
