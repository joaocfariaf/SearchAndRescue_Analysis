import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
import numpy as np
import time

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 500])
data = arr.tolist()
data[:5]


def howmany_within_range(row, minimum=4, maximum=8):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

# start_time = time.time()
# results = []
# for row in data:
#     results.append(howmany_within_range(row, minimum=4, maximum=8))

# print(results[:10])
# end_time = time.time()
# total_time = (end_time - start_time)
# print('\n time1:', total_time)

# start_time = time.time()
# # Step 1: Init multiprocessing.Pool()
# pool = mp.Pool(mp.cpu_count())

# # Step 2: `pool.apply` the `howmany_within_range()`
# results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]

# # Step 3: Don't forget to close
# pool.close()    

# print(results[:10])
# end_time = time.time()
# total_time = end_time - start_time
# print('\n time2:', total_time)
# #> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]


start_time =  time.time()
pool = mp.Pool(mp.cpu_count())

results = pool.map(howmany_within_range, [row for row in data])

pool.close()

print(results[:10])
end_time = time.time()
total_time = end_time - start_time
print('\n time3:', total_time)


start_time = time.time()
pool = mp.Pool(mp.cpu_count())

results = pool.starmap(howmany_within_range, [(row, 4, 8) for row in data])

pool.close()

print(results[:10])
end_time = time.time()
total_time = end_time - start_time
print('\n time4:', total_time)


# Asyncronous
start_time = time.time()
pool = mp.Pool(mp.cpu_count())


results = pool.starmap_async(howmany_within_range, [(row, 4, 8) for row in data]).get()

# With map, use `howmany_within_range_rowonly` instead
# results = pool.map_async(howmany_within_range_rowonly, [row for row in data]).get()

pool.close()
print(results[:10])
end_time = time.time()
total_time = end_time - start_time
print('\n time7:', total_time)