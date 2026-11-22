from redis import Redis


class Redis_Cache(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = Redis()
        return cls._instance


if __name__ == "__main__":
    cache = Redis_Cache()
    cache1 = Redis_Cache()

    print(cache is cache1)
