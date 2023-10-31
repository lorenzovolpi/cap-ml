import logging
import logging.handlers
import multiprocessing
import threading


class Logger:
    __logger_file = "quacc.log"
    __logger_name = "queue_logger"
    __manager = None
    __queue = None
    __thread = None
    __setup = False

    @classmethod
    def __logger_listener(cls, q):
        while True:
            record = q.get()
            if record is None:
                break
            root = logging.getLogger("listener")
            root.handle(record)

    @classmethod
    def setup(cls):
        if cls.__setup:
            return

        # setup root
        root = logging.getLogger("listener")
        root.setLevel(logging.DEBUG)
        rh = logging.FileHandler(cls.__logger_file, mode="a")
        rh.setLevel(logging.DEBUG)
        root.addHandler(rh)
        root.info("-" * 100)

        # setup logger
        if cls.__manager is None:
            cls.__manager = multiprocessing.Manager()

        if cls.__queue is None:
            cls.__queue = cls.__manager.Queue()

        logger = logging.getLogger(cls.__logger_name)
        logger.setLevel(logging.DEBUG)
        qh = logging.handlers.QueueHandler(cls.__queue)
        qh.setLevel(logging.DEBUG)
        qh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s| %(levelname)-8s %(message)s",
                datefmt="%d/%m/%y %H:%M:%S",
            )
        )
        logger.addHandler(qh)

        # start listener
        cls.__thread = threading.Thread(
            target=cls.__logger_listener,
            args=(cls.__queue,),
        )
        cls.__thread.start()

        cls.__setup = True

    @classmethod
    def queue(cls):
        if not cls.__setup:
            cls.setup()

        return cls.__queue

    @classmethod
    def logger(cls):
        if not cls.__setup:
            cls.setup()

        return logging.getLogger(cls.__logger_name)

    @classmethod
    def close(cls):
        if cls.__setup and cls.__thread is not None:
            cls.__queue.put(None)
            cls.__thread.join()
            # cls.__manager.close()


class SubLogger:
    __queue = None
    __setup = False

    @classmethod
    def setup(cls, q):
        if cls.__setup:
            return

        cls.__queue = q

        # setup root
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        rh = logging.handlers.QueueHandler(q)
        rh.setLevel(logging.DEBUG)
        rh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s| %(levelname)-8s\t%(message)s",
                datefmt="%d/%m/%y %H:%M:%S",
            )
        )
        root.addHandler(rh)

        cls.__setup = True

    @classmethod
    def logger(cls):
        if not cls.__setup:
            return None

        return logging.getLogger()
