import logging
import logging.handlers
import multiprocessing
import threading


class Logger:
    __logger_file = "quacc.log"
    __logger_name = "queue_logger"
    __queue = None
    __thread = None
    __setup = False

    @classmethod
    def __logger_listener(cls, q):
        while True:
            record = q.get()
            if record is None:
                break
            root = logging.getLogger()
            root.handle(record)

    @classmethod
    def setup(cls):
        if cls.__setup:
            return

        # setup root
        root = logging.getLogger()
        rh = logging.FileHandler(cls.__logger_file, mode="a")
        root.addHandler(rh)

        # setup logger
        if cls.__queue is None:
            cls.__queue = multiprocessing.Queue()

        logger = logging.getLogger(cls.__logger_name)
        logger.setLevel(logging.DEBUG)
        qh = logging.handlers.QueueHandler(cls.__queue)
        qh.setLevel(logging.DEBUG)
        qh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s| %(levelname)s: %(message)s",
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
    def join_listener(cls):
        if cls.__setup and cls.__thread is not None:
            cls.__thread.join()


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
        rh = logging.handlers.QueueHandler(q)
        rh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s| %(levelname)s: %(message)s",
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
