class Logger:

    def __init__(self, file, 
                 map_fn=lambda x: x, 
                 reduce_fn=None, 
                 header=None,
                 overwrite=False, 
                 period=1):
        self.file = file
        self.fout = None
        self.mode = 'w' if overwrite else 'a'
        self.map_fn = map_fn
        self.reduce_fn = reduce_fn
        self.header = header
        self.period = period
        self.counter = 0
        self.history = []

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def open(self):
        self.fout = open(self.file, self.mode)
        if self.header is not None:
            self.log(self.header)

    def close(self):
        self.fout.close()

    def log(self, *msg, **kwargs):
        print(*msg, file=self.fout, flush=True, **kwargs)

    def step(self, *args, **kwargs):
        self.counter += 1
        if self.reduce_fn is not None:
            self.history.append(self.map_fn(*args, **kwargs))
        if self.counter % self.period == 0:
            value = (self.map_fn(*args, **kwargs) 
                     if self.reduce_fn is None
                     else self.reduce_fn(self.history))
            self.log(value)
            self.history.clear()
