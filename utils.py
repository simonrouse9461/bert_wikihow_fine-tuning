class Logger:
    
    def __init__(self, file, 
                 log_fn=lambda x: x, 
                 aggre_fn=None, 
                 overwrite=False, 
                 period=1):
        self.file = file
        self.fout = None
        self.mode = 'w' if overwrite else 'a'
        self.log_fn = log_fn
        self.aggre_fn = aggre_fn
        self.period = period
        self.counter = 0
        self.history = []
        
    def __enter__(self):
        self.fout = open(self.file, self.mode)
        return self
        
    def __exit__(self, type, value, traceback):
        self.fout.close()
        
    def step(self, *args, **kwargs):
        self.counter += 1
        if self.aggre_fn is not None:
            self.history.append(self.log_fn(*args, **kwargs))
        if self.counter % self.period == 0:
            value = (self.log_fn(*args, **kwargs) 
                     if self.aggre_fn is None
                     else self.aggre_fn(self.history))
            print(value, file=self.fout, flush=True)
            self.history.clear()
