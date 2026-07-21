import torch

CCDL_COMM_STREAM = None

class FakeHandle:
    def wait(self):
        pass

class Work: 
    def __init__(self, async_op = False):
        self.async_op = async_op
        if async_op:
            global CCDL_COMM_STREAM
            if CCDL_COMM_STREAM is None:
                CCDL_COMM_STREAM = torch.cuda.Stream(torch.cuda.current_device())
            if CCDL_COMM_STREAM.device.index != torch.cuda.current_device():
                self.stream = torch.cuda.Stream(torch.cuda.current_device())
            else:
                self.stream = CCDL_COMM_STREAM
        else:
            self.stream = torch.cuda.current_stream()

        self.stream_guard = torch.cuda.stream(self.stream)
        self.event = None

    def __enter__(self):
        self.stream.wait_stream(torch.cuda.current_stream())
        self.stream_guard.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream_guard.__exit__(exc_type, exc_value, traceback)
        self.event = torch.cuda.Event()
        self.event.record(self.stream)
        if not self.async_op:
            self.event.wait()

    def wait(self):
        if self.async_op:
            self.event.wait()
    
    def query(self):
        if self.async_op:
            return self.event.query()
        else:
            return True 



    