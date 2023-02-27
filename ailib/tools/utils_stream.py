
from tqdm.auto import tqdm

class MultiStreamReader():
    def __init__(self, stream_objs, skip_head=False, **kwargs):
        self._stream_objs = stream_objs
        self._skip_head = skip_head
        self._cur_obj = None

    def __iter__(self):
        for stream_obj in tqdm(self._stream_objs):
            for idx, line in enumerate(stream_obj):
                if self._skip_head and idx==0:
                    pass
                yield line

    def read(self, size=-1):
        if self._cur_obj is None:
            self._cur_obj = next(self._stream_objs)
        data = self._cur_obj.read(size)
        if len(data) == size:
            return data
        else:
             self._cur_obj = next( self._stream_objs)
        return data + self.read(size-len(data))
    
    def close(self):
        for stream_obj in self._stream_objs:
            if hasattr(stream_obj, 'close'):
                self.stream_obj.close()