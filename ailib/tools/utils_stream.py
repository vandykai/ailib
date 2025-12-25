
from tqdm.auto import tqdm

class MultiStreamReader():
    def __init__(self, stream_objs, skip_head=False, **kwargs):
        self._stream_objs = stream_objs
        self._skip_head = skip_head
        self._cur_obj = None
        self._quene = []
        self._should_skip_head = skip_head

    def __iter__(self):
        for stream_obj in tqdm(self._stream_objs):
            for _, line in enumerate(stream_obj):
                line = line.split(b'\n')
                self._quene.append(line[0])
                del line[0]
                while line:
                    #self._quene.append(b'\n')
                    if self._should_skip_head:
                        self._should_skip_head = False
                    else:
                        yield b''.join(self._quene)
                    self._quene.clear()
                    self._quene.append(line[0])
                    del line[0]
            if not self._should_skip_head and self._quene:
                yield b''.join(self._quene)

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