import bz2
import json
import os

class DocStore:
    def __init__(self, dir):
        assert dir is not None
        assert os.path.isdir(dir)
        self.dir = dir

    def all(self):
        for root, dir, files in os.walk(self.dir):
            for file in files:
                path = os.path.join(root, file)
                yield self._read(path)

    def get(self, ids):
        assert ids is not None
        if isinstance(ids, str) or isinstance(ids, int):
            return self._get(ids)
        elif isinstance(ids, list):
            return [self._get(id) for id in ids]
        else:
            raise ValueError('Invalid ids: {}'.format(ids))

    def _get(self, id):
        assert id is not None
        path = os.path.join(self.dir, str(id) + '.bz2')
        if not os.path.exists(path):
            raise ValueError('id {} not found'.format(id))
        return self._read(path)

    def _read(self, path):
        with bz2.open(path) as f:
            return json.loads(f.read())
