class Repo(object):
    def __init__(self):
        self._length = 0
        self._graph = []
        self.load_from_file()

    def load_from_file(self):
        f = open("input.txt", "r")
        lines = f.readlines()

        self._length = int(lines[0])
        for i in range(1, self._length + 1):
            self._graph.append([int(j.rstrip()) for j in lines[i].split(',')])


    def get_graph(self):
        return self._graph

    def get_length(self):
        return self._length