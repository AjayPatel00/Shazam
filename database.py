import numpy as np


class Fingerprint:
    def __init__(self,song_id,h,offset):
        self.song_id = song_id
        self.h = h
        self.offset = offset
    
class Database:
    def __init__(self):
        self.db = []
        self.song_table = {}

    def add(self,fingerprint):
        self.db.append(fingerprint)

    def search(self, recording_data):
        # 
        # create a mapper, from hash value -> relative offset
        mapper = {}
        for h,offset in recording_data:
            if h in mapper.keys():
                mapper[h].append(offset)
            else:
                mapper[h] = [offset]


        recording_hashes = list(mapper.keys())
        songs = {}
        results = []
        for fingerprint in self.db:
            if fingerprint.h in recording_hashes:
                if fingerprint.song_id not in songs.keys():
                    songs[fingerprint.song_id] = 1
                else:
                    songs[fingerprint.song_id] += 1
                for recording_offset in mapper[fingerprint.h]:
                    results.append((fingerprint.song_id,fingerprint.offset - recording_offset))
        return results, songs
