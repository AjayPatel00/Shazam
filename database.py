import numpy as np

# Fingerprint class holds 3 fields
# song_id: string, 
# h: for 2 peaks (t1,f1),(t2,f2), h is the 
#    first 20 bits of SHA256(f1,f2,t2-t1)
# offset: t1 (absolute offset from t=0)
class Fingerprint:
    def __init__(self,song_id,h,offset):
        self.song_id = song_id
        self.h = h
        self.offset = offset
    
# Database class holds all of our fingerprints
class Database:
    def __init__(self):
        # db stores all fingerprints
        self.db = []
        # song table: keys are song ids, and values are 
        # number of fingerprints for the song
        self.song_table = {}

    # add fingerprint to db
    def add(self,fingerprint):
        self.db.append(fingerprint)

    # perform linear search in database to find matching
    # hashes. Input is hashes generated ((h1,t1),(h2,t2),...)
    # from recording of a song
    def search(self, recording_data):
        # create a mapper, from hash value -> relative offset
        hash_to_offset = {}
        # turn list of tuples (recording_data) into dictionary
        # where tuple[0] maps to tuple[1]
        for h,offset in recording_data:
            if h in hash_to_offset.keys(): hash_to_offset[h].append(offset)
            else: hash_to_offset[h] = [offset]
        recording_hashes = list(hash_to_offset.keys())
        # matched_songs is dictionary where song_id will map to number of hash matches for
        # song_id. Results holds all (song_id, absolute offset - relative offset) pairs
        matched_songs,results = {},[]
        # perform linear search to find matching hash values
        for fingerprint in self.db:
            # if fingerprint in db is part of recording hashes
            if fingerprint.h in recording_hashes:
                # we have potential match so if we have already found matches for this
                # song, increase the matches counter, otherwise add song_id to songs
                if fingerprint.song_id not in matched_songs.keys(): matched_songs[fingerprint.song_id] = 1
                else: matched_songs[fingerprint.song_id] += 1
                # add offset difference to results
                for relative_offset in hash_to_offset[fingerprint.h]:
                    results.append((fingerprint.song_id,fingerprint.offset - relative_offset))
        return results, matched_songs
