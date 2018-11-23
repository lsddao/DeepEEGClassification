import pymongo
from config import dbconnection

class DBConnection:
    def __init__(self):
        client = pymongo.MongoClient(dbconnection)
        self.db = client["muse"]

    def session_data(self, session_id, collection_suffix):
        col = self.db[session_id + "_" + collection_suffix]
        return col.find().sort("_id", pymongo.ASCENDING)

    def all_sessions(self):
        col = self.db["sessions"]
        all_sessions = set()
        for doc in col.find():
            all_sessions.add(doc['session_id'])
        return all_sessions