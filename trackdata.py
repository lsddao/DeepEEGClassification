import pymongo

class DBConnection:
    def __init__(self, session_id, collection_suffix):
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["muse"]
        col = db[session_id + "_" + collection_suffix]
        self.doc = col.find().sort("_id", pymongo.ASCENDING)