from requests import get, post, put, delete, HTTPError
from pymongo import MongoClient

uri = "mongodb+srv://admin:bnoasojGXBwJqPUa@goemotion.04rdzbd.mongodb.net/?retryWrites=true&w=majority&appName=GoEmotion"
client = MongoClient(uri)
print(client.list_database_names())
