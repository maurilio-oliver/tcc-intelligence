import json

from flask import Flask, request, jsonify
from service.recommender_service import get_recommender
app = Flask(__name__)



@app.put("/product/recommender/<int:cluster_number>")
def recommender_controller(cluster_number:int):  # put application's code here
    body = request.get_json()
    return get_recommender(body.get("options"), body.get("user"))


if __name__ == '__main__':
    app.run()
