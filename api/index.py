def handler(request, response):
    response.status = 200
    response.headers["Content-Type"] = "application/json"
    return {
        "status": "ok",
        "message": "Fashion MNIST deployment endpoint is live"
    }
