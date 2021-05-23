# Deployment of Imagenet Classification Model

For now, predictions are made by uploading data into the data directory
- Run `model/run_deloyed_app.py` to run or export pytorch model to onxx

# Query model
 - Send a get request with `img_url` as name to the endpoint `http://167.71.245.19:8080/app?img_url=<url of image>`