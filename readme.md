# Deployment of Imagenet Classification Model

For now, predictions are made by uploading data into the data directory
- Run `model/run_app.py` to run or export pytorch model to torch script

# Query Model API
 - Send a get request with a valid image url to the endpoint `http://3.212.38.0/app?img_url=<url of image>&pred_num=<number of top predictions>`
 