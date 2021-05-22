# Deployment of Imagenet Classification Model

For now, predictions are made by uploading data into the data directory
- Run `model/run_deloyed_app.py` to run or export pytorch model to onxx

## Running deployed model locally
- Model currently using uWSGI as web-server
- Change `app.run(host='0.0.0.0')` to `app.run(debug='True')` in `run_deployed_app.py` to use local dev server
- Build docker image and run docker container to run onnx model