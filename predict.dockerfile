
FROM trainer:latest
#ENTRYPOINT ["python", "-u", "./src/models/predict_model.py", "evaluate", "/models/checkpoint.pth", "/test.data"]

ENTRYPOINT ["python", "-u", "src/models/predict_model.py", "evaluate","/models/checkpoint.pth"]
