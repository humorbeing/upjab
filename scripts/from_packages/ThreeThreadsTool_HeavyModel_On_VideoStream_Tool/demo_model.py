from time import sleep
import random

compute_time = random.uniform(3.5, 7.5)

def detection_model_fn(video_snippet):
    pred = random.random()  # Simulate a model prediction
    # Simulate computation time
    
    real_compute_time = compute_time + random.uniform(-0.3, 0.3)
    fps = 1 / real_compute_time  # Calculate FPS based on compute time
    sleep(real_compute_time)  # Simulate processing time

    return {
        'Model Prediction': pred, 
        'Model Pred-Per-Second': fps,
        'Model Inference Time': real_compute_time
    }