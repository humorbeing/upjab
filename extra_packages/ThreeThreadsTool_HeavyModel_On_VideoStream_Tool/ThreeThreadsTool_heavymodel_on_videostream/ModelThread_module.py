from threading import Thread


class ModelThread:    

    def __init__(self, frame_queue, detection_model):
        self.frame_queue = frame_queue        
        self.stopped = False
        self.results = {}        
        self.anomaly_detection = detection_model

    def start(self):    
        Thread(target=self.model_predict, args=()).start()
        return self

    def model_predict(self):        
        while not self.stopped:            
            if self.frame_queue.full():
                snippet = list(self.frame_queue.get())
                self.results = self.anomaly_detection(snippet)
                
                
            

    def stop(self):
        self.stopped = True
