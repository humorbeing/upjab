from threading import Thread


class ModelThread:    

    def __init__(self, frame_queue, detection_model):
        self.frame_queue = frame_queue        
        self.stopped = False
        self.results = {
            'Model Result': 'Awaiting...'
        }        
        self.anomaly_detection = detection_model

    def start(self):    
        Thread(target=self.model_predict, args=()).start()
        return self

    def model_predict(self):        
        while not self.stopped:            
            if self.frame_queue.full():
                snippet = list(self.frame_queue.get())
                self.results = self.anomaly_detection(snippet)
            else:
                
                self.results = {
                    'Model Result': f'{len(self.frame_queue)}/{self.frame_queue._max_size} Awaiting...'
                }
                
            

    def stop(self):
        self.stopped = True
