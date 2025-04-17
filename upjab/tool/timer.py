import time

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapse = self.end - self.start
        
        print(f'Elapsed time: {self.elapse:>20.05f} seconds')


timer = Timer()

if __name__ == '__main__':


    with Timer() as t:
        time.sleep(1)
    
    print(f'Print Elapsed time: {t.elapse:>20.2f} seconds')


    timethis = Timer()

    with timethis as tt:
        time.sleep(2)
    
    print(f'Print Elapsed time: {timethis.elapse:>20.2f} seconds')
    print(f'Print Elapsed time: {tt.elapse:>20.2f} seconds')


    with Timer():
        time.sleep(2.5)
    

    with timer:
        time.sleep(1.5)

    
    with timer:
        time.sleep(1.2)
    
    