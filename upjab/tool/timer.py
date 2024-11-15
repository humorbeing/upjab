import time

class timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        
        print(f'Elapsed time: {self.interval:>20.2f} seconds')

timethis = timer()

if __name__ == '__main__':
    with timer() as t:
        time.sleep(1)
    
    print(f'Print Elapsed time: {t.interval:>20.2f} seconds')
    
    with timethis as tt:
        time.sleep(2)
    
    print(f'Print Elapsed time: {timethis.interval:>20.2f} seconds')
    print(f'Print Elapsed time: {tt.interval:>20.2f} seconds')
    
    with timethis:
        time.sleep(1.5)
    print(f'Print Elapsed time: {timethis.interval:>20.2f} seconds')
    