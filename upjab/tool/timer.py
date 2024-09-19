import time

class timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        
        print(f'Elapsed time: {self.interval:>20.2f} seconds')


if __name__ == '__main__':
    with timer() as t:
        time.sleep(1)
    print(f'Print Elapsed time: {t.interval:>20.2f} seconds')
    