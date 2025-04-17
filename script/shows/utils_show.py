import time


class Show:
    def __init__(self):
        self.show_list = []
        self.clean_list = []

    def add_show(self, show):
        self.show_list.append(show)

    def add_clean(self, clean):
        self.clean_list.append(clean)

    def start(self, play_time=1_000, break_time=5):
        counter = 0
        while True:
            if counter >= play_time:
                break
            counter += 1
            print(">>> Upjab show started. <<<")
            for s in self.show_list:
                s()
                time.sleep(break_time)

            print("-->>> Upjab clean started. <<<--")
            for c in self.clean_list:
                c()
                time.sleep(break_time)

    def show(self, break_time=1):

        print(">>> Upjab show started. <<<")
        for s in self.show_list:
            s()
            time.sleep(break_time)

    def clean(self, break_time=1):

        print("-->>> Upjab clean started. <<<--")
        for c in self.clean_list:
            c()
            time.sleep(break_time)
