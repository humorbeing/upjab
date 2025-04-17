from shows.utils_show import Show

from shows.show_upjab import show_upjab1
from shows.show_upjab import clean_upjab1


show = Show()
show.add_show(show_upjab1)
show.add_clean(clean_upjab1)


if __name__ == "__main__":
    # show.start()
    show.clean()
